# -*- coding: utf-8 -*-
"""第一百八十八轮(配套): A 股日线的"涨跌幅 → 次日收益"结构曲线。

目的: 把"右侧为什么不行"从"某个策略回测不好"上升为"市场结构"。
      对每个交易日做横截面: 按当日涨跌幅把全市场分 10 组, 看每组
      "T+1 开盘买 → T+2 开盘卖"(= 右侧策略的最小结构单元) 的净收益。

如果这条曲线单调向下, 那么任何"追当日强势股"的策略在这个市场里都是逆风,
不需要逐个策略去证否; 反过来 dip(买当日弱势) 站在曲线的上端, 是顺风。

用法: python3 tools/exp_left_right_curve.py
输出: output/left_right_curve.json
"""
from __future__ import annotations

import glob
import json
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COST = 0.32


def load():
    fs = sorted(glob.glob(os.path.join(ROOT, "data", "panel", "*.csv.gz")))
    cols = ["ts_code", "trade_date", "open", "close", "pct", "is_paused", "high_limit"]
    fr = [pd.read_csv(f, dtype={"ts_code": str}, usecols=cols) for f in fs]
    px = pd.concat(fr, ignore_index=True)
    px["trade_date"] = px["trade_date"].astype(str).str[:8]
    M = {}
    for c in ["open", "close", "pct", "is_paused", "high_limit"]:
        M[c] = px.pivot_table(index="trade_date", columns="ts_code", values=c,
                              aggfunc="first").astype("float32")
    return M


def main() -> None:
    M = load()
    op, cl, pc, pa, hl = M["open"], M["close"], M["pct"], M["is_paused"], M["high_limit"]
    print(f"面板 {len(op)} 日 x {op.shape[1]} 只")

    # 右侧结构单元: T 日盘后按 pct 分组 → T+1 开盘买 → T+2 开盘卖
    r_next = (op.shift(-2) / op.shift(-1) - 1) * 100 - COST
    # 左侧变体: T+1 开盘买 → T+1 收盘卖(日内, 检验"隔夜跳空"占多少)
    r_intra = (cl.shift(-1) / op.shift(-1) - 1) * 100 - COST + 0.15  # 无隔夜印花税差异, 近似
    tradable = (op.shift(-1) > 0) & (pa.shift(-1) != 1) & (pa.shift(-2) != 1)
    buyable = ~(np.isfinite(hl.shift(-1)) & (op.shift(-1) >= hl.shift(-1) - 1e-6))

    P = pc.to_numpy()
    R = r_next.to_numpy()
    RI = r_intra.to_numpy()
    OK = (tradable & buyable).to_numpy()

    nb = 10
    acc = np.zeros((nb, 3))       # sum, sumsq, n
    acc_i = np.zeros((nb, 3))
    year_rows = []

    for t in range(len(P)):
        p = P[t]
        r = R[t]
        m = OK[t] & np.isfinite(p) & np.isfinite(r)
        n = int(m.sum())
        if n < 200:
            continue
        pv = p[m]
        rv = r[m]
        rvi = RI[t][m]
        # 横截面 10 等分(用分位数, 不用 qcut 排序, 更快)
        qs = np.nanquantile(pv, np.linspace(0, 1, nb + 1))
        qs[0], qs[-1] = -np.inf, np.inf
        idx = np.clip(np.searchsorted(qs, pv, side="right") - 1, 0, nb - 1)
        for b in range(nb):
            sel = idx == b
            k = int(sel.sum())
            if k == 0:
                continue
            acc[b, 0] += rv[sel].sum()
            acc[b, 1] += (rv[sel] ** 2).sum()
            acc[b, 2] += k
            acc_i[b, 0] += rvi[sel].sum()
            acc_i[b, 2] += k

        # 每年单独累计(用日期年份)
        year_rows.append((op.index[t][:4], pv, rv))

    print("\n" + "=" * 92)
    print("全市场横截面: 按当日涨跌幅分 10 组 → T+1 开盘买 → T+2 开盘卖 (扣成本 0.32%)")
    print("=" * 92)
    print(f"{'组':>3} {'当日涨幅均值':>11} {'单笔净收益':>10} {'胜率':>7} {'样本':>10}")
    curve = []
    for b in range(nb):
        n = acc[b, 2]
        if n == 0:
            continue
        mu = acc[b, 0] / n
        sd = np.sqrt(max(acc[b, 1] / n - mu ** 2, 0))
        win = 50 + 0  # 胜率需单独统计, 这里用正态近似不严谨, 下面重算
        curve.append((b, float(mu), int(n), float(sd)))
    # 胜率用日内口径重算不准, 单独再跑一遍: 用分组标记存 index 太重, 改用符号统计
    # 简化: 用 r>0 的比例需要原始值, 已在上面丢弃 → 采用第二次轻量遍历
    sign = np.zeros((nb, 2))  # 正收益笔数, 总笔数
    for t in range(len(P)):
        p = P[t]; r = R[t]
        m = OK[t] & np.isfinite(p) & np.isfinite(r)
        if int(m.sum()) < 200:
            continue
        pv = p[m]; rv = r[m]
        qs = np.nanquantile(pv, np.linspace(0, 1, nb + 1))
        qs[0], qs[-1] = -np.inf, np.inf
        idx = np.clip(np.searchsorted(qs, pv, side="right") - 1, 0, nb - 1)
        for b in range(nb):
            sel = idx == b
            sign[b, 0] += int((rv[sel] > 0).sum())
            sign[b, 1] += int(sel.sum())
        # 记录当日该组涨幅区间
    for b in range(nb):
        n = acc[b, 2]
        if n == 0:
            continue
        mu = acc[b, 0] / n
        win = sign[b, 0] / max(sign[b, 1], 1) * 100
        print(f"D{b + 1:<2d} {'':>11} {mu:+9.3f}% {win:6.1f}% {int(n):10d}")
    print("  (D1=当日跌幅最大 10%, D10=当日涨幅最大 10%)")

    print("\n" + "=" * 92)
    print("同一分组, 只持日内 (T+1 开盘买 → T+1 收盘卖), 看隔夜跳空占多少")
    print("=" * 92)
    for b in range(nb):
        n = acc_i[b, 2]
        if n == 0:
            continue
        print(f"D{b + 1:<2d} 日内 {acc_i[b, 0] / n:+7.3f}%   隔夜 {(acc[b, 0] - acc_i[b, 0]) / n:+7.3f}%")

    # 年度稳定性: D1 vs D10 的差
    print("\n" + "=" * 92)
    print("年度: D1(最弱) 与 D10(最强) 的次日收益, 及两者之差")
    print("=" * 92)
    by_year = {}
    for y, pv, rv in year_rows:
        qs = np.nanquantile(pv, [0.1, 0.9])
        lo = rv[pv <= qs[0]]
        hi = rv[pv > qs[1]]
        d = by_year.setdefault(y, [0, 0, 0, 0])
        d[0] += lo.sum(); d[1] += len(lo)
        d[2] += hi.sum(); d[3] += len(hi)
    for y in sorted(by_year):
        a = by_year[y]
        d1 = a[0] / max(a[1], 1)
        d10 = a[2] / max(a[3], 1)
        print(f"  {y}  D1 {d1:+6.3f}% (n={a[1]:6d})   D10 {d10:+6.3f}% (n={a[3]:6d})   差 {d1 - d10:+6.3f}pp")

    out = {
        "bucket": [{"q": b + 1, "ret": round(c[1], 4), "n": c[2]} for b, c in
                   [(x[0], x) for x in curve]],
        "note": "T日收盘后按当日涨跌幅横截面十分位 T+1开盘买 T+2开盘卖, 扣成本0.32%",
    }
    p = os.path.join(ROOT, "output", "left_right_curve.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"\n已写 {p}")


if __name__ == "__main__":
    main()
