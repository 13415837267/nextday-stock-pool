"""第247轮 D: 卖出规则 —— T+2 盘中挂限价单, 触及 +X% 就提前兑现

现在补位固定 "T+2 收盘卖"。但超跌反弹是脉冲式的: 一根大阳线里冲高 8% 然后收在 +2%
是常态, 收盘卖把这些浮盈全还回去了。

可执行的改法: T+1 开盘买入后, 立刻在 T+2 挂一张限价卖单, 价格 = 买入价 x (1+X%)。
  · 盘中触及 → 成交在 +X%(限价单的最坏情况就是正好 X%, 不乐观)
  · 全天没触及 → 收盘卖(和现在一样)
这不需要预测, 也不是"卖在最高点"的作弊 —— 限价单按自己挂的价格成交, 是真实订单。

X 从 3% 扫到 15%。如果 X=5% 能把单笔从 1.86% 拉到 2.5%, 那就是白捡的钱。
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
COST = 0.30


def metrics(r: pd.Series) -> tuple[float, float, float]:
    nav = (1 + r / 100).cumprod()
    ann = nav.iloc[-1] ** (244 / len(r)) - 1
    dd = (nav / nav.cummax() - 1).min()
    return round(ann * 100, 2), round(dd * 100, 2), round(ann * 100 / abs(dd * 100), 2)


def main() -> None:
    print("=" * 78)
    print("第247轮 D: T+2 盘中挂限价单 (触及 +X% 就卖)")
    print("=" * 78)

    pk = pd.read_csv(OUT / "w247b_N1.csv")
    pk["trade_date"] = pk["trade_date"].astype(int)
    codes = set(pk.ts_code.unique())
    print(f"待检验 {len(pk)} 笔 / {len(codes)} 只票  单笔 {(pk.y - COST).mean():.2f}%")

    # ---- 只读这 140 只票的行情 ----
    frames = []
    for f in sorted(glob.glob(str(ROOT / "data" / "panel" / "*.csv.gz"))):
        d = pd.read_csv(f, usecols=["ts_code", "trade_date", "open", "high", "close",
                                    "factor", "pct", "is_paused", "is_st"])
        d = d[d.ts_code.isin(codes)]
        # ★ 必须与 246b 完全同口径过滤后再 shift: 那边的 y = shift(-2) 是在
        #   过滤掉停牌/ST/异动行之后做的。不过滤会让 T+2 落到别的交易日,
        #   实测有 6 笔差 1~92pp(相关性只有 0.765)。
        d = d[(d["is_paused"] == 0) & (d["is_st"] == 0)]
        d = d[d["pct"].between(-11.5, 11.5)]
        if len(d):
            frames.append(d)
    q = pd.concat(frames, ignore_index=True)
    del frames
    q["ao"] = q["open"] * q["factor"]
    q["ah"] = q["high"] * q["factor"]
    q["ac"] = q["close"] * q["factor"]
    q = q.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    g = q.groupby("ts_code", sort=False)
    q["o1"] = g["ao"].shift(-1)                 # T+1 开盘(买入价)
    q["d2"] = g["trade_date"].shift(-2)         # T+2 交易日
    q["h2"] = g["ah"].shift(-2)
    q["c2"] = g["ac"].shift(-2)
    q["bars2"] = g.cumcount(ascending=False)    # 后面还剩几行
    q = q[q.bars2 >= 2]
    print(f"行情 {len(q):,} 行")

    m = pk.merge(q[["ts_code", "trade_date", "o1", "d2", "h2", "c2"]],
                 on=["ts_code", "trade_date"], how="left")
    miss = m.h2.isna().sum()
    print(f"对齐成功 {len(m) - miss} / {len(m)}"
          + (f"  (缺 {miss} 笔, 停牌/退市边界)" if miss else ""))

    m["y_close"] = (m.c2 / m.o1 - 1) * 100
    m["chk"] = m.y_close - m.y
    print(f"口径核对: y_close vs 原 y  平均差 {m.chk.mean():+.3f}pp  "
          f"最大绝对差 {m.chk.abs().max():.2f}pp  相关性 {m[['y_close','y']].corr().iloc[0,1]:.4f}")
    print(f"  → 差异大的笔数 (>0.5pp): {int((m.chk.abs() > 0.5).sum())}")

    print(f"\n【1】各 X 的效果 (全 256 笔)")
    print(f"{'X':>5}{'触及率':>8}{'单笔':>9}{'胜率':>7}{'合计':>9}{'年化':>9}{'回撤':>9}{'卡玛':>7}")
    base_dates = pd.Series(0.0, index=np.sort(m.trade_date.unique()))
    rows = []
    for X in (2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 999):
        touch = m.h2 >= m.o1 * (1 + X / 100)
        y = np.where(touch, X, m.y_close) - COST
        hit = touch.mean() * 100
        mu = y.mean()
        print(f"{X if X < 999 else '不挂':>5}{hit:>7.0f}%{mu:>9.2f}"
              f"{(y > 0).mean() * 100:>7.0f}{y.sum():>9.0f}", end="")
        # 只看破位路径的净值
        sim = pd.Series(0.0, index=np.arange(len(m)))
        br = m.brk.values
        print()
        rows.append({"X": X, "触及率": round(hit, 1), "单笔": round(mu, 3),
                     "胜率": round((y > 0).mean() * 100, 1), "合计": round(y.sum(), 0)})

    print(f"\n【2】分路径 (X=5%)")
    for path, lbl in ((m.brk, "破位"), (~m.brk, "深低开")):
        s = m[path]
        touch = s.h2 >= s.o1 * 1.05
        y0 = s.y_close - COST
        y1 = np.where(touch, 5, s.y_close) - COST
        print(f"  {lbl}: {len(s)} 笔  触及 {touch.mean()*100:.0f}%  "
              f"收盘卖 {y0.mean():.2f}%  →  挂5% {y1.mean():.2f}%  "
              f"(差 {y1.mean()-y0.mean():+.2f}pp)")

    print(f"\n【3】组合净值 (破位 40% + 深低开 40%)")
    tr = pd.read_csv(OUT / "w232_trades.csv")
    cal = sorted(int(f.split("/")[-1].replace(".csv.gz", ""))
                 for f in glob.glob(str(ROOT / "data" / "panel" / "*.csv.gz")))
    cal = np.array([d for d in cal if d >= tr.signal.min()])
    base = pd.Series(0.0, index=cal)
    for dt, gg in tr.groupby("signal"):
        base[int(dt)] = gg["ret"].mean()
    print(f"  仅主策略 {metrics(base)}")
    print(f"{'X':>5}{'年化':>9}{'回撤':>9}{'卡玛':>7}")
    for X in (3, 4, 5, 6, 7, 8, 10, 999):
        touch = m.h2 >= m.o1 * (1 + X / 100)
        y = np.where(touch, X, m.y_close) - COST
        sim = base.copy()
        v = pd.Series(y * 0.40, index=m.trade_date.values).sort_index()
        sim[v.index] = sim[v.index] + v.values
        a, dd, cm = metrics(sim)
        print(f"{X if X < 999 else '不挂':>5}{a:>9.2f}{dd:>9.2f}{cm:>7.2f}")

    m["touch5"] = m.h2 >= m.o1 * 1.05
    m.to_csv(OUT / "w247d_detail.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(rows).to_csv(OUT / "w247d_grid.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
