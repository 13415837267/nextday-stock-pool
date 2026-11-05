# -*- coding: utf-8 -*-
"""副策略换赛道的起点: 横截面因子扫描(第一百四十五轮)。

为什么先扫因子、不直接写策略
----------------------------
"换赛道"最容易犯的错是拍一个逻辑就回测, 32 笔样本下总能撞出一个好看的数字。
正确顺序是: 先确认**这个市场里哪些因子在横截面上真的有区分度**, 再拿它建策略。
本脚本只做一件事 —— 每 20 个交易日为一个调仓点, 把当日合格池按因子值分 5 组,
看未来 20 日收益是否单调、多空差多大、rank IC 多少。

口径(无未来函数)
----------------
· 调仓日 T 收盘算因子(只用 T 日及以前的数据) → T+1 开盘买入 → T+21 收盘卖出
· 合格池与副策略完全一致: 沪深主板 / 非ST / 非停牌 / 上市满250日 / 成交额≥1亿 / ivol>0
· 等权, 不扣成本(扫描阶段只看信号强弱, 成本留到策略回测)
· 分组用当日截面的分位(quintile), 组内等权

因子(全部是 T 日收盘可知的列, 不新造跨日指标)
--------------------------------------------
  mom3 / mom20      3 日、20 日动量(反转因子的镜像)
  bias20            收盘相对 MA20 的偏离
  ivol20            20 日特质波动(当前副策略用的因子)
  amount            成交额(小市值/低流动性的代理)
  turnover_rate     换手率(投机度)
  vol_ratio         量比(当日放量程度)

用法: python tools/exp_factor_scan.py [--days 1500] [--hold 20] [--json output/exp_factor_scan.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context  # noqa: E402
from src.config import board_of, load_config  # noqa: E402
from src.satellite import MIN_BARS, daily_params  # noqa: E402

FACTORS = ["mom3", "mom20", "bias20", "ivol20", "amount", "turnover_rate", "vol_ratio",
           "SYN 低波+小额", "SYN 低波+超跌", "SYN 低波+小额+超跌"]
SYNTH = {
    "SYN 低波+小额": lambda z: -z("ivol20") - z("amount"),
    "SYN 低波+超跌": lambda z: -z("ivol20") - z("bias20"),
    "SYN 低波+小额+超跌": lambda z: -z("ivol20") - z("amount") - z("bias20"),
}


def _pool(day, cfg):
    """与 src.satellite._masks 同一套条件(不 import 内部函数, 避免扫描被策略实现绑死)。"""
    P = daily_params(cfg)
    uni = cfg.get("universe") or {}
    boards = set(uni.get("boards") or ["main", "gem"])
    m = ((day["is_paused"] == 0) & (day["is_st"] == 0) & (day["bars"] >= MIN_BARS)
         & (day["amount"] >= P["amt_min"]) & (day["ivol20"] > 0))
    m &= day["ts_code"].map(lambda c: board_of(c) in boards).to_numpy()
    return day.loc[m]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=1500)
    ap.add_argument("--hold", type=int, default=20)
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    cfg = load_config()
    print(f"[ctx] 载入 {a.days} 日...", flush=True)
    ctx = prepare_context(cfg, a.days)
    by_date, dates = ctx["by_date"], ctx["main_dates"]
    print(f"[ctx] {len(dates)} 日 {dates[0]} ~ {dates[-1]}, 持有 {a.hold} 日", flush=True)

    acc = {f: [[] for _ in range(5)] for f in FACTORS}
    ics = {f: [] for f in FACTORS}
    n_periods = 0
    rows_per_period = []

    for i in range(0, len(dates) - a.hold - 1, a.hold):
        d0, d1 = dates[i + 1], dates[i + 1 + a.hold]
        if d0 not in by_date or d1 not in by_date:
            continue
        day = by_date[dates[i]]
        pool = _pool(day, cfg)
        if len(pool) < 100:          # 截面太小, 分位无意义
            continue
        px0 = by_date[d0][["ts_code", "open_adj"]]
        px1 = by_date[d1][["ts_code", "close_adj"]]
        fwd = px0.merge(px1, on="ts_code", how="inner")
        fwd["fwd"] = fwd["close_adj"] / fwd["open_adj"] - 1
        fwd = fwd.set_index("ts_code")["fwd"]
        df = pool.set_index("ts_code").join(fwd, how="inner").dropna(subset=["fwd"])
        if len(df) < 100:
            continue
        # 合成因子: 截面 z 分数后等权相加(每个调仓日重新标准化, 无跨期参数)
        for c in ("ivol20", "amount", "bias20", "mom20"):
            sd = float(df[c].std())
            df["_z_" + c] = (df[c] - df[c].mean()) / sd if sd else 0.0
        zz = lambda c: df["_z_" + c]        # noqa: E731
        for name, fn in SYNTH.items():
            df[name] = fn(zz)
        n_periods += 1
        rows_per_period.append(int(len(df)))
        for f in FACTORS:
            s = df[f].astype(float)
            if s.nunique() < 20 or s.isna().all():
                continue
            q = pd.qcut(s.rank(method="first"), 5, labels=False)
            for gi in range(5):
                sub = df.loc[q == gi, "fwd"]
                if len(sub):
                    acc[f][gi].append(float(sub.mean()))
            # Spearman = 两列各自 rank 后的 Pearson; 不走 scipy, 免得多装一个依赖
            ics[f].append(float(s.rank().corr(df["fwd"].rank())))

    out = []
    print(f"\n调仓点 {n_periods} 个, 每期平均 {np.mean(rows_per_period):.0f} 只\n", flush=True)
    print("%-14s %8s %8s %8s %8s %8s %9s %7s %s" % (
        "因子", "Q1低", "Q2", "Q3", "Q4", "Q5高", "Q5-Q1", "IC", "方向"))
    for f in FACTORS:
        g = [float(np.mean(x)) * 100 if x else float("nan") for x in acc[f]]
        spread = g[4] - g[0] if not (np.isnan(g[0]) or np.isnan(g[4])) else float("nan")
        ic = float(np.mean(ics[f])) if ics[f] else float("nan")
        # 单调性: Q1..Q5 相邻差分同号的占比
        diffs = np.diff([x for x in g if not np.isnan(x)])
        mono = float(np.mean(np.sign(diffs) == np.sign(diffs[0]))) if len(diffs) else 0.0
        direction = "越高越好" if spread > 0 else "越低越好"
        out.append({"factor": f, "q": [round(x, 3) for x in g],
                    "spread": round(spread, 3), "ic": round(ic, 4),
                    "mono": round(mono, 2), "n": n_periods})
        print("%-14s %7.2f%% %7.2f%% %7.2f%% %7.2f%% %7.2f%% %8.2f%% %7.3f %s(单调%.0f%%)"
              % (f, g[0], g[1], g[2], g[3], g[4], spread, ic, direction, mono * 100))

    if a.json:
        Path(a.json).write_text(json.dumps({"days": a.days, "hold": a.hold,
                                            "periods": n_periods, "factors": out},
                                           ensure_ascii=False, indent=1), encoding="utf-8")
        print("\n写出", a.json)


if __name__ == "__main__":
    main()
