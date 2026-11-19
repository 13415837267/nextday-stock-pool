"""第260轮 · 开盘执行的容量-成本前沿

第129轮把容量算成 `5% × 全天成交额` = 1250 万, 隐含假设是"一整天慢慢买, 且不影响价格"。
但真实执行是 T+1 开盘一笔进 —— 那一笔只占总成交额的百分之几。
本轮用 730 笔买入日的分钟线把这条前沿画出来: 愿意等多久, 就能装多少钱, 代价是多少溢价。
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MINS = ROOT / "output" / "mins_260"
WINS = [1, 3, 5, 10, 15, 30, 60, 120, 240]


def main():
    d = pickle.load(open(ROOT / "output" / "w258_skip_3200.pkl", "rb"))
    g = [x for x in d if x["name"].startswith("M ")][0]
    t = g["trades"]
    t = t[t.ret_pct.notna()].copy()
    t["key"] = t.code + "_" + t.buy_date

    rows = []
    miss = 0
    for r in t.itertuples():
        f = MINS / f"{r.key}.csv.gz"
        if not f.exists():
            miss += 1
            continue
        m = pd.read_csv(f)
        if not len(m):
            miss += 1
            continue
        m = m.sort_values("trade_time").reset_index(drop=True)
        amt = m.amount.astype(float).values
        px = m.close.astype(float).values
        vol = m.vol.astype(float).values
        tot = amt.sum()
        if tot <= 0 or vol.sum() <= 0:
            miss += 1
            continue
        op = float(m.open.iloc[0])   # 9:25 集合竞价成交价 = 回测的 buy_px
        rec = {"code": r.code, "buy_date": r.buy_date, "day_amt": tot,
               "ret_pct": r.ret_pct, "fill_path": r.fill_path, "pool": r.pool}
        for w in WINS:
            k = min(w, len(amt))
            a = amt[:k].sum()
            v = vol[:k].sum()
            vwap = (amt[:k].sum() / v) if v > 0 else op
            rec[f"share{w}"] = a / tot           # 该窗口成交额占全天比例
            rec[f"px{w}"] = vwap / op - 1        # 窗口 VWAP 相对开盘价的溢价
            rec[f"amt{w}"] = a                   # 窗口绝对成交额(元)
        rows.append(rec)

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "output" / "w260_exec.csv", index=False)
    print(f"样本 {len(df)} 笔 (缺失 {miss})  区间 {df.buy_date.min()}~{df.buy_date.max()}")
    print()

    print("=== A. 开盘窗口成交额占全天比例 (中位 / P25 / P75, %) ===")
    hdr = "窗口   " + "".join(f"{w:>10}" for w in WINS)
    print(hdr)
    for stat, fn in (("中位", "median"), ("P25", lambda s: s.quantile(.25)),
                     ("P75", lambda s: s.quantile(.75))):
        vals = [getattr(df[f"share{w}"], fn if isinstance(fn, str) else "__call__")() if fn == "median"
                else fn(df[f"share{w}"]) for w in WINS]
        print(f"{stat:>4}   " + "".join(f"{v * 100:>9.2f}%" for v in vals))
    print()

    print("=== B. 窗口 VWAP 相对开盘价溢价 (中位 / 均值, %) ===")
    print(hdr)
    for stat in ("median", "mean"):
        vals = [getattr(df[f"px{w}"], stat)() * 100 for w in WINS]
        print(f"{stat[:4]:>4}   " + "".join(f"{v:>9.2f}%" for v in vals))
    print()

    print("=== C. 容量前沿: 参与率 5%, 可装资金 = 5% x 窗口成交额 (中位 / P10, 万元) ===")
    print(hdr)
    for stat, fn in (("中位", lambda s: s.median()), ("P10", lambda s: s.quantile(.10))):
        vals = [fn(df[f"amt{w}"]) * 0.05 / 1e4 for w in WINS]
        print(f"{stat:>4}   " + "".join(f"{v:>9.1f}" for v in vals))
    print()

    print("=== D. 全天成交额口径 (129轮) 对照, 万元 ===")
    for stat, fn in (("中位", lambda s: s.median()), ("P10", lambda s: s.quantile(.10))):
        v = fn(df["day_amt"]) * 0.05 / 1e4
        print(f"  {stat} 全天 x5% = {v:.1f} 万")
    print()

    print("=== E. 若把单笔控制在 300 万, 需要的窗口成交额(参与率5%)与样本内可达比例 ===")
    need = 300e4 / 0.05
    for w in WINS:
        ok = (df[f"amt{w}"] >= need).mean()
        print(f"  前{w:>3}分钟: 窗口成交额中位 {df[f'amt{w}'].median()/1e4:>8.1f} 万, 达标(>=6000万) {ok*100:>5.1f}%")
    print()

    print("=== F. 执行延迟与单笔收益: 用延迟价重算 (毛口径, 未扣费) ===")
    base = 1.0
    print("  延迟到窗口 VWAP 成交后的单笔均值 (原始 %.3f%%)" % df.ret_pct.mean())
    for w in WINS:
        # 原收益 = sell / buy_open - 1; 若 buy 改为窗口 VWAP, 收益 = (1+r)*op/(op*(1+px)) - 1
        adj = ((1 + df.ret_pct / 100) / (1 + df[f"px{w}"]) - 1) * 100
        print(f"  前{w:>3}分钟 VWAP: {adj.mean():>6.3f}%  (Δ {adj.mean()-df.ret_pct.mean():>6.3f}pp)")


if __name__ == "__main__":
    main()
