"""第260轮b · 卖出端容量与成本

买入端已实测: 首分钟占全天 2.78%, 开盘一笔容量 33 万。
这里量卖出端: 收盘集合竞价(最后 1/3 分钟)能吃下多少, 提前卖要付多少价。
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MINS = ROOT / "output" / "mins_260s"
WINS = [1, 3, 5, 15, 30, 60, 120, 240]


def main():
    d = pickle.load(open(ROOT / "output" / "w258_skip_3200.pkl", "rb"))
    g = [x for x in d if x["name"].startswith("M ")][0]
    t = g["trades"]
    t = t[t.ret_pct.notna()].copy()
    t["key"] = t.code + "_" + t.sell_date

    rows = []
    for r in t.itertuples():
        f = MINS / f"{r.key}.csv.gz"
        if not f.exists():
            continue
        m = pd.read_csv(f)
        if not len(m):
            continue
        m = m.sort_values("trade_time").reset_index(drop=True)
        amt = m.amount.astype(float).values
        px = m.close.astype(float).values
        vol = m.vol.astype(float).values
        tot = amt.sum()
        if tot <= 0:
            continue
        close_px = px[-1]
        rec = {"code": r.code, "sell_date": r.sell_date, "day_amt": tot, "ret_pct": r.ret_pct}
        for w in WINS:
            k = min(w, len(amt))
            a = amt[-k:].sum()
            v = vol[-k:].sum()
            vwap = a / v if v > 0 else close_px
            rec[f"share{w}"] = a / tot
            rec[f"px{w}"] = vwap / close_px - 1     # 该窗口 VWAP 相对收盘价的偏离
            rec[f"amt{w}"] = a
        rows.append(rec)

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "output" / "w260s_sell.csv", index=False)
    print(f"样本 {len(df)} 笔 / 成交 {len(t)} 笔")
    print()

    print("=== A. 收盘窗口成交额占全天比例 (中位 / P25 / P75, %) ===")
    hdr = "窗口(最后)   " + "".join(f"{w:>9}" for w in WINS)
    print(hdr)
    for stat in ("median", lambda s: s.quantile(.25), lambda s: s.quantile(.75)):
        nm = stat if isinstance(stat, str) else ("P25" if not hasattr(stat, "_n") else "P75")
        vals = [getattr(df[f"share{w}"], stat)() if isinstance(stat, str) else stat(df[f"share{w}"]) for w in WINS]
        print(f"{str(nm):>10}   " + "".join(f"{v*100:>8.2f}%" for v in vals))
    print()

    print("=== B. 窗口 VWAP 相对收盘价 (中位 / 均值, %) —— 负=开盘前卖更便宜 ===")
    print(hdr)
    for stat in ("median", "mean"):
        vals = [getattr(df[f"px{w}"], stat)() * 100 for w in WINS]
        print(f"{stat[:4]:>10}   " + "".join(f"{v:>8.2f}%" for v in vals))
    print()

    print("=== C. 卖出容量: 5% 参与率, 可卖资金 (中位 / P10, 万元) ===")
    print(hdr)
    for nm, fn in (("中位", lambda s: s.median()), ("P10", lambda s: s.quantile(.10))):
        vals = [fn(df[f"amt{w}"]) * 0.05 / 1e4 for w in WINS]
        print(f"{nm:>10}   " + "".join(f"{v:>8.1f}" for v in vals))
    print()

    print("=== D. 若在收盘前 N 分钟开始卖, 单笔收益的变化 (毛口径) ===")
    base = df.ret_pct.mean()
    print(f"  基准(按收盘价卖) {base:.3f}%")
    for w in WINS:
        # 卖价改为窗口 VWAP: 收益 = (1+r)*close/(close*(1+px)) - 1
        adj = ((1 + df.ret_pct / 100) / (1 + df[f"px{w}"]) - 1) * 100
        print(f"  最后{w:>3}分钟 VWAP: {adj.mean():>6.3f}%  (Δ {adj.mean()-base:>6.3f}pp)")


if __name__ == "__main__":
    main()
