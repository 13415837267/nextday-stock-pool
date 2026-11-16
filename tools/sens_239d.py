"""第二百三十九轮 D: B1 补位的仓位/只数敏感性 —— 49 笔的尾部机会该用多大仓位

B1 补位 = 大盘暴跌日(中位 -2.9%, 跌停 88 只)买"抗跌 + 高波动中下跌"的票,
2019+ 只有 49 次机会, 单笔 +2.22%, 但前 5 笔占 52% 贡献, 2026 年已转负。
这种"低频尾部"该给多大仓位? 本轮扫 仓位 x 只数, 看卡玛在哪最稳。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "output" / "feat234"
OUT = ROOT / "output"
TRADES = OUT / "w232_trades.csv"

COST = 0.10
COLS = ["ts_code", "trade_date", "y", "n1_ok", "pct", "rel_pct", "vol20",
        "is_paused", "is_st", "listed", "circ_mv"]


def load() -> pd.DataFrame:
    parts = []
    for p in sorted(FEAT.glob("f*.parquet")):
        d = pd.read_parquet(p, columns=COLS)
        d = d[(d["is_paused"] == 0) & (d["is_st"] == 0) & (d["n1_ok"] == True)]   # noqa: E712
        d = d[d["pct"].between(-11.5, 11.5) & d["y"].between(-40, 60) & (d["listed"] >= 250)]
        if len(d):
            parts.append(d)
    return pd.concat(parts, ignore_index=True)


def metrics(day_ret: pd.Series) -> tuple[float, float, float]:
    nav = (1 + day_ret / 100).cumprod()
    yrs = len(day_ret) / 244.0
    ann = nav.iloc[-1] ** (1 / yrs) - 1
    dd = (nav / nav.cummax() - 1).min()
    return round(ann * 100, 2), round(dd * 100, 2), round(ann * 100 / abs(dd * 100), 2)


def main() -> None:
    d = load()
    cal = np.sort(d["trade_date"].unique())
    cal = cal[cal >= 20190101]
    nxt = {cal[i]: cal[i + 1] for i in range(len(cal) - 1)}
    nxt[np.int64(cal[-1])] = np.int64(cal[-1])
    d = d[d["trade_date"].isin(cal)]

    tr = pd.read_csv(TRADES)
    tr = tr[tr["signal"].isin(cal)]
    base = pd.Series(0.0, index=cal)
    for dt, g in tr.groupby("signal"):
        base[dt] = g["ret"].mean()
    busy = set()
    for _, r in tr.iterrows():
        busy.add(r["buy"]); busy.add(r["sell"])

    s = d[(d["rel_pct"] > 1.26) & ((d["vol20"] * d["pct"]) < -4.37)]
    days = []
    for dt in s["trade_date"].unique():
        if dt in busy:
            continue
        bt, sl = nxt[dt], nxt[nxt[dt]]
        if bt in busy or sl in busy:
            continue
        days.append(dt)

    print(f"可补位 {len(days)} 日")
    rows = []
    for k in (1, 3, 5, 10):
        for w in (0.3, 0.5, 0.7, 1.0):
            vals = {}
            for dt in days:
                g = s[s["trade_date"] == dt].nsmallest(min(k, len(s[s["trade_date"] == dt])), "circ_mv")
                vals[dt] = g["y"].mean() - COST
            v = pd.Series(vals)
            sim = base.copy()
            sim[v.index] += v.values * w
            ann, dd, cml = metrics(sim)
            rows.append({"只数": k, "仓位": w, "单笔": round(v.mean(), 3),
                         "年化": ann, "回撤": dd, "卡玛": cml})
    r = pd.DataFrame(rows)
    print(r.to_string(index=False))
    r.to_csv(OUT / "w239d_sens.csv", index=False, encoding="utf-8-sig")
    print(f"\n基准 {metrics(base)}")
    best = r.sort_values("卡玛", ascending=False).iloc[0]
    print("卡玛最高:", dict(best))


if __name__ == "__main__":
    main()
