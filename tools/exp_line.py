"""第一百七十七轮: 承接线是悬崖还是斜坡?

第 175 轮的反事实体检给出"缺口未达 702 笔均值 -0.606%/t=-4.12" —— 整体为负,
所以不放宽。但**整体为负 ≠ 每一档都为负**。若 -2.5%~-2.0% 这一档是正期望,
承接线就能从 -2.5 下移到 -2.0, 白捡一批成交日(覆盖率直接提升)。

做法: 对主策略每个"出票但缺口未达"的信号, 按假设成交算出 T+1开/T+2收 收益,
再按"实际缺口"分 0.25pp 细档, 看单调性与各档 t 值。
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

from src.config import load_config          # noqa: E402
from src.backtest import prepare_context    # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=1500)
    ap.add_argument("--out", default="output/exp_line.json")
    a = ap.parse_args()

    cfg = load_config()
    ctx = prepare_context(cfg, a.days)
    by_date, dates = ctx["by_date"], ctx["main_dates"]
    pos = {d: i for i, d in enumerate(dates)}

    bt = json.load(open(ROOT / "output" / "backtest.json"))
    rows = bt.get("trades_all") or []

    def px(df, code):
        m = df[df["ts_code"] == code]
        return None if m.empty else m.iloc[0]

    recs = []
    for r in rows:
        d, code = r["signal_date"], r["code"]
        i = pos.get(d)
        if i is None or i + 2 >= len(dates):
            continue
        bd, sd = dates[i + 1], dates[i + 2]
        rb, rs, r0 = px(by_date[bd], code), px(by_date[sd], code), px(by_date[d], code)
        if rb is None or rs is None or r0 is None:
            continue
        o, c = float(rb["open_adj"]), float(rs["close_adj"])
        pc = float(rb["prev_close_adj"])
        if o <= 0 or c <= 0 or pc <= 0:
            continue
        o_raw = float(rb["open"])
        if o_raw >= float(rb["high_limit"]) - 1e-9 or o_raw <= float(rb["low_limit"]) + 1e-9:
            continue
        gap = (o / pc - 1.0) * 100.0
        cost = (2.5 * 2 + 5 + 10) / 1e4
        ret = (c / o - 1.0 - cost) * 100.0
        recs.append({"signal_date": d, "code": code, "pool": r.get("pool"),
                     "rank": r.get("rank"), "gap": gap, "ret": ret,
                     "was_filled": r.get("ret_pct") is not None})

    df = pd.DataFrame(recs)
    print(f"可复算 {len(df)} 条 (原成交 {df.was_filled.sum()} 条)")
    m = df[df.pool == "main"].copy()

    def stat(sub):
        if len(sub) < 2:
            return f"n={len(sub)}"
        r = sub["ret"].to_numpy()
        t = r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))
        return (f"n={len(r):4d} 均值={r.mean():+.3f}% 中位={np.median(r):+.3f}% "
                f"胜率={(r > 0).mean() * 100:4.1f}% t={t:+5.2f}")

    print("\n=== 承接线附近 0.25pp 细档 (主策略) ===")
    edges = [-99, -3.5, -3.0, -2.75, -2.5, -2.25, -2.0, -1.75, -1.5,
             -1.0, -0.5, 0.0, 0.5, 99]
    for lo, hi in zip(edges[:-1], edges[1:]):
        sub = m[(m.gap >= lo) & (m.gap < hi)]
        print(f"[{lo:>6},{hi:>6}) {stat(sub)}")

    print("\n=== 假设把承接线下移到 N 档 (只取缺口 <= N 的未成交票) ===")
    for cut in (-2.5, -2.25, -2.0, -1.75, -1.5, -1.25, -1.0):
        add = m[(~m.was_filled) & (m.gap <= cut) & (m.gap > -99)]
        print(f"  承接线 {cut:>5}: 新增 {stat(add)}")

    print("\n=== 按年份 × 缺口(<=-2.5 已成交 vs -2.5~-1.5 未成交) ===")
    m["y"] = m.signal_date.str[:4]
    for y, g in m.groupby("y"):
        f = g[g.was_filled]
        near = g[(~g.was_filled) & (g.gap <= -1.5)]
        print(f"  {y} 成交 {stat(f)} | 近门槛 {stat(near)}")

    Path(ROOT / a.out).write_text(df.to_json(orient="records", force_ascii=False),
                                  encoding="utf-8")
    print(f"\n写出 {a.out}")


if __name__ == "__main__":
    main()
