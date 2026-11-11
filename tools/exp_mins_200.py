# -*- coding: utf-8 -*-
"""第200轮 · 分钟级执行面: 买入时点 / 卖出时点 / 开盘形态

数据: tools/fetch_mins_200.py 抓的 output/mins_200/{code}_{date}.csv.gz
口径: 分钟价为**未复权**，但买卖两日之间只有极少数笔跨除权，且本脚本只做
      "同一笔内部的相对比例"(p_t / p_ref)，复权因子在日内与两日内不变，
      所以相对比例有效。绝对价不参与计算。

三件事:
  A. 买入日开盘 30 分钟的形态，对整笔收益有没有预测力
  B. 买入时点从 09:31 挪到 10:00/10:30/11:30/14:00，整笔收益怎么变
  C. 卖出时点从 14:55 挪到 10:00/10:30/11:30/13:00/14:00，整笔收益怎么变
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MINS = ROOT / "output" / "mins_200"


def punch(path: Path, day: str) -> dict | None:
    if not path.exists():
        return None
    x = pd.read_csv(path)
    if not len(x):
        return None
    x["t"] = x.trade_time.astype(str).str[-4:].astype(int)
    x = x.sort_values("t")
    o0 = float(x.open.iloc[0])

    def px(hhmm: int) -> float | None:
        s = x[x.t <= hhmm]
        return float(s.close.iloc[-1]) if len(s) else None

    def lo_hi(a: int, b: int):
        s = x[(x.t >= a) & (x.t <= b)]
        return (float(s.low.min()), float(s.high.max())) if len(s) else (None, None)

    lo30, hi30 = lo_hi(931, 1000)
    p1030 = px(1030)
    hi_d = float(x.high.max())
    return {"open": o0, "p1000": px(1000), "p1030": p1030, "p1130": px(1130),
            "p1300": px(1300), "p1400": px(1400), "p1455": px(1455), "p1500": px(1500),
            "lo_first30": lo30, "hi_first30": hi30, "lo_day": float(x.low.min()),
            "hi_day": hi_d}


def main():
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    t = pd.read_csv(ROOT / "output" / f"exp_factor_197_trades_{days}.csv",
                    dtype={"buy": str, "sell": str})
    t["buy"] = t.buy.str.replace("-", "")
    t["sell"] = t.sell.str.replace("-", "")
    rows = []
    for _, r in t.iterrows():
        b = punch(MINS / f"{r.code}_{r.buy}.csv.gz", r.buy)
        s = punch(MINS / f"{r.code}_{r.sell}.csv.gz", r.sell)
        if not b or not s:
            continue
        rows.append({"code": r.code, "buy": r.buy, "sell": r.sell, "ret": float(r.ret),
                     "b_open": b["open"], "b_p1000": b["p1000"], "b_p1030": b["p1030"],
                     "b_p1130": b["p1130"], "b_p1400": b["p1400"],
                     "b_lo30": b["lo_first30"], "b_hi30": b["hi_first30"], "b_lo": b["lo_day"],
                     "s_open": s["open"], "s_p1000": s["p1000"], "s_p1030": s["p1030"],
                     "s_p1130": s["p1130"], "s_p1300": s["p1300"], "s_p1400": s["p1400"],
                     "s_p1455": s["p1455"], "s_p1500": s["p1500"]})
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "output" / f"exp_mins_200_{days}.csv", index=False)
    print(f"{len(df)} 笔有完整分钟数据（原始 {len(t)} 笔）")

    # ---------- A. 买入日开盘形态 ----------
    df["r30"] = df.b_p1000 / df.b_open - 1
    df["dd30"] = df.b_lo30 / df.b_open - 1
    df["pos30"] = (df.b_p1000 - df.b_lo30) / (df.b_hi30 - df.b_lo30).replace(0, np.nan)
    df["day_ret"] = df.ret / 100
    print("\n===== A. 买入日开盘 30 分钟形态 -> 整笔收益 =====")
    for col in ("r30", "dd30", "pos30"):
        q = pd.qcut(df[col], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"], duplicates="drop")
        g = df.groupby(q, observed=False).ret.agg(["count", "mean", "median",
                                                   lambda s: (s > 0).mean() * 100])
        g.columns = ["笔数", "均值%", "中位%", "胜率%"]
        print(f"\n-- {col} 五分位 --")
        print(g.round(2).to_string())
        print("  相关系数 %.3f" % df[[col, "ret"]].corr().iloc[0, 1])

    # ---------- B. 买入时点 ----------
    print("\n===== B. 买入时点（整笔收益，成本不变）=====")
    out = {}
    base = float(np.prod(1 + df.ret / 100) ** (1 / len(df)) - 1)
    for col, lab in [("b_p1000", "10:00"), ("b_p1030", "10:30"), ("b_p1130", "11:30"),
                     ("b_p1400", "14:00")]:
        # 用该时点价买入：收益倍数 = (1+ret) * (open / p_t)
        mult = (1 + df.ret / 100) * (df.b_open / df[col])
        nr = (mult - 1) * 100
        geo = float(np.prod(mult) ** (1 / len(df)) - 1)
        win = float((mult > 1).mean() * 100)
        print(f"  {lab:>6}: 单笔几何 {geo*100:+.3f}%  (基线 {base*100:+.3f}%)  胜率 {win:.1f}%  "
              f"差价 {((df.b_open/df[col]-1)*100).mean():+.3f}%/笔")
        out[lab] = {"几何%": round(geo * 100, 3), "胜率%": round(win, 2),
                    "买价差%": round(float(((df.b_open / df[col] - 1) * 100).mean()), 3)}

    # ---------- C. 卖出时点 ----------
    print("\n===== C. 卖出时点（以 14:55 为基准）=====")
    out2 = {}
    for col, lab in [("s_p1000", "10:00"), ("s_p1030", "10:30"), ("s_p1130", "11:30"),
                     ("s_p1300", "13:00"), ("s_p1400", "14:00"), ("s_p1500", "15:00")]:
        ratio = df[col] / df.s_p1455
        mult = (1 + df.ret / 100) * ratio
        geo = float(np.prod(mult) ** (1 / len(df)) - 1)
        diff = float((ratio - 1).mean() * 100)
        print(f"  {lab:>6}: 单笔几何 {geo*100:+.3f}%  相对 14:55 卖 {diff:+.3f}%/笔  "
              f"好于基准的笔 {(ratio > 1).mean()*100:.0f}%")
        out2[lab] = {"几何%": round(geo * 100, 3), "相对%": round(diff, 3)}

    (ROOT / "output" / f"exp_mins_200_{days}.json").write_text(
        json.dumps({"A": {"n": len(df)}, "B": out, "C": out2}, ensure_ascii=False, indent=1),
        encoding="utf-8")


if __name__ == "__main__":
    main()
