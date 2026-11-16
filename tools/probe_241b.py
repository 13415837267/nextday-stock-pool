"""第 241 轮 B: W7 的钱到底是选股 alpha 还是熊市 beta

walk-forward 暴露: 固定阈值 2019~2026 共 144 笔, alpha 合计只有 +11.0pp
(144 笔摊下来 alpha 均值 +0.08%, 等于零)。而 2019 单年就 +29.6pp、2022/2024 各 -13~-14。
—— 那 W7 赚的就不是"选股", 是"破位日买什么都弹"。

本脚本三件事:
  ① 破位日(等权指数距 20 日线 <= -3.51) 每天到底有多少票可选, 等权全买是什么收益
  ② 超跌深度分档: beta 是否随跌幅加深
  ③ 破位日内部, 6 个选股键五档的 y 与 y_ex —— 有没有真能分出胜负的
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "output" / "feat234"
OUT = ROOT / "output"
COST = 0.10
COLS = ["ts_code", "trade_date", "y", "n1_ok", "pct", "rel_pct", "circ_mv", "idx_dist",
        "dist_high60", "turnover_rate", "amp", "vol20", "close_pos", "amt_yi",
        "is_paused", "is_st", "listed"]


def load() -> pd.DataFrame:
    parts = []
    for p in sorted(FEAT.glob("f*.parquet")):
        d = pd.read_parquet(p, columns=COLS)
        d = d[(d["is_paused"] == 0) & (d["is_st"] == 0) & (d["n1_ok"] == True)]      # noqa: E712
        d = d[d["pct"].between(-11.5, 11.5) & d["y"].between(-40, 60) & (d["listed"] >= 250)]
        d = d[d["ts_code"].str[:2].isin(["60", "00"])]
        if len(d):
            parts.append(d)
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    d = load()
    d["yr"] = d["trade_date"] // 10000
    d["y_ex"] = d["y"] - d.groupby("trade_date")["y"].transform("mean")
    brk = d[(d["idx_dist"] <= -3.51) & (d["yr"] >= 2019)].copy()
    print(f"破位日 {brk.trade_date.nunique()} 天  {len(brk):,} 个候选  "
          f"(平均每天 {len(brk) / brk.trade_date.nunique():.0f} 只)")
    print(f"破位日全市场等权 y 均值 {brk.groupby('trade_date')['y'].mean().mean():+.2f}%")

    # ① 每天不同买法
    print("\n" + "=" * 84)
    print("① 破位日每天怎么买（2019+，扣 0.10% 成本）")
    rows = []
    for k, tag in ((1, "最小市值1只"), (3, "最小市值3只"), (10, "最小市值10只"), (0, "等权全买")):
        v = {}
        for dt, g in brk.groupby("trade_date"):
            if k == 0:
                v[dt] = g["y"].mean() - COST
            else:
                v[dt] = g.nsmallest(min(k, len(g)), "circ_mv")["y"].mean() - COST
        s = pd.Series(v)
        nav = (1 + s / 100).cumprod()
        ann = nav.iloc[-1] ** (244 / len(s)) - 1
        dd = (nav / nav.cummax() - 1).min()
        rows.append({"买法": tag, "天数": len(s), "单笔": round(s.mean(), 3),
                     "胜率%": round((s > 0).mean() * 100, 1),
                     "年化(只看这些天)%": round(ann * 100, 2),
                     "最差": round(s.min(), 2)})
    print(pd.DataFrame(rows).to_string(index=False))

    # ② 深度分档
    print("\n" + "=" * 84)
    print("② 超跌深度分档（等权全买 / 最小市值1只）")
    bins = [-100, -8, -6, -5, -4.5, -3.51]
    labs = ["<=-8%", "-8~-6", "-6~-5", "-5~-4.5", "-4.5~-3.51"]
    brk["depth"] = pd.cut(brk["idx_dist"], bins=bins, labels=labs)
    out = []
    for lb, g in brk.groupby("depth", observed=True):
        eq = g.groupby("trade_date")["y"].mean() - COST
        mn = g.groupby("trade_date").apply(lambda x: x.nsmallest(1, "circ_mv")["y"].iloc[0]) - COST
        ex = g.groupby("trade_date")["y_ex"].mean()
        out.append({"深度": lb, "天数": g.trade_date.nunique(), "日均可选": round(len(g) / g.trade_date.nunique(), 0),
                    "等权单笔": round(eq.mean(), 2), "最小1只": round(mn.mean(), 2),
                    "alpha": round(ex.mean(), 2), "胜率%": round((eq > 0).mean() * 100, 0)})
    print(pd.DataFrame(out).to_string(index=False))

    # ③ 破位日内部选股键
    print("\n" + "=" * 84)
    print("③ 破位日内部，选股键五档（y = 含 beta, y_ex = 扣当日全市场均值）")
    KEYS = [("circ_mv", "流通市值"), ("turnover_rate", "换手率"), ("amp", "振幅"),
            ("vol20", "20日波动"), ("close_pos", "收盘位置"), ("rel_pct", "当日相对大盘"),
            ("dist_high60", "距60日高")]
    for k, name in KEYS:
        s = brk.dropna(subset=[k]).copy()
        s["q"] = pd.qcut(s[k], 5, labels=["Q1低", "Q2", "Q3", "Q4", "Q5高"], duplicates="drop")
        g = s.groupby("q", observed=True).agg(n=("y", "size"), y=("y", "mean"), y_ex=("y_ex", "mean"))
        spread = g["y"].iloc[-1] - g["y"].iloc[0]
        print(f"  {name:<8} y: " + " ".join(f"{x:+.2f}" for x in g["y"])
              + f"   |  y_ex: " + " ".join(f"{x:+.2f}" for x in g["y_ex"])
              + f"   |  首尾差 {spread:+.2f}pp")

    # ④ 破位日单笔分布：是"少数大反弹"还是"普遍小赚"
    v = brk.groupby("trade_date").apply(lambda x: x["y"].mean()).sort_values(ascending=False)
    tot = v.sum()
    print("\n" + "=" * 84)
    print(f"④ 破位日等权全买 {len(v)} 天合计 {tot:+.0f}pp  前5天占 {v.head(5).sum() / tot * 100:.0f}%  "
          f"前10天占 {v.head(10).sum() / tot * 100:.0f}%  中位 {v.median():+.2f}%")
    print("   最好的5天:", ", ".join(f"{i}({x:+.1f})" for i, x in v.head(5).items()))
    print("   最差的5天:", ", ".join(f"{i}({x:+.1f})" for i, x in v.tail(5).items()))

    brk[["trade_date", "ts_code", "y", "y_ex", "circ_mv", "pct", "idx_dist"]].to_csv(
        OUT / "w241b_brk.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
