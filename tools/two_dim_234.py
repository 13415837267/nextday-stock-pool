"""第二百三十四轮: 二维交叉 —— dist_high20 是"新信息"还是"跌得多的另一种说法"

单变量显示 dist_high20(距20日最高点)是最强的信号(17/17 年同向, 极差1.18pp),
但它和 pct(当日跌幅) 天然相关: 今天大跌的票, 距前高当然也远。
所以必须问: 控制住 pct 之后, dist_high20 还有没有增量?

做法: pct 六档 x dist_high20 六档 的 36 格均值表。
  若增量存在 -> 同一列(pct固定)内各行不同 -> 是真的一维新信息
  若不存在   -> 只有对角线有值 -> 是同一个信号的两种写法, 不必加

同时做条件贡献分解: 从 817 万基准出发, 逐个套上现有策略的条件, 看均值怎么爬。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "output" / "feat234"


def load(scope: str = "lowopen") -> pd.DataFrame:
    cols = ["trade_date", "is_paused", "is_st", "y", "n1_ok", "pct", "listed",
            "dist_high20", "dist_high60", "bias20", "bias60", "circ_mv", "amp",
            "vr", "close_pos", "low_tail", "amt_yi", "turnover_rate", "ret20",
            "maxdd20", "intraday", "gap", "amt_ratio", "vol20"]
    parts = []
    for p in sorted(FEAT.glob("f*.parquet")):
        d = pd.read_parquet(p, columns=cols)
        d = d[(d["is_paused"] == 0) & (d["is_st"] == 0)]
        d = d[d["pct"].between(-11.5, 11.5)]
        d = d[d["y"].between(-40, 60)]
        d = d[d["listed"] >= 250]
        if scope != "all":
            d = d[d["n1_ok"] == True]      # noqa: E712
        if len(d):
            parts.append(d)
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    d = load()
    print(f"{len(d)} 样本, 基准 {d['y'].mean():+.3f}%", flush=True)
    d["year"] = (d["trade_date"] // 10000).astype(int)

    # ---------- 1. 二维: pct x dist_high20 ----------
    print("\n== 相关矩阵 ==", flush=True)
    cs = ["pct", "dist_high20", "dist_high60", "bias20", "bias60",
          "circ_mv", "amp", "vr", "close_pos", "low_tail"]
    print(d[cs].corr().round(2).to_string(), flush=True)

    pb = [-11.5, -7, -5, -3, -1, 1, 11.5]
    db = np.quantile(d["dist_high20"].dropna(), [0, .15, .3, .5, .7, .85, 1])
    db = np.unique(np.round(db, 3))
    d["pq"] = pd.cut(d["pct"], pb, labels=[f"{pb[i]:g}~{pb[i+1]:g}" for i in range(len(pb) - 1)])
    d["dq"] = pd.cut(d["dist_high20"], db, labels=[f"{db[i]:.0f}~{db[i+1]:.0f}" for i in range(len(db) - 1)])

    tab = d.pivot_table(index="dq", columns="pq", values="y", aggfunc="mean", observed=True)
    cnt = d.pivot_table(index="dq", columns="pq", values="y", aggfunc="size", observed=True)
    print("\n== 均值表: 行=距20日高点(%), 列=当日涨跌(%) ==", flush=True)
    print(tab.round(2).to_string(), flush=True)
    print("\n== 样本数 ==", flush=True)
    print(cnt.to_string(), flush=True)
    tab.to_csv(ROOT / "output" / "w234_2d_high20_pct.csv", encoding="utf-8-sig")
    cnt.to_csv(ROOT / "output" / "w234_2d_high20_pct_n.csv", encoding="utf-8-sig")

    # 固定 pct 列, 看行间极差(即"控制跌幅后 dist_high20 的增量")
    print("\n== 每一列(固定跌幅)内部的行间极差 ==", flush=True)
    rows = []
    for c in tab.columns:
        s = tab[c].dropna()
        if len(s) < 3:
            continue
        rows.append({"pct档": str(c), "最高": round(float(s.max()), 3),
                     "最低": round(float(s.min()), 3),
                     "行间极差": round(float(s.max() - s.min()), 3)})
    r = pd.DataFrame(rows)
    print(r.to_string(index=False), flush=True)
    r.to_csv(ROOT / "output" / "w234_2d_rowspread.csv", index=False, encoding="utf-8-sig")

    # ---------- 2. 条件贡献分解: 逐个套现有策略条件 ----------
    print("\n== 条件贡献分解(顺序累加, 只看均值爬到哪) ==", flush=True)
    steps = [
        ("基准(低开子集)", lambda x: x.index == x.index),
        ("+ 当日跌 3~6%", lambda x: x["pct"].between(-6, -3)),
        ("+ 收盘位置 <=0.32", lambda x: x["close_pos"] <= 0.32),
        ("+ bias20 <= 0", lambda x: x["bias20"] <= 0),
        ("+ 量比 >= 0.8", lambda x: x["vr"] >= 0.8),
        ("+ 成交额 >=0.8亿", lambda x: x["amt_yi"] >= 0.8),
        ("+ 换手 <=12%", lambda x: x["turnover_rate"] <= 12),
        ("+ 距20日高点 <= -8%", lambda x: x["dist_high20"] <= -8),
    ]
    cur = d
    prow = []
    for name, fn in steps:
        cur = cur[fn(cur)]
        prow.append({"步骤": name, "样本": len(cur),
                     "均值%": round(float(cur["y"].mean()), 3) if len(cur) else np.nan,
                     "胜率%": round(float((cur["y"] > 0).mean()) * 100, 1) if len(cur) else np.nan})
        print(f"{name:26s} n={len(cur):8d}  均值 {prow[-1]['均值%']}", flush=True)
    pd.DataFrame(prow).to_csv(ROOT / "output" / "w234_contrib.csv",
                              index=False, encoding="utf-8-sig")

    # ---------- 3. 每个条件的"独立贡献"(留一法) ----------
    print("\n== 留一法: 从完整条件里去掉某一条 ==", flush=True)
    full = [
        ("pct", lambda x: x["pct"].between(-6, -3)),
        ("close_pos", lambda x: x["close_pos"] <= 0.32),
        ("bias20", lambda x: x["bias20"] <= 0),
        ("vr", lambda x: x["vr"] >= 0.8),
        ("amt", lambda x: x["amt_yi"] >= 0.8),
        ("turnover", lambda x: x["turnover_rate"] <= 12),
    ]
    base = d
    for _, fn in full:
        base = base[fn(base)]
    b = float(base["y"].mean())
    lv = [{"去掉": "（完整）", "样本": len(base), "均值%": round(b, 3),
           "Δ%": 0.0}]
    print(f"{'（完整）':12s} n={len(base):6d} 均值 {b:+.3f}%", flush=True)
    for name, fn in full:
        sub = d
        for n2, f2 in full:
            if n2 == name:
                continue
            sub = sub[f2(sub)]
        m = float(sub["y"].mean())
        lv.append({"去掉": name, "样本": len(sub), "均值%": round(m, 3),
                   "Δ%": round(m - b, 3)})
        print(f"去掉 {name:12s} n={len(sub):6d} 均值 {m:+.3f}%  Δ {m-b:+.3f}pp", flush=True)
    pd.DataFrame(lv).to_csv(ROOT / "output" / "w234_leaveoneout.csv",
                            index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
