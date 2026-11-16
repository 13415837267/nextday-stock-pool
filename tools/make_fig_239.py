"""第二百三十九轮配图: 样本内挖 vs 样本外挖, 以及唯一活下来的补位规则"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
FEAT = OUT / "feat234"
COST = 0.10
COLS = ["ts_code", "trade_date", "y", "n1_ok", "pct", "rel_pct", "vol20",
        "is_paused", "is_st", "listed", "circ_mv"]

SHORT = {
    "A1 反弹中继": "A1 反弹中继",
    "A2 急杀后首日反弹": "A2 急杀后首日反弹",
    "A3 暴跌日抗跌": "A3 暴跌日抗跌",
    "A4 抗跌+跌穿20日回撤": "A4 抗跌+跌穿回撤",
    "A5 抗跌+高波动下跌": "A5 抗跌+高波动",
    "A6 20日弱+10日涨": "A6 20日弱+10日涨",
    "B1 抗跌+高波动下跌": "B1 抗跌+高波动",
    "B2 20日线附近+10日强": "B2 20日线+10日强",
    "B3 5日弱+10日强": "B3 5日弱+10日强",
}


def nav_series():
    parts = []
    for p in sorted(FEAT.glob("f*.parquet")):
        d = pd.read_parquet(p, columns=COLS)
        d = d[(d["is_paused"] == 0) & (d["is_st"] == 0) & (d["n1_ok"] == True)]   # noqa: E712
        d = d[d["pct"].between(-11.5, 11.5) & d["y"].between(-40, 60) & (d["listed"] >= 250)]
        if len(d):
            parts.append(d[["trade_date", "y", "rel_pct", "vol20", "pct", "circ_mv"]])
    d = pd.concat(parts, ignore_index=True)
    cal = np.sort(d["trade_date"].unique())
    cal = cal[cal >= 20190101]
    nxt = {cal[i]: cal[i + 1] for i in range(len(cal) - 1)}
    nxt[np.int64(cal[-1])] = np.int64(cal[-1])
    d = d[d["trade_date"].isin(cal)]

    tr = pd.read_csv(OUT / "w232_trades.csv")
    tr = tr[tr["signal"].isin(cal)]
    base = pd.Series(0.0, index=cal)
    for dt, g in tr.groupby("signal"):
        base[dt] = g["ret"].mean()
    busy = set()
    for _, r in tr.iterrows():
        busy.add(r["buy"]); busy.add(r["sell"])

    s = d[(d["rel_pct"] > 1.26) & ((d["vol20"] * d["pct"]) < -4.37)]
    vals = {}
    for dt in s["trade_date"].unique():
        if dt in busy:
            continue
        bt, sl = nxt[dt], nxt[nxt[dt]]
        if bt in busy or sl in busy:
            continue
        g = s[s["trade_date"] == dt]
        vals[dt] = g.loc[g["circ_mv"].idxmin(), "y"] - COST
    sim = base.copy()
    v = pd.Series(vals)
    sim[v.index] = v.values
    return base, sim, v, cal


def main() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17.6, 6.4),
                             gridspec_kw={"width_ratios": [1.15, 1.0, 1.05]})
    a, b, c = axes

    # ---------- ① 样本内 vs 样本外 ----------
    d = pd.read_csv(OUT / "w239b_combo.csv")
    d["lab"] = d["条件"].map(lambda s: SHORT.get(s, s))
    d = d.sort_values(["组", "补位单笔"], ascending=[False, True])
    ys = np.arange(len(d))
    col = ["#95a5a6" if g.startswith("A") else "#c0392b" for g in d["组"]]
    a.barh(ys, d["补位单笔"], color=col, height=0.62)
    for y, v in zip(ys, d["补位单笔"]):
        a.text(v + (0.06 if v >= 0 else -0.06), y, f"{v:+.2f}", va="center",
               ha="left" if v >= 0 else "right", fontsize=9, color="#333")
    a.axvline(0, color="#333", lw=1)
    a.set_yticks(ys)
    a.set_yticklabels(d["lab"], fontsize=9)
    a.set_xlim(-0.75, 3.6)
    a.set_xlabel("补位单笔收益 %（T+1 开盘买 → T+2 收盘卖，已扣 0.1% 成本）", fontsize=9.5)
    a.set_title("① 同一段行情(2019+)，条件挖自哪里决定还剩多少\n"
                "灰=A 全史挖(对这段是样本内)　红=B 只用 2009~2018 挖(纯样本外)\n"
                "补位单笔均值：A 组 +1.52%　→　B 组 +0.70%（腰斩）",
                fontsize=10.8, pad=8)
    a.axhline(2.5, color="#999", ls="--", lw=1)
    a.grid(axis="x", ls=":", color="#ccc", alpha=0.6)
    a.set_axisbelow(True)

    # ---------- ② B1 分年 ----------
    p = pd.read_csv(OUT / "w239c_b1_picks.csv")
    p["yr"] = (p["signal"] // 10000).astype(int)
    g = p.groupby("yr")["y"].agg(["size", "sum"])
    xs = np.arange(len(g))
    colors = ["#c0392b" if v > 0 else "#27ae60" for v in g["sum"]]
    b.bar(xs, g["sum"], color=colors, width=0.62)
    for x, (s_, n_) in enumerate(zip(g["sum"], g["size"])):
        b.text(x, s_ + (1.8 if s_ >= 0 else -1.8), f"{s_:+.0f}", ha="center",
               va="bottom" if s_ >= 0 else "top", fontsize=9.5, fontweight="bold", color="#333")
        b.text(x, -3.6 if s_ >= 0 else 1.0, f"{n_}笔", ha="center",
               va="top" if s_ >= 0 else "bottom", fontsize=8, color="#777")
    b.axhline(0, color="#333", lw=1)
    b.set_xticks(xs)
    b.set_xticklabels([str(x) for x in g.index], fontsize=9)
    b.set_ylim(-8, 66)
    b.set_ylabel("该年补位累计贡献（百分点，按 100% 净值计）", fontsize=9.5)
    b.set_title("② 唯一活下来的补位规则：B1 抗跌+高波动中下跌\n"
                "8 年 49 次机会，正贡献 7 年，但前 5 笔占 52%、2026 已转负", fontsize=11, pad=8)
    b.grid(axis="y", ls=":", color="#ccc", alpha=0.6)
    b.set_axisbelow(True)
    b.text(len(g) - 0.4, 58, "2024 单年 +56", ha="right", fontsize=8.5, color="#c0392b",
           path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])

    # ---------- ③ 净值 ----------
    base, sim, v, cal = nav_series()
    nb = (1 + base / 100).cumprod()
    ns = (1 + sim / 100).cumprod()
    x = pd.to_datetime(pd.Series(cal).astype(str), format="%Y%m%d")
    c.plot(x, nb, color="#7f8c8d", lw=1.7, label=f"现行策略  年化 117% / 回撤 −24% / 卡玛 4.95")
    c.plot(x, ns, color="#c0392b", lw=1.9, label=f"现行 + B1 补位  年化 147% / 回撤 −24% / 卡玛 6.24")
    c.set_yscale("log")
    c.set_ylabel("净值（对数轴，起点 1）", fontsize=9.5)
    c.set_title("③ 补位只占空闲资金：49 天各买 1 只，回撤完全不变\n"
                "但 49 次里 5 次贡献一半，本质是暴跌日彩票，不是稳定增益", fontsize=11, pad=8)
    c.legend(fontsize=8.8, loc="upper left", framealpha=0.92)
    c.grid(ls=":", color="#ccc", alpha=0.6)
    c.set_axisbelow(True)
    c.tick_params(axis="x", labelsize=8.5)

    fig.text(0.5, 0.015,
             "数据 2019-01~2026-11（1891 交易日）｜B 组条件只用 2009~2018 数据挖掘、阈值原样照搬｜"
             "补位 = 基准空仓且资金不被占用时才买，满仓单只，扣 0.1% 往返成本",
             ha="center", fontsize=8.6, color="#666")
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(ROOT / "第239轮_样本外挖出来的条件.png", dpi=150, facecolor="white")
    print("saved")


if __name__ == "__main__":
    main()
