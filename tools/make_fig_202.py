"""第 202 轮配图: 样本量 / 闸门 / 池深 / 面板与引擎的分歧."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
C1, C2, C3 = "#2c3e50", "#c0392b", "#2980b9"


def main() -> None:
    q = pd.read_parquet(OUT / "panel_dip_202.parquet")
    q["d"] = q["d"].astype(str)
    daily = pd.read_csv(OUT / "panel_daily_202.csv", dtype={"d": str}).set_index("d").sort_index()
    daily["nav"] = (1 + daily["ret"] / 100).cumprod()
    daily["gate"] = (daily["nav"] >= daily["nav"].rolling(20).mean()).astype(int)
    q = q.join(daily[["gate"]], on="d")
    q["pool"] = q.groupby("d")["ret"].transform("size")

    fig = plt.figure(figsize=(13.0, 8.6))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.26)

    # (a) 样本量
    ax = fig.add_subplot(gs[0, 0])
    labels = ["可用股票日\n(17 年面板)", "合格 dip 形态\n(本轮新样本)", "策略实际成交\n(3000 日)"]
    vals = [14_470_905, len(q), 717]
    cols = ["#bdc3c7", C3, C2]
    bars = ax.bar(labels, vals, color=cols, width=.58)
    ax.set_yscale("log")
    ax.set_ylim(300, 4e7)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.35, f"{v:,}", ha="center",
                fontsize=9.5, fontweight="bold")
    ax.set_ylabel("观测数（对数）", fontsize=9)
    ax.set_title("(a) 数据一直够用，是样本用得太少", fontsize=10, loc="left")
    ax.grid(axis="y", alpha=.2, lw=.6)
    ax.text(.40, .72, f"合格样本\n= 成交笔数的 {len(q)/717:.0f} 倍", transform=ax.transAxes,
            ha="left", va="center", fontsize=8.6, color=C3)

    # (b) 闸门价差分年
    ax = fig.add_subplot(gs[0, 1])
    q["y"] = q["d"].str[:4]
    g = q.groupby(["y", "gate"]).ret.mean().unstack()
    g["sp"] = g[1] - g[0]
    g = g[g.sp.notna()]
    xs = np.arange(len(g))
    ax.bar(xs, g.sp, color=[C3 if v > 0 else C2 for v in g.sp], width=.66)
    ax.axhline(0, color="#555", lw=.8)
    ax.axhline(g.sp.median(), color=C1, ls="--", lw=1)
    ax.set_ylim(-1.2, 6.9)
    ax.text(.1, 6.55, f"中位 +{g.sp.median():.2f}pp", ha="left", fontsize=8, color=C1)
    ax.set_xticks(xs)
    ax.set_xticklabels(g.index, rotation=45, fontsize=7.6)
    ax.set_ylabel("闸门内 − 闸门外（%/笔）", fontsize=9)
    ax.set_title("(b) 闸门价差：16 年里 15 年为正", fontsize=10, loc="left")
    ax.grid(axis="y", alpha=.2, lw=.6)

    # (c) 池深
    ax = fig.add_subplot(gs[1, 0])
    q["pb"] = pd.cut(q.pool, [0, 10, 20, 40, 80, 10 ** 6], labels=["≤10", "11-20", "21-40", "41-80", "80+"])
    piv = q.pivot_table(index="pb", columns="gate", values="ret", aggfunc="mean", observed=False)
    cnt = q.pivot_table(index="pb", columns="gate", values="ret", aggfunc="size", observed=False)
    xs = np.arange(len(piv))
    ax.bar(xs - .19, piv[1], width=.36, color=C3, label="闸门内")
    ax.bar(xs + .19, piv[0], width=.36, color="#d5b8b4", label="闸门外")
    for i, (a, b) in enumerate(zip(piv[1], piv[0])):
        ax.text(i - .19, a + .06, f"{a:.2f}", ha="center", fontsize=8, color=C3)
        ax.text(i + .19, b - .10, f"{b:.2f}", ha="center", va="top", fontsize=8, color="#8e6b66")
    ax.axhline(0, color="#555", lw=.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(piv.index, fontsize=8.5)
    ax.set_ylabel("单笔收益 %", fontsize=9)
    ax.set_ylim(-2.1, 1.95)
    ax.legend(fontsize=8, frameon=False, loc="lower left", ncol=2)
    ax.set_title("(c) 池越深越好，且闸门内才为正", fontsize=10, loc="left")
    ax.grid(axis="y", alpha=.2, lw=.6)

    # (d) 面板 vs 引擎
    ax = fig.add_subplot(gs[1, 1])
    proxy = json.load(open(OUT / "gate_proxy_202.json"))
    eng = {x["ma"]: x for x in json.load(open(OUT / "exp_gate_202_1500.json"))}
    base = json.load(open(OUT / "backtest.json"))["metrics"]
    eng.setdefault(20, {"年化%": base["年化收益率%"]})
    mas = [5, 8, 10, 12, 20]
    pv = [next(r["proxy_mean"] for r in proxy if r["ma"] == m) for m in mas]
    ev = [eng[m]["年化%"] for m in mas]
    ax.plot(range(len(mas)), pv, "o-", color=C3, lw=1.9, ms=6, label="面板：每笔均值 %")
    ax.set_ylabel("面板每笔均值 %", color=C3, fontsize=9)
    ax.tick_params(axis="y", colors=C3)
    ax.set_ylim(0.2, 1.35)
    ax2 = ax.twinx()
    ax2.plot(range(len(mas)), ev, "s--", color=C2, lw=1.9, ms=6, label="引擎：年化 %")
    ax2.set_ylabel("引擎年化 %", color=C2, fontsize=9)
    ax2.tick_params(axis="y", colors=C2)
    ax2.set_ylim(58, 132)
    for i, (a, b) in enumerate(zip(pv, ev)):
        ax.text(i, a + .05, f"{a:.2f}", ha="center", fontsize=7.8, color=C3)
        ax2.text(i, b + 3.6, f"{b:.0f}%", ha="center", fontsize=7.8, color=C2)
    ax.set_xticks(range(len(mas)))
    ax.set_xticklabels([f"MA{m}" for m in mas])
    ax.set_title("(d) 面板说越短越好，引擎说 MA20 最好", fontsize=10, loc="left")
    ax.grid(alpha=.2, lw=.6)

    fig.text(.012, .012,
             "样本：3824 个交易日 / 14,470,905 个股票日（2009-12 → 2026-11）；合格 dip 形态 42,327 个；"
             "引擎为 1500 日窗口。收益口径 T+1 开盘买 → T+2 收盘卖。",
             fontsize=8.2, color="#666")
    fig.savefig("/workspace/第202轮_数据不是不够是没用上.png", dpi=170, bbox_inches="tight")
    print("saved")


if __name__ == "__main__":
    main()
