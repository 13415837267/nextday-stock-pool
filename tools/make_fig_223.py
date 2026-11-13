#!/usr/bin/env python3
"""第223轮: 近半年实测 —— 策略 vs 你。红涨绿跌(中国市场惯例)。"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130

UP, DOWN = "#d62728", "#2ca02c"   # 中国惯例: 涨红跌绿
GRAY, BLUE, ORANGE = "#8c8c8c", "#1f4e79", "#e07b39"


def main() -> None:
    r = json.load(open(ROOT / "output" / "backtest.json"))
    eq = pd.DataFrame(r["equity"])
    bm = pd.DataFrame(r["benchmark"])
    eq["d"] = pd.to_datetime(eq["d"])
    bm["d"] = pd.to_datetime(bm["d"])
    n0 = eq["nav"].iloc[0]
    eq["v"] = eq["nav"] / n0 * 100
    bm["v"] = bm["nav"] / bm["nav"].iloc[0] * 100

    tr = pd.DataFrame(r["trades_executed_recent"])
    tr["ret_pct"] = pd.to_numeric(tr["ret_pct"], errors="coerce")
    tr["buy_date"] = pd.to_datetime(tr["buy_date"], format="%Y%m%d")

    mo = pd.DataFrame(r["monthly"])
    mo["ret_pct"] = pd.to_numeric(mo["ret_pct"], errors="coerce")

    fig = plt.figure(figsize=(15.6, 5.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.55, 1.25, 1.0], wspace=0.26)

    # ---------- ① 净值 ----------
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(eq["d"], eq["v"], color=BLUE, lw=2.4, label="这套策略", zorder=3)
    ax.plot(bm["d"], bm["v"], color=GRAY, lw=1.8, ls="--", label="全市场等权", zorder=2)
    ax.fill_between(eq["d"], 100, eq["v"], color=BLUE, alpha=0.10)
    ax.axhline(100, color="k", lw=0.8, alpha=0.4)
    ax.annotate(f"+{eq['v'].iloc[-1]-100:.1f}%", xy=(eq["d"].iloc[-1], eq["v"].iloc[-1]),
                xytext=(-6, 10), textcoords="offset points", color=UP,
                fontsize=13, fontweight="bold", ha="right")
    ax.annotate(f"{bm['v'].iloc[-1]-100:.1f}%", xy=(bm["d"].iloc[-1], bm["v"].iloc[-1]),
                xytext=(-6, -16), textcoords="offset points", color=DOWN,
                fontsize=11, fontweight="bold", ha="right")
    ax.set_ylabel("净值（2026-05-07 = 100）", fontsize=9)
    ax.set_title("① 同一段时间：策略 +60.7% vs 市场 −19.2%", fontsize=10.5, pad=8)
    ax.legend(fontsize=8.5, loc="upper left", framealpha=0.9)
    ax.grid(alpha=0.25)
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m月"))
    ax.tick_params(labelsize=8)

    # ---------- ② 逐笔 ----------
    ax = fig.add_subplot(gs[0, 1])
    x = np.arange(len(tr))
    cols = [UP if v > 0 else DOWN for v in tr["ret_pct"]]
    ax.bar(x, tr["ret_pct"], color=cols, width=0.68, edgecolor="k", lw=0.4, alpha=0.88)
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(tr["ret_pct"].mean(), color=ORANGE, ls="--", lw=1.2,
               label=f"均值 {tr['ret_pct'].mean():+.2f}%")
    ax.legend(fontsize=8, loc="lower right", framealpha=0.92)
    ax.set_xticks(x[::2], [f"{i+1}" for i in range(len(tr))][::2], fontsize=7)
    ax.set_xlabel("第 n 笔（共 21 笔，7 个月）", fontsize=9)
    ax.set_ylabel("单笔收益 (%)", fontsize=9)
    ax.set_title("② 21 笔明细：14 胜 7 负，最大 +18.1% / −7.0%", fontsize=10.5, pad=8)
    ax.grid(axis="y", alpha=0.22)
    ax.tick_params(labelsize=8)

    # ---------- ③ 月度 ----------
    ax = fig.add_subplot(gs[0, 2])
    lbl = [f"{int(str(m)[-2:])}月" for m in mo["month"]]
    cols = [UP if v > 0 else DOWN for v in mo["ret_pct"]]
    bars = ax.bar(range(len(mo)), mo["ret_pct"], color=cols, width=0.62,
                  edgecolor="k", lw=0.4, alpha=0.88)
    for b, v in zip(bars, mo["ret_pct"]):
        ax.text(b.get_x() + b.get_width() / 2, v + (0.7 if v >= 0 else -1.6),
                f"{v:+.1f}", ha="center", fontsize=7.5,
                color=UP if v > 0 else DOWN, fontweight="bold")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(mo)), lbl, fontsize=8)
    ax.set_ylabel("月度收益 (%)", fontsize=9)
    ax.set_title("③ 7 个月里 6 个月为正", fontsize=10.5, pad=8)
    ax.grid(axis="y", alpha=0.22)
    ax.set_ylim(min(mo["ret_pct"]) - 5, max(mo["ret_pct"]) + 5)
    ax.tick_params(labelsize=8)

    fig.suptitle("第223轮：你亏 75% 的那半年，这套策略赚 60.7% —— 差距在执行，不在选股",
                 fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0.01, 1, 0.93])
    out = ROOT / "第223轮_同期策略与你.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print("saved", out)


if __name__ == "__main__":
    main()
