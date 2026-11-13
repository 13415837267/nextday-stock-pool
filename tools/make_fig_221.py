#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第 221 轮配图: 可转债 T+0 日内反转 + 与股票策略叠加"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

for fp in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
           "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"]:
    try:
        font_manager.fontManager.addfont(fp)
    except Exception:
        pass
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

RED = "#c62828"
GREEN = "#2e7d32"
BLUE = "#1565c0"
GRAY = "#78909c"
ORANGE = "#ef6c00"


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.6, 6.6),
                                   gridspec_kw={"width_ratios": [1.05, 1]})

    # ---------- (a) gap 分档 -> 日内收益 ----------
    t = pd.read_csv("/tmp/gap_bucket.csv")
    xs = np.arange(len(t))
    colors = [GREEN if v > 0 else RED for v in t["mu"]]
    bars = ax1.bar(xs, t["mu"], color=colors, width=0.68, alpha=0.88, zorder=3)
    ax1.axhline(0, color="#37474f", lw=1.2, zorder=4)
    for i, (v, n, wr) in enumerate(zip(t["mu"], t["n"], t["wr"])):
        ax1.text(i, v + (0.038 if v >= 0 else -0.038), f"{v:+.2f}",
                 ha="center", va="bottom" if v >= 0 else "top",
                 fontsize=11, fontweight="bold",
                 color=GREEN if v > 0 else RED, zorder=6)
        if abs(v) > 0.09:
            ax1.text(i, v * 0.5, f"{wr:.0f}%", ha="center", va="center",
                     fontsize=10, color="white", fontweight="bold", zorder=7)


    ax1.set_xticks(xs)
    ax1.set_xticklabels(t["lab"], rotation=32, ha="right", fontsize=9.4)
    ax1.set_ylabel("开盘买入、收盘卖出的平均收益（%）", fontsize=12)
    ax1.set_ylim(-0.60, 0.55)
    ax1.set_xlabel("T 日开盘缺口（相对前收盘）", fontsize=12, labelpad=8)
    ax1.grid(axis="y", ls=":", color="#cfd8dc", zorder=0)
    ax1.set_title("(a) 可转债日内反转：低开越深，修复越强（−1%~−2% 为甜点）",
                  fontsize=13.5, fontweight="bold", pad=12)
    ax1.text(0.985, 0.955, "柱内白字=日内胜率", transform=ax1.transAxes,
             ha="right", va="top", fontsize=9.0, color="#78909c")


    # 标注"接飞刀"区
    ax1.annotate("跌超 3% 变飞刀\n胜率跌破 50%", xy=(0.35, -0.10),
                 xytext=(1.35, -0.44), fontsize=9.4, color=RED,
                 arrowprops=dict(arrowstyle="->", color=RED, lw=1.4),
                 bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=RED, lw=1.0, alpha=0.92))

    # ---------- (b) 净值叠加 ----------
    n = pd.read_csv("/tmp/nav_stack.csv", parse_dates=["date"])
    n = n.sort_values("date")
    ax2.plot(n["date"], n["n_stk"], color=BLUE, lw=2.4, label="股票策略单独（T+2 单吊）")
    ax2.plot(n["date"], n["n_cb"], color=GRAY, lw=2.0, label="可转债 T+0 单独（日内）")
    ax2.plot(n["date"], n["n_combo"], color=RED, lw=2.9, label="叠加（空仓期做转债）")
    ax2.set_yscale("log")
    ax2.set_ylabel("净值（对数轴，起点=1）", fontsize=12)
    ax2.grid(True, which="both", ls=":", color="#cfd8dc", alpha=0.8)
    ax2.set_title("(b) 叠加是帕累托改进：股票 43.7% 时间在场，其余由转债填补",
                  fontsize=13.5, fontweight="bold", pad=12)

    # 终点标注
    last = n.iloc[-1]
    for col, color, lab, dy in [
        ("n_combo", RED, f"{last['n_combo']:.0f}×  年化 135.7%", 1.30),
        ("n_stk", BLUE, f"{last['n_stk']:.0f}×  年化 105.7%", 1.30),
        ("n_cb", GRAY, f"{last['n_cb']:.1f}×  年化 51.2%", 0.72),
    ]:
        ax2.annotate(lab, xy=(last["date"], last[col]),
                     xytext=(-6, int(12 * np.log(dy) / 0.26) if abs(np.log(dy)) > 0.1 else 12),
                     textcoords="offset points", ha="right",
                     fontsize=10.4, color=color, fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.9))

    ax2.legend(loc="upper left", fontsize=10.6, frameon=False)
    ax2.set_ylim(0.7, 400)

    fig.text(0.012, 0.008,
             "可转债 1062 只（含 740 只已退市，无幸存者偏差）· 2005-2026 · "
             "成本 0.10%/次（佣金双边+保守滑点）· 股票段为 1500 交易日实盘配置回测",
             fontsize=9.2, color="#546e7a")
    fig.tight_layout(rect=[0, 0.045, 1, 1])
    out = Path(__file__).resolve().parents[1] / "第221轮_可转债T0与叠加.png"
    fig.savefig(out, dpi=155, bbox_inches="tight", facecolor="white")
    print("saved", out)


if __name__ == "__main__":
    main()
