#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第 220 轮配图: 622% 是怎么一层层掉到 2% 的"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.4, 6.6),
                                   gridspec_kw={"width_ratios": [1, 1.16]})

    # ---- (a) 成本敏感度 ----
    costs = [0.30, 1.00, 1.50, 2.00]
    ann = [622.4, 313.9, 177.4, 85.5]
    ax1.plot(costs, ann, "o-", color=RED, lw=2.6, ms=9, zorder=4)
    for x, y in zip(costs, ann):
        ax1.annotate(f"{y:.0f}%", (x, y), textcoords="offset points", xytext=(0, 14),
                     ha="center", fontsize=12, fontweight="bold", color=RED)
    ax1.axhline(75.10, color=BLUE, ls="--", lw=2.0, zorder=3)
    ax1.text(0.20, 96, "现行规则 75.1%", ha="left", va="bottom", fontsize=11,
             color=BLUE, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9))
    ax1.fill_between([0.3, 2.0], [75.1, 75.1], [700, 700], color="#ffcdd2", alpha=0.22, zorder=1)
    ax1.text(1.15, 380, "纸上收益区\n（成本 < 1% 才成立）", ha="center", fontsize=11,
             color="#b71c1c", alpha=0.85, linespacing=1.5)
    ax1.set_xlabel("双边交易成本（%）", fontsize=12.5)
    ax1.set_ylabel("年化收益率（%）", fontsize=12.5)
    ax1.set_xlim(0.15, 2.12)
    ax1.set_ylim(0, 700)
    ax1.grid(alpha=0.25, ls=":")
    ax1.set_title("(a) 成本每 +0.5pp，年化砍一半", fontsize=13.5, pad=12)

    # ---- (b) 约束叠加 ----
    labels = ["纸面\n0.3% 成本", "+ 现实成本\n1.0%", "+ 只买\n成交额>2亿",
              "+ 两条\n同时", "+ 大盘MA20\n上方"]
    vals = [622.4, 313.9, 118.2, 24.0, 2.1]
    mdds = [-63.3, -74.8, -93.2, -99.5, -96.7]
    cols = [RED, "#e53935", ORANGE, "#f9a825", GRAY]
    xs = np.arange(len(labels))
    bars = ax2.bar(xs, vals, color=cols, width=0.62, alpha=0.9, edgecolor="white", lw=1.4)
    for b, v, m in zip(bars, vals, mdds):
        cx = b.get_x() + b.get_width() / 2
        ax2.text(cx, v + 18, f"{v:.0f}%", ha="center", va="bottom", fontsize=13,
                 fontweight="bold", color="#b71c1c" if v > 100 else "#37474f", zorder=6,
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.92))
        if v > 120:
            ax2.text(cx, v * 0.45, f"回撤\n{m:.0f}%", ha="center", va="center",
                     fontsize=10, color="white", fontweight="bold", linespacing=1.4, zorder=6)
        else:
            ax2.text(cx, v + 72, f"回撤 {m:.0f}%", ha="center", va="bottom",
                     fontsize=9.6, color="#455a64", zorder=6,
                     bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.88))

    ax2.axhline(75.10, color=BLUE, ls="--", lw=2.0, zorder=3)
    ax2.text(-0.5, 106, "现行规则 75.1%", ha="left", va="bottom",
             fontsize=11, color=BLUE, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9))
    ax2.set_xticks(xs)
    ax2.set_xticklabels(labels, fontsize=10.2, linespacing=1.5)
    ax2.set_ylabel("年化收益率（%）", fontsize=12.5)
    ax2.set_ylim(0, 720)
    ax2.set_xlim(-0.55, len(labels) - 0.45)
    ax2.grid(axis="y", alpha=0.25, ls=":")
    ax2.set_title("(b) 加一条现实约束，收益就掉一个量级", fontsize=13.5, pad=12)

    fig.suptitle("第 220 轮 · 全历史训练：模型是真会选，但选的是「装不下钱」的票",
                 fontsize=16.5, fontweight="bold", y=0.985)
    fig.text(0.5, 0.922,
             "2015-2026 逐年滚动训练（训练 t-3~t-1，预测 t），单吊 T+1 开盘买 / T+2 收盘卖，960 笔",
             ha="center", fontsize=10.4, color="#546e7a")
    fig.tight_layout(rect=[0, 0.02, 1, 0.9])
    out = Path("第220轮_全历史训练的可执行性.png")
    fig.savefig(out, dpi=155, bbox_inches="tight", facecolor="white")
    print("saved", out)


if __name__ == "__main__":
    main()
