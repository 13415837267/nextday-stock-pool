"""第256轮: 补位改漏斗 —— 三种形态的价格 + 名次分布去了哪

① 2500 日：A 三选一 / B 漏斗三层 / C 只补一只
② 破位腿按名次：A 靠第 2 名的 7 笔 7.08% 撑均值，B 是第 1 层 18 笔 2.58%
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

RED, DARK, GREY, LIGHT, BLUE, GREEN = "#c0392b", "#2c3e50", "#95a5a6", "#d5dbdf", "#2e6da4", "#27ae60"
plt.rcParams["axes.unicode_minus"] = False
for fp in ("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
           "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
           "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"):
    if Path(fp).exists():
        font_manager.fontManager.addfont(fp)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()
        break
plt.rcParams["figure.dpi"] = 130

NAMES = ["A 三选一（现状）\n回撤 −17.2%", "B 漏斗三层（本轮）\n回撤 −17.2%", "C 只补一只\n回撤 −20.0%"]
ANN = [96.38, 95.20, 94.37]
DD = [17.21, 17.21, 20.03]
CAL = [5.60, 5.53, 4.71]
N_FILL = [50, 50, 31]
AVG_FILL = [1.71, 1.30, 1.75]

# 破位腿: 名次 -> (笔数, 单笔)
A_RK = {1: (10, 2.16), 2: (7, 7.08), 3: (10, -0.36)}
B_RK = {1: (18, 2.58), 2: (4, 0.82), 3: (5, -0.49)}


def fig_main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 5.0),
                                   gridspec_kw={"width_ratios": [1, 1.15]})

    x = np.arange(3)
    cols = [GREY, GREEN, "#8e44ad"]
    bars = ax1.bar(x, ANN, color=cols, width=0.56, zorder=2)
    for i, (b, a, c, n) in enumerate(zip(bars, ANN, CAL, N_FILL)):
        ax1.text(b.get_x() + b.get_width() / 2, a + 1.4, f"{a:.2f}%", ha="center",
                 va="bottom", fontsize=11.5, fontweight="bold", color=DARK)
        ax1.text(b.get_x() + b.get_width() / 2, 10, f"卡玛 {c:.2f}", ha="center",
                 va="bottom", fontsize=9, color="white")
        ax1.text(b.get_x() + b.get_width() / 2, 3, f"补位 {n} 笔", ha="center",
                 va="bottom", fontsize=8.6, color="white")
    ax1.axhline(96.38, color=BLUE, ls=":", lw=1.1)
    ax1.set_xlim(-0.62, 2.98)
    ax1.text(2.56, 96.38, "现状\n96.38%", fontsize=8.2, color=BLUE, va="center", ha="left")
    ax1.set_xticks(x)
    ax1.set_xticklabels(NAMES, fontsize=9.2)
    ax1.set_ylabel("年化收益率 %（2500 日）", fontsize=10)
    ax1.set_ylim(0, 108)
    ax1.set_title("只掉 1.2pp，但决策从「三选一」变成「一层一只」", fontsize=11.5,
                  color=DARK, loc="left", pad=10, fontweight="bold")
    ax1.grid(axis="y", color=LIGHT, lw=0.8, zorder=0)
    for s in ("top", "right"):
        ax1.spines[s].set_visible(False)

    w = 0.36
    for j, (lab, d, col) in enumerate([("A 三选一", A_RK, GREY), ("B 漏斗三层", B_RK, GREEN)]):
        xs = np.arange(1, 4) + (j - 0.5) * w
        ns = [d[k][0] for k in (1, 2, 3)]
        av = [d[k][1] for k in (1, 2, 3)]
        ax2.bar(xs, ns, width=w, color=col, label=lab, zorder=2)
        for xi, n, a in zip(xs, ns, av):
            ax2.text(xi, n + 0.45, f"{n}笔\n{a:+.2f}%", ha="center", va="bottom",
                     fontsize=8.4, color=DARK)
    ax2.set_xticks([1, 2, 3])
    ax2.set_xticklabels(["第 1 层", "第 2 层", "第 3 层"], fontsize=10)
    ax2.set_ylabel("破位腿成交笔数（2500 日，共 27 笔）", fontsize=10)
    ax2.set_ylim(0, 24)
    ax2.legend(fontsize=9, frameon=False, loc="upper right")
    ax2.set_title("漏斗真的在用第 2/3 层：18 / 4 / 5", fontsize=11.5,
                  color=DARK, loc="left", pad=10, fontweight="bold")
    ax2.grid(axis="y", color=LIGHT, lw=0.8, zorder=0)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)
    ax2.annotate("A 的 2.50% 均值\n靠这一档撑起来", xy=(2.18, 7), xytext=(2.3, 14.5),
                 fontsize=8.8, color=RED,
                 arrowprops=dict(arrowstyle="->", color=RED, lw=1.0))

    fig.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "第256轮_补位漏斗.png"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_main()
    print("done")
