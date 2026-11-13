#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第 219 轮配图: 年化 200% 的价目表"""
import json
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
    lev = json.loads(Path("output/exp_lev_219.json").read_text())["runs"]
    keys = sorted(lev, key=float)
    ls = [float(k) for k in keys]
    cagr = [lev[k]["年化%"] for k in keys]
    mdd = [lev[k]["回撤%"] for k in keys]
    kelly = [lev[k]["卡玛"] for k in keys]
    xs = np.arange(len(ls))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.6, 6.8),
                                   gridspec_kw={"width_ratios": [1.18, 1]})

    # ================= 左: 年化(柱) + 回撤(线) vs 杠杆 =================
    cols = [RED if v >= 200 else ORANGE for v in cagr]
    bars = ax1.bar(xs, cagr, color=cols, width=0.6, alpha=0.9, edgecolor="white", lw=1.4, zorder=2)
    for b, v, k, d in zip(bars, cagr, kelly, mdd):
        cx = b.get_x() + b.get_width() / 2
        ax1.text(cx, v + 20, f"{v:.0f}%", ha="center", va="bottom", zorder=6,
                 fontsize=13, fontweight="bold", color="#b71c1c" if v >= 200 else "#e65100",
                 bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.92))
        ax1.text(cx, v + 3, f"回撤 {d:.1f}%", ha="center", va="bottom", zorder=6,
                 fontsize=9.4, color="#455a64",
                 bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="none", alpha=0.92))
        ax1.text(cx, 8, f"卡玛 {k}", ha="center", va="bottom",
                 fontsize=9.6, color="white", fontweight="bold", zorder=4)

    ax1.axhline(200, color="#263238", ls="--", lw=1.8, zorder=3)
    ax1.text(-0.45, 203, "年化 200% 目标", ha="left", va="bottom", fontsize=11,
             color="#263238", fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.9))


    ax1.set_xticks(xs)
    ax1.set_xticklabels([f"{l:g}×" for l in ls], fontsize=11.5)
    ax1.set_xlabel("融资杠杆倍数（总敞口 / 权益）", fontsize=12.5)
    ax1.set_ylabel("年化收益率（%）", fontsize=12.5, color="#b71c1c")
    ax1.tick_params(axis="y", colors="#b71c1c")
    ax1.set_ylim(0, 360)
    ax1.set_xlim(-0.6, len(ls) - 0.4)
    ax1.grid(axis="y", alpha=0.22, ls=":")
    ax1.set_title("(a) 1500 日同口径：杠杆到 1.53× 才越过 200%", fontsize=13.5, pad=12)

    # ================= 右: 空仓期市场 =================
    yrs = ["2020", "2021", "2022", "2023", "2024", "2025", "2026"]
    vals = [-7.7, 7.6, -14.4, -7.4, -22.4, 3.4, -16.3]
    days = [55, 119, 130, 154, 153, 104, 125]
    xs2 = np.arange(len(yrs))
    cols2 = [RED if v > 0 else GREEN for v in vals]
    bars2 = ax2.bar(xs2, vals, color=cols2, width=0.62, alpha=0.9, edgecolor="white", lw=1.2)
    for b, v, n in zip(bars2, vals, days):
        if v > 0:
            ax2.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v:+.1f}%", ha="center",
                     va="bottom", fontsize=11, fontweight="bold", color=RED)
            ax2.text(b.get_x() + b.get_width() / 2, 0.5, f"{n}天", ha="center",
                     va="bottom", fontsize=9.4, color="white", fontweight="bold")
        else:
            ax2.text(b.get_x() + b.get_width() / 2, v - 0.7, f"{v:+.1f}%", ha="center",
                     va="top", fontsize=11, fontweight="bold", color=GREEN)
            ax2.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{n}天", ha="center",
                     va="bottom", fontsize=9.4, color="white", fontweight="bold")
    ax2.axhline(0, color="#37474f", lw=1.2)
    ax2.set_xticks(xs2)
    ax2.set_xticklabels(yrs, fontsize=11)
    ax2.set_ylabel("空仓期间 市场等权基准 累计收益（%）", fontsize=12)
    ax2.set_ylim(-28, 11)
    ax2.grid(axis="y", alpha=0.22, ls=":")
    ax2.set_title("(b) 76% 的时间空仓 —— 那段时间市场在跌", fontsize=13.5, pad=12)
    ax2.text(0.5, 0.965, "空仓 840 天 累计 −47.1%　|　持仓 658 天 累计 +173.9%",
             transform=ax2.transAxes, ha="center", va="top", fontsize=11.2,
             color="#37474f", fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.45", fc="#eceff1", ec="#b0bec5", lw=1.1))

    fig.suptitle("第 219 轮 · 年化 200% 的价目表：不加杠杆无路，1.53× 即达",
                 fontsize=16.5, fontweight="bold", y=0.985)
    fig.text(0.5, 0.922,
             "杠杆各档在 1500 日、同一 config（仅改 execution.leverage）下重跑；单笔 1.444% 恒定不变——只放大仓位，不改选股",
             ha="center", fontsize=10.2, color="#546e7a")
    fig.tight_layout(rect=[0, 0.02, 1, 0.9])
    out = Path("第219轮_200%的价目表.png")
    fig.savefig(out, dpi=155, bbox_inches="tight", facecolor="white")
    print("saved", out)


if __name__ == "__main__":
    main()
