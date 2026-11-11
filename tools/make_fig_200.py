# -*- coding: utf-8 -*-
"""第200轮配图: 分钟级执行面 —— 全部为负，除了一处口径修正"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
OUT = Path("/workspace/第200轮_分钟级执行面全否.png")


def fig():
    fig = plt.figure(figsize=(13.4, 5.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.15, .8], wspace=.30)

    # (a) 买入时点后移的代价
    ax = fig.add_subplot(gs[0, 0])
    lab = ["09:31\n开盘（现行）", "10:00", "10:30", "11:30", "14:00"]
    mult = [109.0, 13.9, 7.0, 3.7, 2.9]
    x = np.arange(len(lab))
    bars = ax.bar(x, mult, .58, color=["#c0392b"] + ["#95a5a6"] * 4)
    ax.set_yscale("log")
    ax.set_ylim(2, 220)
    for xi, v in zip(x, mult):
        ax.text(xi, v * 1.12, f"{v:.1f}×", ha="center", va="bottom", fontsize=9,
                color="#c0392b" if v > 100 else "#444")
    ax.set_xticks(x)
    ax.set_xticklabels(lab, fontsize=8.6)
    ax.set_ylabel("1500 日总倍数（对数）")
    ax.set_title("(a) 买入晚上 30 分钟，109 倍变 13.9 倍", fontsize=11, loc="left")
    ax.grid(axis="y", alpha=.22, ls=":")


    # (b) 日内退出规则
    ax = fig.add_subplot(gs[0, 1])
    ax.axhline(109.0, color="#c0392b", ls="--", lw=1.2)
    ax.text(.03, .93, "基准 109×（收盘卖）", transform=ax.transAxes, fontsize=8.2,
            color="#c0392b", va="top")
    tr_x = [0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0]
    tr_y = [18.6, 17.6, 21.1, 32.8, 32.4, 70.2, 71.3]
    tk_x = [2, 3, 5, 7, 10, 12, 15, 20]
    tk_y = [40.3, 61.4, 95.7, 97.2, 147.7, 125.4, 130.2, 113.4]
    ax.plot(tr_x, tr_y, "o-", color="#2980b9", ms=5, lw=1.6, label="移动止盈：从日内高点回落 x%")
    ax.plot(tk_x, tk_y, "s-", color="#27ae60", ms=5, lw=1.6, label="固定止盈：开盘后涨 y% 卖")
    ax.set_xlabel("阈值（%）")
    ax.set_ylabel("总倍数（对数）")
    ax.set_yscale("log")
    ax.set_ylim(14, 260)
    ax.set_title("(b) 移动止盈全在基准下方；固定止盈是带峰噪声", fontsize=11, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.grid(alpha=.22, ls=":")

    # (c) 14:55 -> 15:00
    ax = fig.add_subplot(gs[0, 2])
    vals = [0.028, 0.044]
    b = ax.bar([0, 1], vals, .46, color="#8e44ad")
    for xi, v in zip([0, 1], vals):
        ax.text(xi, v + .002, f"+{v:.3f}%", ha="center", fontsize=9.5, color="#8e44ad")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["1500 日", "3000 日"])
    ax.set_ylim(0, .075)
    ax.set_ylabel("每笔多赚（%）")
    ax.set_title("(c) 唯一正向：14:55 → 15:00", fontsize=11, loc="left")
    ax.grid(axis="y", alpha=.22, ls=":")
    ax.text(.5, .80, "双窗口同向\n年化 +1.8pp / +1.0pp", transform=ax.transAxes,
            ha="center", va="top", fontsize=8.4, color="#555")

    fig.suptitle("第 200 轮：分钟数据拿得到（2015 年起，240 根/日），但没有一分钱是白捡的",
                 fontsize=14, y=1.04, x=.008, ha="left", fontweight="bold")
    fig.text(.008, -.10,
             "口径：zzshare stk_mins(trade_time=YYYYMMDD) 取 1 分钟 K 线，与日线 OHLC 逐位一致（未复权）；"
             "逐笔来自 197 轮回测（355/717 笔），所有阈值均用 T 日或 T+2 当日已实现路径，无未来函数。\n"
             "移动止盈在 99.7% 的笔上被触发、且 98.9% 发生在开盘首小时 —— 这不是择时，是被开盘的噪声震出去；"
             "固定止盈的峰值在 10%（43 笔触发），相邻 8%/12% 掉到 132/125，是选峰不是平台。",
             fontsize=7.6, color="#555")
    fig.savefig(OUT, dpi=170, bbox_inches="tight", facecolor="white")
    print("wrote", OUT)


if __name__ == "__main__":
    fig()
