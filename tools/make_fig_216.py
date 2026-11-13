#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第 216 轮配图: gap_above 高开降仓倍率的收益/回撤权衡"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

for fp in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
           "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"]:
    try:
        font_manager.fontManager.addfont(fp)
    except Exception:
        pass
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

RED = "#d62728"
BLUE = "#1f77b4"
GREEN = "#2ca02c"
GRAY = "#7f7f7f"


def load(days):
    d = json.loads(Path(f"output/gapabove_216_{days}.json").read_text())
    return d


def main():
    full = load(3664)
    recent = load(1500)
    # 顺序: 基线(0.5), 0.6, 0.7, 0.8, 0.9, 关闭(1.0)
    labels = ["0.5\n(基线)", "0.6", "0.7", "0.8", "0.9", "关闭\n(1.0)"]
    fc = [r["年化%"] for r in full]
    fd = [r["回撤%"] for r in full]
    fk = [r["卡玛"] for r in full]
    rc = [r["年化%"] for r in recent]
    rd = [r["回撤%"] for r in recent]
    rk = [r["卡玛"] for r in recent]
    x = list(range(len(labels)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.4), dpi=130)

    # 左: 年化(柱) + 卡玛(线)
    w = 0.36
    b1 = ax1.bar([i - w / 2 for i in x], fc, w, label="全史·年化", color=BLUE, alpha=0.85)
    b2 = ax1.bar([i + w / 2 for i in x], rc, w, label="近6年·年化", color=RED, alpha=0.85)
    for bars in (b1, b2):
        for b in bars:
            v = b.get_height()
            ax1.text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v:.1f}",
                     ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#1a1a1a")
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("年化收益率 %", fontsize=11)
    ax1.set_title("高开降仓倍率 ↑ → 收益 ↑（双窗口）", fontsize=12.5, fontweight="bold", color=BLUE)
    ax1.set_ylim(0, 155)
    ax1b = ax1.twinx()
    ax1b.plot(x, fk, "o-", color=GREEN, lw=1.5, label="全史·卡玛")
    ax1b.plot(x, rk, "s-", color=GRAY, lw=1.5, label="近6年·卡玛")
    ax1b.set_ylabel("卡玛比率", fontsize=10, color=GREEN)
    ax1b.set_ylim(2.8, 6.6)
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax1b.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left", framealpha=0.9)

    # 右: 回撤(线) 双窗口
    ax2.plot(x, fd, "o-", color=BLUE, lw=1.8, label="全史·回撤")
    ax2.plot(x, rd, "s-", color=RED, lw=1.8, label="近6年·回撤")
    for i, v in enumerate(fd):
        ax2.text(i, v - 0.6, f"{v:.1f}", fontsize=8, color=BLUE, ha="center", va="top")
    for i, v in enumerate(rd):
        ax2.text(i, v + 0.6, f"{v:.1f}", fontsize=8, color=RED, ha="center", va="bottom")
    # 标注 0.6 帕累托点
    ax2.axvline(1, color=GREEN, ls="--", lw=1.2, alpha=0.8)
    ax2.text(1.06, -24.6, "mult=0.6\n近6年回撤不变\n卡玛↑", fontsize=8.5, color=GREEN, fontweight="bold")
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("最大回撤 %", fontsize=11)
    ax2.set_title("回撤随倍率单调恶化 → 0.6 是最佳平衡点", fontsize=12.5, fontweight="bold", color=RED)
    ax2.legend(fontsize=9, loc="lower left", framealpha=0.9)
    ax2.set_ylim(-26, -16)

    fig.suptitle("第 216 轮：高开降仓（gap_above）倍率细扫 —— 0.5→0.6 是双窗口正收益的帕累托改进",
                 fontsize=13.5, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = "/workspace/第216轮_高开降仓倍率优化.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print("已生成", out)


if __name__ == "__main__":
    main()
