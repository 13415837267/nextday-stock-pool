# -*- coding: utf-8 -*-
"""第 214 轮配图: 左=选股门槛放宽(全部恶化); 右=排序/备用池消融(唯一双向正是不可落地的gap)。
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

for fp in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
           "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"]:
    try:
        font_manager.fontManager.addfont(fp)
    except Exception:
        pass
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Serif CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

RED = "#c0392b"
GREEN = "#1e8449"
GRAY = "#555555"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.8))

# ---- 左: 选股门槛放宽(全史年化) ----
labels1 = ["基线", "跌幅\n-7~-2", "跌幅\n-6~-2", "收盘位置\n0.45", "动量\n到35", "距低点\n到4"]
vals1 = [75.73, 36.34, 39.41, 72.49, 56.20, 66.77]
colors1 = [RED] + [GREEN] * (len(vals1) - 1)
bars = ax1.bar(labels1, vals1, color=colors1, alpha=0.85, width=0.62)
ax1.axhline(75.73, color=RED, lw=1, ls="--", alpha=0.6)
ax1.set_ylabel("全史年化收益率 %", fontsize=11)
ax1.set_title("选股门槛：放宽一个即破（全史）", fontsize=13, fontweight="bold", color=RED)
ax1.set_ylim(0, 92)
for b, v in zip(bars, vals1):
    ax1.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}",
             ha="center", fontsize=10, fontweight="bold", color="#1a1a1a")
ax1.text(0.02, 0.96, "基线 75.73%（红） · 其余为放宽门槛，全部恶化",
         transform=ax1.transAxes, fontsize=9, color=GRAY, va="top")
ax1.tick_params(axis="x", labelsize=9)

# ---- 右: 排序 / 备用池消融(全史年化 vs 近6年) ----
labels2 = ["score\n(基线)", "gap\n(不可落地)", "gap_band\n[-3,-1]", "砍\npullback"]
full = [75.73, 76.93, 72.95, 80.72]
recent = [110.71, 112.02, None, 109.58]
x = np.arange(len(labels2))
w = 0.36
b1 = ax2.bar(x - w / 2, full, w, label="全史 3664日", color=RED, alpha=0.85)
b2 = ax2.bar(x + w / 2, [v if v else 0 for v in recent], w, label="近6年 1500日", color=GRAY, alpha=0.75)
ax2.set_xticks(x)
ax2.set_xticklabels(labels2, fontsize=9)
ax2.set_ylabel("年化收益率 %", fontsize=11)
ax2.set_title("排序/备用池：双向正的只有 gap，且不可落地", fontsize=13, fontweight="bold", color=GRAY)
ax2.legend(fontsize=9, loc="upper right")
ax2.set_ylim(0, 130)
for b, v in zip(b1, full):
    ax2.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}",
             ha="center", fontsize=9, fontweight="bold", color="#1a1a1a")
for b, v in zip(b2, recent):
    if v is not None:
        ax2.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}",
                 ha="center", fontsize=9, color="#1a1a1a")
ax2.text(0.02, 0.96, "gap +1.2pp 来自次日开盘缺口挑票，盘后发布单只无法复现 → 伪提升",
         transform=ax2.transAxes, fontsize=9, color=GRAY, va="top")

fig.suptitle("第 214 轮：主策略再提盈 23 变体全扫描 —— 无可落地的进一步优化",
             fontsize=14, fontweight="bold", color="#1a1a1a", y=1.01)
fig.tight_layout()
out = "/workspace/第214轮_主策略再提盈的扫描判决.png"
fig.savefig(out, dpi=110, bbox_inches="tight", facecolor="white")
print("saved", out)
