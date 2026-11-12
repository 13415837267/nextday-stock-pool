# -*- coding: utf-8 -*-
"""第 212 轮配图: 左=右侧追高证伪(全负), 右=左侧深跌甜点(越深越正)。
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

# 中文字体
for fp in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
           "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"]:
    try:
        font_manager.fontManager.addfont(fp)
    except Exception:
        pass
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Serif CJK JP",
                                   "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

RED = "#c0392b"
GREEN = "#1e8449"
GRAY = "#555555"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.6))

# ---- 左: 右侧追高证伪 ----
right_labels = ["全部涨停", "首板", "二板", "三板", "龙虎榜\n净买入>0",
                "净买入\n占比>10%", "接近\n20日新高"]
right_vals = [-0.442, -0.286, -0.732, -2.103, -0.571, -0.788, -0.312]
colors = [GREEN] * len(right_vals)
y = np.arange(len(right_labels))[::-1]
bars = ax1.barh(y, right_vals, color=colors, alpha=0.85)
ax1.axvline(0, color="black", lw=0.8)
for yi, v in zip(y, right_vals):
    ax1.text(v - (0.06 if v < 0 else -0.06), yi, f"{v:+.2f}%",
             va="center", ha="right" if v < 0 else "left",
             fontsize=10, color="#1a1a1a", fontweight="bold")
ax1.set_yticks(y)
ax1.set_yticklabels(right_labels, fontsize=10)
ax1.set_xlim(-2.6, 0.6)
ax1.set_xlabel("T+1开盘买 → T+2收盘卖 净收益(%)", fontsize=10)
ax1.set_title("右侧追高：全部负期望", fontsize=13, fontweight="bold", color=GREEN)

# ---- 右: 左侧深跌甜点 ----
deep_labels = ["-80~-65", "-65~-55", "-55~-45", "-45~-35", "-35~-25", "-25~-15"]
deep_vals = [1.312, 0.671, 0.177, -0.197, -0.269, -0.246]
deep_colors = [RED if v > 0 else GREEN for v in deep_vals]
y2 = np.arange(len(deep_labels))[::-1]
ax2.barh(y2, deep_vals, color=deep_colors, alpha=0.85)
ax2.axvline(0, color="black", lw=0.8)
for yi, v in zip(y2, deep_vals):
    ax2.text(v + (0.05 if v >= 0 else -0.05), yi, f"{v:+.2f}%",
             va="center", ha="left" if v >= 0 else "right",
             fontsize=10, color="#1a1a1a", fontweight="bold")
ax2.set_yticks(y2)
ax2.set_yticklabels(deep_labels, fontsize=10)
ax2.set_xlim(-0.6, 1.8)
ax2.set_xlabel("T+1开盘买 → T+2收盘卖 净收益(%)", fontsize=10)
ax2.set_title("左侧深跌：距250日高点越深越正", fontsize=13, fontweight="bold", color=RED)
fig.text(0.62, 0.03, "深跌池内「加速暴跌」子集(20日跌>30%)：+3.19%、胜率66%",
         fontsize=9, color=RED, ha="center")

fig.suptitle("第212轮 · 从零重建判决：右侧追高无解，左侧只有深跌反弹赚钱",
             fontsize=14, fontweight="bold", y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = "/workspace/第212轮_右侧证伪与左侧深跌甜点.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print("saved", out)
