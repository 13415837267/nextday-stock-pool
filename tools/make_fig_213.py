# -*- coding: utf-8 -*-
"""第 213 轮配图: 左侧=右侧形态×持有期收益(唯一正的是逼近新高); 右侧=逼近新高单吊 vs 全池。
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

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.6))

# ---- 左: 右侧形态 × 持有期(净收益) ----
forms = {
    "连板":        [-0.42, -1.26, -1.23, -2.33],
    "涨停":        [-0.34, -0.54, -0.71, -0.90],
    "首板":        [-0.31, -0.29, -0.53, -0.40],
    "放量突破":    [-0.21, -0.35, -0.48, -0.22],
    "均线多头":    [-0.17, -0.14, -0.10, +0.33],
    "逼近250新高": [-0.03, +0.27, +0.58, +1.39],
}
holds = ["1日", "3日", "5日", "10日"]
x = np.arange(len(holds))
for i, (nm, vals) in enumerate(forms.items()):
    c = RED if nm == "逼近250新高" else GRAY
    lw = 2.6 if nm == "逼近250新高" else 1.3
    ax1.plot(x, vals, marker="o", ms=4, color=c, lw=lw,
             label=nm, alpha=0.9 if nm == "逼近250新高" else 0.55)
ax1.axhline(0, color="black", lw=0.8)
ax1.set_xticks(x); ax1.set_xticklabels(holds)
ax1.set_ylabel("T+1买·持有h日卖 净收益(%)", fontsize=10)
ax1.set_title("右侧形态 × 持有期：只有逼近250日新高随持有变正", fontsize=12, fontweight="bold")
ax1.legend(fontsize=9, ncol=2, frameon=False)
ax1.grid(axis="y", ls=":", alpha=0.4)

# ---- 右: 逼近新高池 全池 vs 单吊第1名 ----
labels = ["全池均值(ret10)", "距新高最近", "动量最强", "量比最小",
          "波动最低", "连涨最多", "当日涨幅最小"]
vals = [1.39, -0.19, 0.02, -0.27, 0.22, -0.44, -0.29]
colors = [RED] + [GREEN] * (len(vals) - 1)
y = np.arange(len(labels))[::-1]
ax2.barh(y, vals, color=colors, alpha=0.85)
ax2.axvline(0, color="black", lw=0.8)
for yi, v in zip(y, vals):
    ax2.text(v + (0.06 if v >= 0 else -0.06), yi, f"{v:+.2f}%",
             va="center", ha="left" if v >= 0 else "right",
             fontsize=10, color="#1a1a1a", fontweight="bold")
ax2.set_yticks(y); ax2.set_yticklabels(labels, fontsize=8.5)
ax2.set_xlim(-1.6, 1.9)
ax2.set_xlabel("ret10 净收益(%)", fontsize=10)
ax2.set_title("逼近新高池：全池 +1.39%，单吊第1名全在 ±0.3% 内", fontsize=12, fontweight="bold")
ax2.grid(axis="x", ls=":", alpha=0.4)

fig.suptitle("第 213 轮：右侧追高持有多日仍无解 —— 副策略删除", fontsize=14, fontweight="bold", y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = "/workspace/第213轮_右侧持有多日仍无解_副策略删除.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print("写出", out)
