# -*- coding: utf-8 -*-
"""第259轮配图: 补位的 edge 是"日子"的属性, 不是"族"的属性"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

for fp in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
           "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
           "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"):
    if Path(fp).exists():
        font_manager.fontManager.addfont(fp)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130

RED, BLUE, GREY, DARK, ORANGE = "#c0392b", "#2c6fbb", "#95a5a6", "#2c3e50", "#d68910"

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.4, 4.6),
                                    gridspec_kw={"width_ratios": [1.05, 1.0, 0.95]})

# ---------- 左: 同一个 momentum 族, 三个入口 ----------
names = ["只接“没票日”\n（现状）", "再接“评分门日”\n（本轮 B）", "再接“缺口门日”\n（全顺延 F）"]
vals = [1.30, 0.61, 0.14]
cnt = [68, 68, 28]
cols = [BLUE, ORANGE, RED]
b = ax1.bar(range(3), vals, 0.58, color=cols, zorder=3)
for i, (v, c) in enumerate(zip(vals, cnt)):
    ax1.text(i, v + 0.07, f"+{v:.2f}%", ha="center", va="bottom", fontsize=13,
             color=cols[i], fontweight="bold")
    ax1.text(i, -0.22, f"{c} 笔", ha="center", va="top", fontsize=10, color="#7f8c8d")
ax1.axhline(1.50, color=GREY, ls="--", lw=1.3, zorder=2)
ax1.text(2.44, 1.54, "主力单笔 1.50%", ha="right", va="bottom", fontsize=9, color="#6b7785")
ax1.set_xticks(range(3))
ax1.set_xticklabels(names, fontsize=9.5)
ax1.set_ylim(-0.5, 2.05)
ax1.set_ylabel("momentum 备用票的单笔收益（%）", fontsize=10)
ax1.set_title("同一个族，换个日子进场，期望差 10 倍", fontsize=11.5, color=DARK, pad=10)
ax1.grid(axis="y", color="#e8ecef", zorder=0)
for s in ("top", "right"):
    ax1.spines[s].set_visible(False)

# ---------- 中: 三窗口年化 ----------
WIN = ["1500 日", "2500 日", "3200 日"]
A = [152.10, 95.89, 110.84]
Bv = [150.75, 92.66, 105.97]
F = [146.68, 90.98, None]
x = np.arange(3)
w = 0.26
b1 = ax2.bar(x - w, A, w, color=BLUE, label="现状（单槽漏斗）", zorder=3)
b2 = ax2.bar(x, Bv, w, color=ORANGE, label="评分门也接（B）", zorder=3)
b3 = ax2.bar(x + w, [v if v else 0 for v in F], w, color=RED, label="全顺延（F）", zorder=3)
for bars, vs in ((b1, A), (b2, Bv), (b3, F)):
    for bb, v in zip(bars, vs):
        if v:
            ax2.text(bb.get_x() + bb.get_width() / 2, v - 4.5, f"{v:.1f}", ha="center",
                     va="top", fontsize=8.4, color="white", fontweight="bold")
ax2.text(2 + w, 4.0, "未跑", ha="center", va="bottom", fontsize=8.4, color="#b03a2e")
ax2.set_xticks(x)
ax2.set_xticklabels(WIN, fontsize=10)
ax2.set_ylabel("年化收益率（%）", fontsize=10)
ax2.set_ylim(0, 175)
ax2.set_title("多接的每一笔都在拉低年化", fontsize=11.5, color=DARK, pad=10)
ax2.legend(fontsize=8.8, frameon=False, loc="upper right")
ax2.grid(axis="y", color="#e8ecef", zorder=0)
for s in ("top", "right"):
    ax2.spines[s].set_visible(False)

# ---------- 右: 31-40 档位非单调 ----------
mults = ["三成\n0.3", "半仓\n0.5", "七成\n0.7", "满仓\n1.0"]
pv = [148.01, 150.56, 152.72, 152.10]
cc = [RED, ORANGE, BLUE, GREY]
ax3.plot(range(4), pv, "-o", color="#b0b8c0", lw=1.4, zorder=2, ms=0)
for i, (v, c) in enumerate(zip(pv, cc)):
    ax3.scatter(i, v, s=90, color=c, zorder=3, edgecolor="white", linewidth=1.2)
    ax3.text(i, v + 0.34, f"{v:.1f}", ha="center", va="bottom", fontsize=10.5, color=c,
             fontweight="bold")
ax3.set_xticks(range(4))
ax3.set_xticklabels(mults, fontsize=9.5)
ax3.set_ylim(146.8, 154.2)
ax3.set_ylabel("1500 日年化（%）", fontsize=10)
ax3.set_title("31-40 只降档：不单调 = 噪声", fontsize=11.5, color=DARK, pad=10)
ax3.grid(axis="y", color="#e8ecef", zorder=0)
for s in ("top", "right"):
    ax3.spines[s].set_visible(False)

fig.tight_layout()
out = Path(__file__).resolve().parents[1] / "第259轮_补位的日子.md".replace(".md", ".png")
plt.savefig(out, bbox_inches="tight", facecolor="white")
print(out)
