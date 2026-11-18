# -*- coding: utf-8 -*-
"""第257轮配图: 全历史事件表挖出来的东西, 以及为什么只落地一条"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from pathlib import Path

for fp in ("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
           "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
           "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"):
    if Path(fp).exists():
        font_manager.fontManager.addfont(fp)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()
        break
plt.rcParams["figure.dpi"] = 130
plt.rcParams["axes.unicode_minus"] = False

RED, GREEN, BLUE, DARK, GREY = "#c0392b", "#27ae60", "#2e6da4", "#2c3e50", "#95a5a6"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.6, 6.0))

# ---------- 左: 破位腿按当日跌幅 ----------
labels = ["≤ -5%\n（本轮门槛）", "-5 ~ -3%", "> -3%\n（指数崩但个股没崩）"]
vals = [4.19, 0.85, 0.71]
ns = [63, 19, 72]
cols = [RED, GREY, GREY]
x = range(len(vals))
bars = ax1.bar(x, vals, color=cols, width=0.5, zorder=2)
for i, (b, v, n) in enumerate(zip(bars, vals, ns)):
    ax1.text(b.get_x() + b.get_width() / 2, v + 0.14, f"{v:.2f}%", ha="center",
             va="bottom", fontsize=13, fontweight="bold", color=DARK)
    ax1.text(b.get_x() + b.get_width() / 2, -0.5, f"{n} 笔", ha="center",
             va="top", fontsize=9.6, color="#5d6d7e")
ax1.axvline(0.5, color=BLUE, ls="--", lw=1.3)
ax1.text(0.55, 4.6, "跨过这条线：单笔从 0.8% 跳到 4.2%\n（其中跌停级 ≤-9% 那 44 笔是 4.13%）",
         fontsize=9.6, color=BLUE, va="top")
ax1.set_xticks(list(x))
ax1.set_xticklabels(labels, fontsize=9.8)
ax1.set_ylabel("次日开盘买、T+2 收盘卖的单笔收益 %", fontsize=10)
ax1.set_title("破位腿：指数崩还不够，这只票自己也得崩\n（全历史 2010-2026，154 个破位日，次日不高开才计入）",
              fontsize=11.5, pad=12)
ax1.set_ylim(-1.0, 6.0)
ax1.grid(axis="y", ls=":", alpha=0.45, zorder=0)
for sp in ("top", "right"):
    ax1.spines[sp].set_visible(False)

# ---------- 右: 深低开带的"事件层好看、回测更差" ----------
names = ["-8 ~ -5%\n（现状 256）", "-6 ~ -4%\n（事件层最优）", "-10 ~ -4%\n（放最宽）"]
ev = [2.03, 2.97, 2.09]
dd = [23.31, 25.75, 25.75]
x2 = range(3)
w = 0.36
b1 = ax2.bar([i - w / 2 for i in x2], ev, width=w, color=BLUE,
             label="事件层单笔收益 %（左轴）", zorder=2)
ax2b = ax2.twinx()
b2 = ax2b.bar([i + w / 2 for i in x2], dd, width=w, color=RED, alpha=0.85,
              label="3200 日最大回撤 %（右轴，取绝对值）", zorder=2)
for b, v in zip(b1, ev):
    ax2.text(b.get_x() + b.get_width() / 2, v + 0.06, f"{v:.2f}%", ha="center",
             va="bottom", fontsize=10, color=BLUE, fontweight="bold")
for b, v in zip(b2, dd):
    ax2b.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.1f}%", ha="center",
              va="bottom", fontsize=10, color=RED, fontweight="bold")
ax2.set_xticks(list(x2))
ax2.set_xticklabels(names, fontsize=9.8)
ax2.set_ylabel("事件层单笔收益 %", fontsize=10, color=BLUE)
ax2.set_ylim(0, 3.9)
ax2b.set_ylabel("3200 日最大回撤 %（绝对值，越高越差）", fontsize=10, color=RED)
ax2b.set_ylim(18, 28.5)
ax2.set_title("深低开带：事件层均值越好，回测回撤越坏\n（-5~-4% 在股灾里接着跌，均值掩盖了尾部）",
              fontsize=11.5, pad=12)
h1, l1 = ax2.get_legend_handles_labels()
h2, l2 = ax2b.get_legend_handles_labels()
ax2.legend(h1 + h2, l1 + l2, fontsize=8.8, frameon=False, loc="upper left")
ax2.spines["top"].set_visible(False)
ax2b.spines["top"].set_visible(False)

plt.tight_layout()
out = "/workspace/repo/第257轮_全历史挖条件.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print("saved", out)
