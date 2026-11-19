# -*- coding: utf-8 -*-
"""第260轮b配图: 卖出端 —— 收盘卖是最优点, 但收盘那一下装不下太多"""
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
ROOT = Path(__file__).resolve().parents[1]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 4.7))

# ---------- 左: 两端容量 ----------
w = [1, 5, 15, 30, 60]
buy = [32.9, 87.2, 194.1, 306.9, 477.1]
sell = [13.3, 31.9, 88.1, 145.1, 245.6]
xx = np.arange(len(w))
ax1.plot(xx, buy, color=BLUE, lw=2.4, marker="o", ms=6, label="买（开盘后第 N 分钟）", zorder=3)
ax1.plot(xx, sell, color=ORANGE, lw=2.4, marker="s", ms=6, label="卖（收盘前 N 分钟）", zorder=3)
for i, (b, s) in enumerate(zip(buy, sell)):
    ax1.annotate(f"{b:.0f}", (xx[i], b), textcoords="offset points", xytext=(0, 10),
                 ha="center", fontsize=8.6, color=BLUE)
    ax1.annotate(f"{s:.0f}", (xx[i], s), textcoords="offset points",
                 xytext=(14 if i == 0 else 0, -14 if i == 0 else -14),
                 ha="left" if i == 0 else "center", fontsize=8.6, color="#8a5a00")
ax1.set_xticks(xx)
ax1.set_xticklabels([f"{v} 分钟" for v in w], fontsize=9.5)
ax1.set_xlabel("愿意给这只票的成交时间", fontsize=10)
ax1.set_ylabel("可装资金（万元，5% 参与率）", fontsize=10)
ax1.set_title("同样等 1 分钟：开盘能装 33 万，收盘只能装 13 万", fontsize=11.5, color=DARK, pad=10)
ax1.grid(color="#e8ecef", zorder=0)
ax1.set_ylim(-10, 540)
ax1.legend(fontsize=9, frameon=False, loc="upper left")
for s in ("top", "right"):
    ax1.spines[s].set_visible(False)

# ---------- 右: 卖出时点 ----------
lab = ["10:00", "10:30", "11:00", "11:30", "13:30", "14:00", "14:30", "14:55", "15:00"]
val = [1.315, 1.446, 1.495, 1.544, 1.607, 1.614, 1.633, 1.647, 1.689]
cols = [RED if v < 1.60 else BLUE for v in val[:-1]] + [DARK]
bars = ax2.bar(range(len(val)), val, 0.6, color=cols, zorder=3)
for i, v in enumerate(val):
    ax2.text(i, v + 0.03, f"{v:.2f}", ha="center", va="bottom", fontsize=8.4,
             color=(DARK if i == len(val) - 1 else "#5d6d7e"))
ax2.axhline(1.689, color=GREY, ls="--", lw=1.2, zorder=2)
ax2.text(0.0, 1.70, "收盘卖", fontsize=8.6, color="#6b7785", va="bottom")
ax2.set_xticks(range(len(val)))
ax2.set_xticklabels(lab, fontsize=8.6)
ax2.set_ylabel("单笔收益（%，未扣费）", fontsize=10)
ax2.set_ylim(1.0, 1.86)
ax2.set_title("T+2 越晚卖越好 —— 提前一次卖，每 30 分钟丢 0.04~0.06pp", fontsize=11.5, color=DARK, pad=10)
ax2.grid(axis="y", color="#e8ecef", zorder=0)
for s in ("top", "right"):
    ax2.spines[s].set_visible(False)

fig.text(0.5, -0.05,
         "口径：730 笔卖出日分钟线（2013-09~2026-11）。左图为 5% 参与率下的可成交金额；"
         "右图为在该时点一次性卖出的单笔收益。",
         ha="center", fontsize=8.6, color="#5d6d7e")
plt.tight_layout()
plt.savefig(ROOT / "第260轮b_卖得出去吗.png", bbox_inches="tight")
print("ok")
