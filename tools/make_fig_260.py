# -*- coding: utf-8 -*-
"""第260轮配图: 开盘一笔能装多少钱, 装不下要付多少"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

sc = pd.read_csv(ROOT / "output" / "w260_scale.csv")
x = sc["规模(万)"].values
ann = sc["年化%"].values
part = sc["部分成交笔占比%"].values

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 5.3))

# ---------- 左: 规模 -> 年化 ----------
ax1.plot(x, ann, color=BLUE, lw=2.4, marker="o", ms=6, zorder=3)
ax1.axhline(112.5, color=GREY, ls="--", lw=1.3, zorder=1)
ax1.text(11500, 114.0, "不计容量 112.5%", fontsize=9, color="#6b7785", va="bottom", ha="right")
for xi, yi, lab, off in ((30, 106.2, "30万\n106%", (10, -26)), (100, 96.0, "100万\n96%", (8, 8)),
                         (300, 80.6, "300万\n81%", (8, 8)), (1000, 55.5, "1000万\n56%", (8, 8))):
    ax1.scatter([xi], [yi], s=90, color=RED, zorder=4)
    ax1.annotate(lab, (xi, yi), textcoords="offset points", xytext=off,
                 fontsize=9, color=RED, fontweight="bold")
ax1.set_xscale("log")
ax1.set_xticks([30, 50, 100, 300, 1000, 3000, 10000])
ax1.set_xticklabels(["30万", "50万", "100万", "300万", "1000万", "3000万", "1亿"], fontsize=9)
ax1.set_xlabel("单笔投入规模（对数轴）", fontsize=10)
ax1.set_ylabel("可实现年化（%，3200 日）", fontsize=10)
ax1.set_title("年化随规模掉的是“换来的成交价”，不是手续费", fontsize=11.5, color=DARK, pad=10)
ax1.grid(color="#e8ecef", zorder=0)
ax1.set_ylim(0, 125)
for s in ("top", "right"):
    ax1.spines[s].set_visible(False)

# ---------- 右: 装不下多少笔 ----------
b = ax2.bar(range(len(x)), part, 0.62, color=ORANGE, zorder=3)
for i, v in enumerate(part):
    if v >= 3:
        ax2.text(i, v + 1.6, f"{v:.0f}%", ha="center", va="bottom", fontsize=8.6, color="#8a5a00")
ax2.axhline(50, color=GREY, ls=":", lw=1.2, zorder=2)
ax2.text(0.7, 52, "一半的笔装不下", ha="left", fontsize=9, color="#6b7785")
ax2.set_xticks(range(len(x)))
ax2.set_xticklabels([f"{int(v)}万" if v < 10000 else f"{v/10000:.0f}亿" for v in x],
                    fontsize=8.2, rotation=40, ha="right")
ax2.set_xlim(-0.75, len(x) - 0.25)
ax2.set_ylabel("被迫减仓的笔占比（%）", fontsize=10)
ax2.set_ylim(0, 108)
ax2.set_title("规模越大，越多笔只能装一部分（剩余资金空转）", fontsize=11.5, color=DARK, pad=10)
ax2.grid(axis="y", color="#e8ecef", zorder=0)
for s in ("top", "right"):
    ax2.spines[s].set_visible(False)

fig.text(0.5, -0.055,
         "口径：730 笔买入日分钟线（2013-09~2026-11），参与率上限 5%，装不下按容量比例部分成交、剩余资金空转。\n"
         "首分钟（含集合竞价）成交额只占全天 2.78%，开盘一笔的真实容量中位 33 万；第129轮的 1125 万是“一整天慢慢买”的口径。",
         ha="center", fontsize=8.6, color="#5d6d7e")
plt.tight_layout()
plt.savefig(ROOT / "第260轮_开盘能装多少钱.png", bbox_inches="tight")
print("ok")
