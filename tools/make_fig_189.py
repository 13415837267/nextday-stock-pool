# -*- coding: utf-8 -*-
"""第一百八十九轮配图: 杠杆前沿 —— 唯一还能动的那一维, 以及它的实盘边界。"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = ["Noto Serif CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RED, GREEN, BLUE, GREY = "#c0392b", "#1e8449", "#2e6da4", "#7f8c8d"

d = json.load(open(os.path.join(ROOT, "output", "lev_frontier_189.json")))
rows = d["rows"]
L = [r["lev"] for r in rows]
tot = [1 + r["total"] / 100 for r in rows]
mdd = [abs(r["mdd"]) for r in rows]
cal = [r["calmar"] for r in rows]
grt = []
for r in rows:
    lv = r["lev"]
    eq = lv * 100 * (1 + r["mdd"] / 100)
    debt = (lv - 1) * 100
    grt.append(eq / debt * 100 if debt > 0 else np.nan)

fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.6))

# --- (a) 收益柱 + 回撤文字
ax = axes[0]
x = np.arange(len(L))
ax.bar(x, tot, 0.54, color=[RED if lv <= 1.5 else "#e6b0aa" for lv in L],
       edgecolor="white", linewidth=0.9)
ax.set_yscale("log")
ax.set_ylabel("1500 日总倍数（对数轴）", fontsize=10.5, color=RED)
ax.tick_params(axis="y", labelcolor=RED)
for i, (v, m) in enumerate(zip(tot, mdd)):
    ax.text(i, v * 1.55, f"{v:,.0f}x" if v < 10000 else f"{v / 10000:.1f}万x",
            ha="center", fontsize=10, color=RED, fontweight="bold")
    ax.text(i, v * 1.06, f"回撤 −{m:.1f}%", ha="center", fontsize=8.6, color="#555")
ax.set_ylim(60, max(tot) * 14)
ax.set_xticks(x)
ax.set_xticklabels([f"{lv:.2f}x" for lv in L], fontsize=9.5)
ax.set_xlabel("总敞口（权益 × 杠杆）", fontsize=10.5)
ax.set_title("① 收益随敞口指数增长，回撤只线性增长\n"
             "（352 笔 / 单笔 1.44% 全程不变，只放大仓位）", fontsize=12, pad=12)
ax.grid(axis="y", ls=":", color="#ddd", alpha=0.8)
ax.axvspan(2.5, 5.5, color=GREY, alpha=0.07)
ax.text(1.75, max(tot) * 3.2, "1.25x → 1.50x\n收益 ×2.96\n回撤只 +3.9pp",
        fontsize=9.4, color="#333", ha="center",
        bbox=dict(boxstyle="round,pad=0.5", fc="#fdf6e3", ec="#ddd"))

# --- (b) 卡玛柱 + 担保比例线
ax = axes[1]
ax.bar(x, cal, 0.54, color=BLUE, alpha=0.22, edgecolor=BLUE, linewidth=1.2)
for i, v in enumerate(cal):
    ax.text(i, v + 0.30, f"{v:.2f}", ha="center", fontsize=9.8, color=BLUE, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.85))
ax.set_ylim(0, max(cal) * 1.5)
ax.set_ylabel("卡玛比率（年化 / 最大回撤）", fontsize=10.5, color=BLUE)
ax.tick_params(axis="y", labelcolor=BLUE)

ax2 = ax.twinx()
ax2.plot(x, grt, "s--", color=RED, lw=2.0, ms=6.5, zorder=5)
ax2.set_yscale("log")
ax2.axhline(130, color=RED, ls=":", lw=1.5)
ax2.text(5.42, 138, "130% 强平线", ha="right", va="bottom", fontsize=9, color=RED,
         bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9))
for i, v in enumerate(grt):
    if np.isnan(v):
        continue
    ax2.annotate(f"{v:.0f}%", (i, v), textcoords="offset points", xytext=(0, 11),
                 ha="center", fontsize=9, color=RED, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.9))
ax2.set_ylabel("历史最坏时刻担保比例 %（对数轴）", fontsize=10.5, color=RED)
ax2.set_ylim(70, 3000)
ax2.tick_params(axis="y", labelcolor=RED)

ax.set_xticks(x)
ax.set_xticklabels([f"{lv:.2f}x" for lv in L], fontsize=9.5)
ax.set_xlabel("总敞口（权益 × 杠杆）", fontsize=10.5)
ax.set_title("② 卡玛一路上升，但 2.0x 起担保比例逼近强平\n"
             "—— 实盘可行上限 1.50x（历史最坏时仍有 223%）", fontsize=12, pad=12)
ax.grid(axis="y", ls=":", color="#ddd", alpha=0.8)
ax.axvspan(3.5, 5.5, color=RED, alpha=0.06)
ax.text(4.5, max(cal) * 0.62, "不可行", ha="center", fontsize=11.5, color=RED, alpha=0.85)

plt.tight_layout()
p = os.path.join(ROOT, "第189轮_杠杆前沿与实盘边界.png")
plt.savefig(p, bbox_inches="tight", facecolor="white")
print("saved", p)
