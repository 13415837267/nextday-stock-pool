# -*- coding: utf-8 -*-
"""第 192 轮配图: 副策略的结构天花板 —— 持有期网格 + 杠杆前沿。"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
plt.rcParams["font.family"] = ["Noto Serif CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
RED, BLUE, ORANGE, GREY, GREEN = "#c0392b", "#2c6fbb", "#e08a1e", "#8a8a8a", "#2e8b57"

hold = json.load(open(os.path.join(ROOT, "output", "exp_sat_hold_192.json")))["rows"]
lev = json.load(open(os.path.join(ROOT, "output", "exp_sat_lev_192.json")))["rows"]

fig, axes = plt.subplots(1, 2, figsize=(15.8, 6.6))

# ---------------- (a) 持有期: H10 是峰 ----------------
ax = axes[0]
hs = [r["hold"] for r in hold]
mult = [1 + r["total"] / 100 for r in hold]
avg = [r["avg"] for r in hold]
ax.bar(hs, mult, 0.62, color=[BLUE if h == 10 else GREY for h in hs],
       edgecolor="white", linewidth=0.8, alpha=0.9)
ax.set_yscale("log")
ax.set_ylim(0.4, 3000)
ax.set_xticks(hs)
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}×"))
ax.set_xlabel("持有交易日（T+1 开盘买入，第 N 个交易日收盘卖出）", fontsize=10.5)
ax.set_ylabel("1500 日累计倍数（对数轴）", fontsize=10.5, color="#333")
for h, m in zip(hs, mult):
    ax.text(h, m * 1.18, f"{m:.1f}×", ha="center", fontsize=9.5,
            color=BLUE if h == 10 else "#444", fontweight="bold" if h == 10 else "normal")

ax2 = ax.twinx()
ax2.plot(hs, avg, "o-", color=RED, lw=2.0, ms=6.5, zorder=6)
ax2.set_ylabel("平均单笔收益 %", fontsize=10.5, color=RED)
ax2.tick_params(axis="y", colors=RED)
ax2.set_ylim(-0.7, 3.6)
OFF = {3: (0, -18), 10: (-42, 4), 20: (2, 14)}
for h, a in zip(hs, avg):
    if h in OFF:
        ax2.annotate(f"{a:+.2f}%", (h, a), textcoords="offset points", xytext=OFF[h],
                     ha="center", fontsize=9.2, color=RED, fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.92))

ax.annotate("现行口径", (10, mult[4]), textcoords="offset points", xytext=(34, 64),
            ha="center", fontsize=9.6, color=BLUE, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.1),
            bbox=dict(boxstyle="round,pad=0.26", fc="white", ec=BLUE, alpha=0.96))
ax.text(4.2, 9.5, "持有期缩到 3 日：单笔只剩 +0.08%\n总收益 -27.8%（越短越差）", fontsize=9.2, color=RED,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=RED, alpha=0.94))

ax.set_title("① 持有期不是瓶颈：越短越差\nH3 单笔 +0.08%、总收益 -27.8%；H10 的 +2.55% 是整条网格的最高点",
             fontsize=12.4, pad=12)
ax.grid(alpha=0.22, axis="y", ls=":")

# ---------------- (b) 杠杆: 百倍够不到, 2.0x 已破强平 ----------------
ax = axes[1]
ls = [r["lev"] for r in lev]
lm = [r["mult"] for r in lev]
cols = [RED if (r["ratio"] and r["ratio"] < 130) else BLUE for r in lev]
ax.bar(ls, lm, 0.34, color=cols, edgecolor="white", linewidth=0.9, alpha=0.9)
ax.set_yscale("log")
ax.set_ylim(3, 900)
ax.set_xlim(0.6, 2.9)
ax.set_xticks(ls)
ax.set_xticklabels([f"{l:.2f}x" for l in ls])
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}×"))
ax.set_xlabel("杠杆（satellite.leverage）· 融资年化 5.5%，逐日计提", fontsize=10.5)
ax.set_ylabel("1500 日累计倍数（对数轴）", fontsize=10.5, color="#333")

for l, m, r in zip(ls, lm, lev):
    hot = bool(r["ratio"] and r["ratio"] < 130)
    ax.text(l, m * 1.14, f"{m:.1f}×", ha="center", fontsize=10.0,
            color=RED if hot else "#333", fontweight="bold")
    if r["ratio"]:
        ax.text(l, m * 0.60, f"担保\n{r['ratio']:.0f}%", ha="center", va="top", fontsize=8.6,
                color="white", linespacing=1.4)

ax.axhline(100, color=GREEN, lw=1.7, ls="--", zorder=2)
ax.text(0.66, 112, "百倍线", color=GREEN, fontsize=9.8, va="bottom", fontweight="bold")

ax.axvspan(1.83, 2.9, color=RED, alpha=0.07, zorder=0)
ax.text(2.37, 760, "历史最坏时已被强平\n（担保 <130%）", color=RED, fontsize=9.4,
        ha="center", va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.93))

ax.set_title("② 杠杆救不了：1.75x 是实盘边界（担保 151%），只到 24.9×\n"
             "2.0x 起历史最坏即破 130% 强平线，2.5x 只剩 88%",
             fontsize=12.4, pad=12)
ax.grid(alpha=0.22, axis="y", ls=":")

plt.tight_layout()
out = os.path.join(ROOT, "第192轮_副策略的天花板.png")
plt.savefig(out, bbox_inches="tight")
print("saved", out)
