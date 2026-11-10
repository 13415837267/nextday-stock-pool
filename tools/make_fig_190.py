# -*- coding: utf-8 -*-
"""第一百九十轮配图: 为什么"分散 + 杠杆"救不了, 以及回撤节流能换回多少安全性。"""
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
RED, GREEN, BLUE, GREY, ORANGE = "#c0392b", "#1e8449", "#2e6da4", "#7f8c8d", "#d68910"

d1 = json.load(open(os.path.join(ROOT, "output", "lev_frontier_190.json")))["rows"]
d2 = json.load(open(os.path.join(ROOT, "output", "dd_throttle_190.json")))["rows"]

fig, axes = plt.subplots(1, 2, figsize=(14.2, 6.0))

# ============ (a) 分散 x 杠杆: 四条收益曲线, 单吊始终在上 ============
ax = axes[0]
slots = [(1, "单吊 1 只(全仓)"), (2, "分 2 只(各 50%)"),
         (3, "分 3 只(各 33%)"), (4, "分 4 只(各 25%)")]
colors = [RED, BLUE, GREEN, GREY]
for (tn, lab), c in zip(slots, colors):
    rs = sorted([r for r in d1 if r["top_n"] == tn], key=lambda r: r["lev"])
    x = [r["lev"] for r in rs]
    y = [1 + r["total"] / 100 for r in rs]
    ax.plot(x, y, "o-" if tn == 1 else "o--", color=c, lw=2.4 if tn == 1 else 1.6,
            ms=6 if tn == 1 else 4.5, label=lab, zorder=5 if tn == 1 else 3)
ax.set_yscale("log")
ax.set_xlabel("融资杠杆(倍)", fontsize=10.5)
ax.set_ylabel("1500 日累计倍数(对数轴)", fontsize=10.5)
ax.set_title("① 分散从来不是杠杆的替代品\n同一个杠杆下, 单吊的收益高出 1~2 个数量级",
             fontsize=12.5, pad=10)
ax.grid(alpha=0.25, which="both", ls=":")
ax.legend(fontsize=9, loc="lower right", framealpha=0.95)
ax.set_xticks([1.5, 2.0, 2.5, 3.0])
ax.text(0.03, 0.95, "凯利公式说分散降 σ 可抬 f*\n实测: μ 掉得比 σ² 快,\n总收益反而塌一个量级",
        transform=ax.transAxes, fontsize=8.8, color="#444", va="top",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#ddd", alpha=0.95))

# ============ (b) 全史杠杆前沿 vs 最坏担保比例 ============
ax2 = axes[1]
d3 = json.load(open(os.path.join(ROOT, "output", "dd_throttle_full_190.json")))["rows"]

plain = [r for r in d3 if not r["tiers"]]
cutr = [r for r in d3 if r["tiers"]]

ax2.plot([r["ratio"] for r in plain], [1 + r["total"] / 100 for r in plain],
         "o-", color=RED, lw=2.0, ms=8, zorder=5, label="单吊·无节流")
for r in cutr:
    ax2.scatter(r["ratio"], 1 + r["total"] / 100, s=110, marker="D", color=ORANGE,
                edgecolor="white", lw=0.9, zorder=6, label="单吊·回撤节流")

ax2.axvspan(90, 130, color=RED, alpha=0.10, zorder=0)
ax2.axvline(130, color=RED, lw=1.8, ls="--", zorder=4)
ax2.axvline(150, color=ORANGE, lw=1.4, ls="--", zorder=4)
ax2.set_yscale("log")
ax2.set_xlim(105, 240)
ax2.set_ylim(2.2e4, 1.3e7)

NOTE = {1.5: ("1.50x\n177,217x / 205%", (12, -14), "left"),
        1.6: ("1.60x\n349,429x / 178%", (16, -2), "left"),
        1.7: ("1.70x\n679,541x / 157%", (-18, -4), "right"),
        1.75: ("1.75x\n942,923x / 149%", (2, 22), "center")}
for r in plain:
    key = round(r["lev"], 2)
    if key not in NOTE:
        continue
    txt, off, ha = NOTE[key]
    col = GREEN if key == 1.7 else "#333"
    ax2.annotate(txt, (r["ratio"], 1 + r["total"] / 100), textcoords="offset points",
                 xytext=off, ha=ha, fontsize=8.4, color=col,
                 fontweight="bold" if key == 1.7 else "normal",
                 bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="none", alpha=0.9), zorder=8)

for r in cutr:
    ax2.annotate("2.00x + 回撤节流\n1,149,410x / 128%\n历史最坏仍破 130%", (r["ratio"], 1 + r["total"] / 100),
                 textcoords="offset points", xytext=(-8, 42), fontsize=8.2, color=ORANGE,
                 bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=ORANGE, alpha=0.92), zorder=9)

ax2.set_xlabel("历史最坏时刻的维持担保比例(%, 两融强平线 130%)", fontsize=10.5)
ax2.set_ylabel("全史累计倍数(对数轴)", fontsize=10.5)
ax2.set_title("② 全史口径下, 杠杆只能加到 1.70x\n2.00x 哪怕加了回撤节流, 历史最坏时仍会被强平",
              fontsize=12.5, pad=10)
ax2.grid(alpha=0.25, which="both", ls=":")
ax2.text(127, 2.4e4, "强平区域", color=RED, fontsize=9, ha="right", va="bottom")
ax2.text(152, 2.4e4, "150%\n安全线", color=ORANGE, fontsize=8.2, va="bottom")
ax2.legend(fontsize=8.8, loc="upper left", framealpha=0.95)

plt.tight_layout()
out = os.path.join(ROOT, "第190轮_分散不是杠杆的替代品.png")
plt.savefig(out, bbox_inches="tight")
print("saved", out)
