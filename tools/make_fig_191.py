# -*- coding: utf-8 -*-
"""第 191 轮配图: 回撤超越概率 + 历史最坏造成的安全错觉。"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("mc", os.path.join(ROOT, "tools", "exp_mc_191.py"))
mc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mc)

plt.rcParams["font.family"] = ["Noto Serif CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
RED, BLUE, ORANGE, GREY, GREEN = "#c0392b", "#2c6fbb", "#e08a1e", "#8a8a8a", "#2e8b57"

r125 = mc.load_daily()
n = len(r125)
xs = np.linspace(-92, -8, 240)

fig, axes = plt.subplots(1, 2, figsize=(15.4, 6.2))

# ---------------- (a) 回撤超越概率 ----------------
ax = axes[0]
CASES = [(1.25, BLUE), (1.50, ORANGE), (1.70, RED)]
lab_pos = {1.50: (30, 44), 1.70: (22, -40)}
for lev, col in CASES:
    r = mc.to_lev(r125, lev)
    mat = mc.boot(r, B=3000, block=20, horizon=n, seed=707)
    nav = np.cumprod(1 + mat, axis=1)
    dd = (nav / np.maximum.accumulate(nav, axis=1) - 1).min(axis=1) * 100
    _, dd0 = mc.path_stats(r)
    surv = np.array([(dd <= x).mean() * 100 for x in xs])
    ax.semilogy(xs, surv, "-", color=col, lw=2.2, label=(f"{lev:.2f}x 重采样" + ("（强平线外 0%）" if lev == 1.25 else "")))
    thr = mc.crash_dd(lev)
    p = (dd < thr).mean() * 100
    ax.plot([thr], [max(p, 0.05)], "o", color=col, ms=9, mec="white", mew=1.4, zorder=8)
    ax.plot([dd0], [max((dd <= dd0).mean() * 100, 0.05)], "D", color=col, ms=8,
            mec="white", mew=1.2, zorder=8, alpha=0.9)
    if lev in lab_pos:
        ax.annotate(f"{lev:.2f}x  强平 {thr:.0f}%  →  爆仓 {p:.1f}%",
                    (thr, max(p, 0.05)), textcoords="offset points", xytext=lab_pos[lev],
                    fontsize=9.2, color=col, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.32", fc="white", ec=col, lw=1.2, alpha=0.96),
                    zorder=9)

ax.set_xlim(-92, -8)
ax.set_ylim(0.07, 160)

ax.set_xlabel("回撤深度(%, 越往左越深)", fontsize=10.5)
ax.set_ylabel("回撤至少这么深的概率(%, 对数轴)", fontsize=10.5)
ax.set_title("① 实线 = 换 3000 种下跌顺序后的回撤分布\n"
             "●=强平线处的爆仓概率  ◆=历史那一条路径的最坏回撤", fontsize=12.5, pad=10)
ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
ax.grid(alpha=0.22, which="both", ls=":")

# ---------------- (b) 安全错觉 ----------------
ax2 = axes[1]
fine = json.load(open(os.path.join(ROOT, "output", "fine_lev_191.json"), encoding="utf-8"))
le = np.array([f["lev"] for f in fine])
pc = np.array([f["pc"] for f in fine])
margin = np.array([(mc.path_stats(mc.to_lev(r125, lv))[1]) - mc.crash_dd(lv) for lv in le])

x = np.arange(len(le))
bars = ax2.bar(x, pc, 0.62, color=RED, alpha=0.30, edgecolor=RED, linewidth=1.3)
for i, v in enumerate(pc):
    ax2.text(x[i], v + 0.7, f"{v:.2f}%", ha="center", fontsize=8.4, color=RED, fontweight="bold")

ax2.set_xticks(x)
ax2.set_xticklabels([f"{v:.2f}x" for v in le], fontsize=9)
ax2.set_ylim(0, 9.5)
ax2.set_ylabel("整段历史内至少被强平一次的概率(%)", fontsize=10.5, color=RED)
ax2.set_xlabel("杠杆倍数", fontsize=10.5)

ax3 = ax2.twinx()
ax3.plot(x, margin, "o-", color=GREEN, lw=2.2, ms=7, zorder=6)
ax3.set_ylabel("历史最坏回撤距强平线的余量(pp, 绿)", fontsize=10.5, color=GREEN)
ax3.set_ylim(0, 62)
ax3.axhspan(0, 3, color=GREEN, alpha=0.10)
for i, v in enumerate(margin):
    ax3.text(x[i], v + 1.6, f"{v:.0f}", ha="center", fontsize=8.2, color=GREEN)

i140 = list(le).index(1.40)
ax2.axvspan(i140 - 0.5, i140 + 0.5, color=GREEN, alpha=0.09, zorder=0)
ax2.text(i140, 8.9, "1.40x\n爆仓<1%", ha="center", va="top", fontsize=9, color=GREEN,
         fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=GREEN, alpha=0.95), zorder=10)

ax2.set_title("② 绿线(余量)看着一路安全, 红线(真实爆仓概率)指数上升\n"
              "1.50x 余量还有 25pp, 爆仓概率已是 5.6%", fontsize=12.5, pad=10)
ax2.grid(alpha=0.18, ls=":")

plt.tight_layout()
out = os.path.join(ROOT, "第191轮_回撤分布与杠杆上限.png")
plt.savefig(out, bbox_inches="tight", dpi=140)
print("saved", out)
