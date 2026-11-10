# -*- coding: utf-8 -*-
"""第一百八十八轮配图: 右侧为什么不存在的两块硬证据。"""
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
RED, GREEN, GREY = "#c0392b", "#1e8449", "#7f8c8d"

curve = json.load(open(os.path.join(ROOT, "output", "left_right_curve.json")))
lhb = json.load(open(os.path.join(ROOT, "output", "lhb_hist_188.json")))

# ---------------- 图 A
fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4))

ax = axes[0]
qs = [b["q"] for b in curve["bucket"]]
rs = [b["ret"] for b in curve["bucket"]]
ax.bar(qs, rs, color=[GREEN if r < 0 else RED for r in rs], width=0.68,
       edgecolor="white", linewidth=0.8)
ax.axhline(0, color="black", lw=1.0)
for x, r in zip(qs, rs):
    ax.text(x, r - 0.022, f"{r:.2f}", ha="center", va="top", fontsize=8.4, color="#333")
ax.set_xticks(qs)
ax.set_xticklabels([f"D{i}" for i in qs], fontsize=9)
ax.set_xlabel("按当日涨跌幅把全市场分 10 组（D1 = 跌得最狠 10%，D10 = 涨得最猛 10%）", fontsize=10)
ax.set_ylabel("T+1 开盘买 → T+2 开盘卖，单笔净收益 %", fontsize=10)
ax.set_title("① 全市场 3,545 日 × 1,350 万笔：按“当日涨跌幅”选股，\n"
             "十个分组没有一个能覆盖成本", fontsize=12, pad=10)
ax.set_ylim(-0.50, 0.30)
ax.grid(axis="y", ls=":", color="#bbb", alpha=0.7)
ax.text(0.16, 0.98, "最好的 D4 组 −0.250%，最差的 D10 组 −0.414%\n"
                    "两端都亏 → 这不是“左侧/右侧”的选择问题，\n"
                    "是“只看当日涨跌幅”本身不构成信号",
        transform=ax.transAxes, fontsize=9, color="#444", va="top",
        bbox=dict(boxstyle="round,pad=0.5", fc="#fdf6e3", ec="#ddd"))

ax = axes[1]
yrs = sorted(lhb["year_H1"])
vs = [lhb["year_H1"][y][0] for y in yrs]
ax.bar(yrs, vs, color=GREEN, width=0.68, edgecolor="white", linewidth=0.8)
ax.axhline(0, color="black", lw=1.0)
for x, v in zip(yrs, vs):
    ax.text(x, v - 0.03, f"{v:.2f}", ha="center", va="top", fontsize=8, color="#333")
ax.set_xticks(range(len(yrs)))
ax.set_xticklabels([y[2:] for y in yrs], fontsize=9)
ax.set_ylabel("龙虎榜次日（T+1 开 → T+2 开）单笔净收益 %", fontsize=10)
ax.set_xlabel("上榜年份", fontsize=10)
ax.set_title("② 龙虎榜 11 年 13.0 万笔：没有一年是正的\n"
             "（−0.59% ~ −1.05%，不是 2025 年才失效）", fontsize=12, pad=10)
ax.set_ylim(-1.25, 0.15)
ax.grid(axis="y", ls=":", color="#bbb", alpha=0.7)
ax.text(0.03, 0.05, "基准 −0.735%/笔，t = −38.3\n"
                    "持有越久越差：H3 −1.58% / H5 −2.14% / H10 −2.98%",
        transform=ax.transAxes, fontsize=9, color="#444",
        bbox=dict(boxstyle="round,pad=0.5", fc="#fdf6e3", ec="#ddd"))

plt.tight_layout()
p1 = os.path.join(ROOT, "第188轮_涨跌幅无信号与龙虎榜十一年.png")
plt.savefig(p1, bbox_inches="tight", facecolor="white")
print("saved", p1)
plt.close()

# ---------------- 图 B
years = ["2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024", "2025", "2026"]
open_r = [-0.88, -0.65, -0.86, -0.03, -0.89, -0.46, -0.86, 0.03, 1.32, 0.60, -1.08]
close_r = [-1.08, -0.86, -0.42, 0.22, -0.73, -0.12, -0.30, 1.57, 3.72, 1.36, -0.63]

fig, ax = plt.subplots(figsize=(11.2, 5.2))
x = np.arange(len(years))
w = 0.38
ax.bar(x - w / 2, open_r, w, label="T+2 开盘卖（与主策略同口径）", color="#5d6d7e",
       edgecolor="white", linewidth=0.8)
ax.bar(x + w / 2, close_r, w, label="T+2 收盘卖（多拿一天）", color="#e59866",
       edgecolor="white", linewidth=0.8)
ax.axhline(0, color="black", lw=1.0)
ax.set_xticks(x)
ax.set_xticklabels(years, fontsize=9.5)
ax.set_ylabel("单笔净收益 %", fontsize=10.5)
ax.set_xlabel("上榜年份", fontsize=10.5)
ax.set_title("③ 龙虎榜“最好看”的那个组合（跌<−9% & 净买>0 & 市值<30亿），按年拆开\n"
             "—— 2024 一年贡献全部收益，其余 10 年合计归零", fontsize=12.5, pad=12)
ax.legend(fontsize=9.5, loc="upper left", framealpha=0.9)
ax.grid(axis="y", ls=":", color="#bbb", alpha=0.7)
ax.annotate("2024：+3.72% × 348 笔 = 331,248x\n（年初微盘股崩盘 + 924 反弹）",
            xy=(8 + w / 2, 3.72), xytext=(5.1, 3.55), fontsize=9.2, color="#a04000",
            arrowprops=dict(arrowstyle="->", color="#a04000", lw=1.1))
ax.text(0.012, 0.05, "剔除 2024 年：开盘卖 10 年累计 0.0005x，收盘卖 10 年累计 0.25x —— 全部收益只来自这一年",
        transform=ax.transAxes, fontsize=9.5, color="#333",
        bbox=dict(boxstyle="round,pad=0.45", fc="#fdf6e3", ec="#ddd"))
ax.set_ylim(-1.6, 4.3)
plt.tight_layout()
p2 = os.path.join(ROOT, "第188轮_最优组合全靠2024.png")
plt.savefig(p2, bbox_inches="tight", facecolor="white")
print("saved", p2)
