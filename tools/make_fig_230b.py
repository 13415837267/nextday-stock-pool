"""第230轮 图3/4：评分门槛曲线 + 排序的价值"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
g = pd.read_csv(ROOT / "output" / "w230e_scoregate.csv")
g["x"] = g["门槛"].astype(float).fillna(0.90)
g = g.sort_values("x")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))

labs = ["关门槛\n(0.9)", "1.0\n现行", "1.1", "1.2", "1.3", "1.5", "1.8"]
xi = list(range(len(g)))
a1.plot(xi, g["年化"], "-o", color="#c0392b", lw=2, ms=7)
a1.set_xticks(xi)
a1.set_xticklabels(labs, fontsize=9)
a1.set_xlabel("min_top1_score（评分门槛）")
a1.set_ylabel("年化 %", color="#c0392b")
a1.tick_params(axis="y", colors="#c0392b")
a1b = a1.twinx()
a1b.plot(xi, g["卡玛"], "--s", color="#2980b9", lw=1.3, ms=5, alpha=0.6)
a1b.set_ylabel("卡玛", color="#2980b9")
a1b.tick_params(axis="y", colors="#2980b9")
a1.set_title("评分门槛：1.0 是尖峰，两头都掉", fontsize=12)
a1.grid(alpha=0.25)
for i, (y, n) in enumerate(zip(g["年化"], g["笔数"])):
    a1.annotate(f"{y:.1f}%\n{n}笔", (i, y), textcoords="offset points", xytext=(0, -24),
                ha="center", fontsize=8, color="#c0392b")
a1.set_ylim(min(g["年化"]) - 12, max(g["年化"]) + 16)

d = pd.read_csv(ROOT / "output" / "w230d_shuffle.csv")
names = ["基准\n（按评分）", "不看评分\n（权重全 0）", "评分取反\n（越差越优先）"]
vals = d["年化"].values
cols = ["#2980b9", "#c0392b", "#c0392b"]
a2.bar(range(3), vals, color=cols, width=0.6)
a2.set_xticks(range(3))
a2.set_xticklabels(names, fontsize=10)
a2.set_ylabel("年化 %")
a2.set_title("排序本身值多少钱：不看评分 −47pp，取反 −33pp", fontsize=12)
a2.grid(axis="y", alpha=0.25)
for i, v in enumerate(vals):
    a2.text(i, v + 1.5, f"{v:.1f}%", ha="center", fontsize=11, fontweight="bold")
a2.set_ylim(0, max(vals) * 1.25)

fig.suptitle("评分系统是策略的主要价值来源——但它的最优形态已经在这了", fontsize=13, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.9])
fig.savefig(ROOT / "第230轮_评分门槛与排序价值.png", dpi=130)
print("ok")
