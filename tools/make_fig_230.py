"""第230轮 图表：排名置信度诊断"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
d = pd.read_csv(ROOT / "output" / "w230_trades.csv").dropna(subset=["margin", "score", "pool_n"])

fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))

for ax, col, label, color in [
    (axs[0], "score", "评分 score（越高越好）", "#2980b9"),
    (axs[1], "margin", "领先幅度 margin（top1−top2）", "#c0392b"),
    (axs[2], "pool_n", "当日候选池深度 pool_n", "#16a085"),
]:
    q = pd.qcut(d[col], 5, duplicates="drop")
    g = d.groupby(q, observed=True)["ret"].agg(["count", "mean"])
    r = d[col].corr(d["ret"])
    ax.bar(range(len(g)), g["mean"], color=color, alpha=0.85)
    ax.axhline(d["ret"].mean(), color="#333", ls="--", lw=1.2,
               label=f"整体 {d['ret'].mean():.2f}%")
    ax.set_xticks(range(len(g)))
    ax.set_xticklabels([f"{iv.left:.2f}–{iv.right:.2f}" for iv in g.index], fontsize=8, rotation=20)
    ax.set_title(f"{label}\n相关系数 {r:+.3f}", fontsize=11)
    ax.set_ylabel("平均单笔收益 %")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    for i, v in enumerate(g["mean"]):
        ax.text(i, v + 0.04, f"{v:.2f}", ha="center", fontsize=8)

fig.suptitle("第 1 名到底有多可信？——评分与收益无关（+0.013），领先幅度反而负相关（−0.141）", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.9])
fig.savefig(ROOT / "第230轮_排名置信度诊断.png", dpi=130)
print("fig ok")

# 门槛权衡
m = pd.read_csv(ROOT / "output" / "w230_margin.csv")
m["门槛"] = m["门槛"].astype(float).fillna(0.0)
m = m.sort_values("门槛")
fig, a1 = plt.subplots(figsize=(8.6, 4.8))
a1.plot(m["门槛"], m["年化"], "-o", color="#c0392b", lw=2, label="年化（左轴）")
a1.set_xlabel("min_top1_margin：只在领先幅度 ≥ 该值时买入")
a1.set_ylabel("年化 %", color="#c0392b")
a1.tick_params(axis="y", colors="#c0392b")
a2 = a1.twinx()
a2.plot(m["门槛"], m["笔数"], "-s", color="#2980b9", lw=1.6, label="笔数（右轴）")
a2.set_ylabel("交易笔数", color="#2980b9")
a2.tick_params(axis="y", colors="#2980b9")
for x, y, n in zip(m["门槛"], m["年化"], m["笔数"]):
    a1.annotate(f"{y:.0f}%", (x, y), textcoords="offset points", xytext=(0, 8),
                ha="center", fontsize=8, color="#c0392b")
a1.set_title("把 margin 当买入门槛：每砍 1% 的笔数，年化掉约 0.6pp\n（2500 日，559 笔 → 174 笔）", fontsize=12)
a1.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(ROOT / "第230轮_margin门槛的代价.png", dpi=130)
print("fig2 ok")
