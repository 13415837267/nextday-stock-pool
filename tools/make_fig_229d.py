"""第229轮 图4：分箱看起来有信号，改成条件后反转（替补效应）"""
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
df = pd.read_csv(ROOT / "output" / "w229_probe_2500.csv")
main = df[df["pool"] == "main"]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.2))

# 左：主池按流通市值五分位
q = pd.qcut(main["circ_mv"], 5, duplicates="drop")
g = main.groupby(q, observed=True)["ret"].agg(["count", "mean"])
labels = [f"{iv.left:.0f}–{iv.right:.0f}亿\n(n={int(r['count'])})" for iv, (_, r) in zip(g.index, g.iterrows())]
colors = ["#c0392b" if i < 2 else "#95a5a6" for i in range(len(g))]
a1.bar(range(len(g)), g["mean"], color=colors)
a1.axhline(main["ret"].mean(), color="#2980b9", ls="--", lw=1.4,
           label=f"主池整体 {main['ret'].mean():.3f}%")
a1.set_xticks(range(len(g)))
a1.set_xticklabels(labels, fontsize=9)
a1.set_ylabel("平均单笔收益 %")
a1.set_xlabel("流通市值（亿元，五分位）")
a1.set_title("分箱：五档从 0.26 爬到 2.29，看起来“挑档就能提升”\n（但形状是 U 型，没有经济含义）", fontsize=11)
a1.legend(fontsize=9)
a1.grid(axis="y", alpha=0.25)
for i, v in enumerate(g["mean"]):
    a1.text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=9)

# 右：预期 vs 实测
keep = g["mean"].iloc[1:]
w = g["count"].iloc[1:]
expect = float((keep * w).sum() / w.sum())
actual = 1.124      # circ_mv_min=27 真实回测（2500日）
now = main["ret"].mean()
bars = [now, expect, actual]
names = ["现状\n(不加条件)", "分箱推算\n(剔除 9–27 亿后)", "真实回测\n剔除 9–27 亿"]
cs = ["#2980b9", "#95a5a6", "#c0392b"]
a2.bar(range(3), bars, color=cs, width=0.6)
a2.set_xticks(range(3))
a2.set_xticklabels(names, fontsize=10)
a2.set_ylabel("平均单笔收益 %")
a2.set_title("实测：按分箱推算该升到 1.330，实际只有 1.124", fontsize=12)
a2.grid(axis="y", alpha=0.25)
for i, v in enumerate(bars):
    a2.text(i, v + 0.03, f"{v:.3f}%", ha="center", fontsize=11, fontweight="bold")
a2.annotate("", xy=(2, actual + 0.02), xytext=(1, expect - 0.02),
            arrowprops=dict(arrowstyle="->", color="#c0392b", lw=2))
a2.text(1.5, (expect + actual) / 2 + 0.12, "−0.21pp", color="#c0392b", ha="center", fontsize=11)

fig.suptitle("为什么“筛掉差票”总是伤收益：第 1 名被换掉了，替补比被淘汰的更差", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(ROOT / "第229轮_分箱与实测反转.png", dpi=130)
print("ok", round(now, 3), round(expect, 3), actual)
