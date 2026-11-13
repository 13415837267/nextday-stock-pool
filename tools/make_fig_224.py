#!/usr/bin/env python3
"""第224轮 图: 风控无空间, 成本有 20 个点。"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130

RED, GRAY, DARK = "#c0392b", "#9aa0a6", "#2c3e50"


def main() -> None:
    scan = pd.read_csv(ROOT / "output" / "regime_scan_224.csv")
    cost = pd.read_csv(ROOT / "output" / "cost_224.csv")

    fig, axes = plt.subplots(1, 2, figsize=(14.6, 5.3), gridspec_kw={"width_ratios": [1.15, 1]})

    # ---------- ① 风控热力 ----------
    ax = axes[0]
    sub = scan[scan["vol"] == 2.2].copy()
    ms = [10, 20, 30, 60]
    bs = [0.20, 0.25, 0.30, 0.35]
    mat = np.full((len(ms), len(bs)), np.nan)
    for _, r in sub.iterrows():
        if r["ma"] in ms and round(float(r["breadth"]), 2) in bs:
            mat[ms.index(r["ma"]), bs.index(round(float(r["breadth"]), 2))] = r["年化"]

    im = ax.imshow(mat, cmap="Blues", aspect="auto", vmin=55, vmax=120)
    ax.set_xticks(range(len(bs)), [f"{b:.2f}" for b in bs])
    ax.set_yticks(range(len(ms)), [f"{m}日" for m in ms])
    ax.set_xlabel("breadth_min（上涨家数占比门槛）", fontsize=9)
    ax.set_ylabel("market_ma（大盘均线）", fontsize=9)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:.0f}%", ha="center", va="center", fontsize=10,
                    fontweight="bold", color="white" if v > 90 else "black")
    # 现行 + 最优: ma=20 行(索引1), breadth 0.20~0.30 列(索引0~2)
    ax.add_patch(plt.Rectangle((-0.5, 0.5), 3, 1, fill=False, edgecolor=RED, lw=2.6))
    ax.set_title("① 大盘风控：ma=20 是硬最优（红框=现行）", fontsize=10.5, pad=8)
    ax.text(0.5, -0.19, "相邻取值只差 0~3pp，说明这层已经榨干；换 10 日或 60 日均线要丢掉 30~44pp",
            transform=ax.transAxes, ha="center", fontsize=7.8, color="#444")
    plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="年化 %")

    # ---------- ② 成本阶梯 ----------
    ax = axes[1]
    lbl = ["现行\n滑千1+佣万2.5", "免滑点\n佣万2.5", "低费\n滑千0.2+佣万1",
           "低费+现金1.8\n（可执行）", "零成本\n理论上限"]
    vals = cost["年化%"].tolist()
    cols = [GRAY, "#e8908a", "#d97067", RED, "#b0b0b0"]
    bars = ax.bar(range(len(vals)), vals, color=cols, width=0.62, edgecolor="k", lw=0.5, alpha=0.9)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.6, f"{v:.1f}%",
                ha="center", fontsize=9, fontweight="bold")
    ax.axhline(vals[0], color=GRAY, ls="--", lw=1.2)
    ax.annotate("", xy=(3.0, vals[3] + 0.8), xytext=(3.0, vals[0] + 0.8),
                arrowprops=dict(arrowstyle="<->", color=RED, lw=1.6))
    ax.text(3.0, (vals[0] + vals[3]) / 2, "+20.3pp", color=RED, fontsize=8.5,
            fontweight="bold", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=RED, lw=1))
    ax.set_xticks(range(len(lbl)), lbl, fontsize=7.8)
    ax.set_ylabel("1500 日年化 (%)", fontsize=9)
    ax.set_ylim(95, 150)
    ax.set_title("② 成本假设值 20 个点：滑点占 95%，佣金只占 5%", fontsize=10.5, pad=8)
    ax.grid(axis="y", alpha=0.22)
    ax.text(0.5, -0.19, "你只在低开时买、9:25 挂前收价 → 集合竞价必然按开盘价成交，滑点机制上就是 0",
            transform=ax.transAxes, ha="center", fontsize=7.8, color="#444")

    fig.suptitle("第224轮：风控已经到顶，成本还有 20 个点——3 万本金的真实年化是 130% 不是 110%",
                 fontsize=12.5, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0.03, 1, 0.93])
    out = ROOT / "第224轮_风控到顶成本还有20个点.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print("saved", out)


if __name__ == "__main__":
    main()
