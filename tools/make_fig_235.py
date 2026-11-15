"""第二百三十五轮配图: 钱从哪来 + 换评分全掉"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"


def fig1() -> None:
    tab = pd.read_csv(OUT / "w235b_scores.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(13.6, 6.4),
                               gridspec_kw={"width_ratios": [1, 1.35]})

    # ---- 左: 收益阶梯 ----
    steps = [("全市场\n(817万样本)", 0.235, "#bdc3c7"),
             ("同日同池\n(7.6万样本)", 0.579, "#7f8c8d"),
             ("现行评分 top1\n(474笔)", 1.421, "#c0392b"),
             ("oracle 上限\n(预知未来)", 15.845, "#e8dcdc")]
    xs = np.arange(len(steps))
    vals = [s[1] for s in steps]
    a.bar(xs, vals, color=[s[2] for s in steps], width=0.6,
          edgecolor=["#95a5a6", "#5d6d7e", "#922b21", "#d5b8b8"])
    a.set_yscale("log")
    a.set_ylim(0.15, 40)
    a.set_xticks(xs)
    a.set_xticklabels([s[0] for s in steps], fontsize=9.5)
    a.set_ylabel("单笔收益 %（对数轴）", fontsize=10.5)
    for x, v in zip(xs, vals):
        a.text(x, v * 1.16, f"{v:.2f}%", ha="center", va="bottom", fontsize=11,
               fontweight="bold", color="#222",
               path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])
    # 倍数注释
    a.annotate("", xy=(1, 0.60), xytext=(0, 0.24),
               arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=1.6))
    a.text(0.5, 0.34, "池 +146%", ha="center", fontsize=9, color="#2c3e50",
           path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])
    a.annotate("", xy=(2, 1.46), xytext=(1, 0.60),
               arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.6))
    a.text(1.62, 0.44, "评分 +145%", ha="center", fontsize=9, color="#c0392b",
           path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])
    a.set_title("① 每一块钱从哪来：漏斗 +146%，评分再 +145%", fontsize=11.5, pad=10)
    a.grid(axis="y", ls=":", color="#ccc", alpha=0.7)
    a.set_axisbelow(True)

    # ---- 右: 换评分 ----
    d = tab[tab["评分"] != "★现行评分(实际成交)"].copy()
    d = d[~d["评分"].str.startswith("[最佳]")]
    d = d.sort_values("单笔%")
    real = float(tab[tab["评分"] == "★现行评分(实际成交)"]["单笔%"].iloc[0])
    rnd = float(tab[tab["评分"] == "随机选一只(500次均值)"]["单笔%"].iloc[0])
    best_rnd = float(tab[tab["评分"] == "随机选一只(500次最好)"]["单笔%"].iloc[0])
    d = d[d["评分"] != "随机选一只(500次最好)"]

    name_map = {"dist_high20": "距20日高点", "bias60": "60日乖离", "ret20": "20日涨幅",
                "pct": "当日涨跌幅", "circ_mv": "流通市值", "amp": "当日振幅",
                "turnover_rate": "换手率", "close_pos": "收盘位置",
                "8因子等权(方向按234轮)": "8 因子等权组合"}
    def pretty(s: str) -> str:
        for k, v in name_map.items():
            s = s.replace(k, v)
        return s.replace("越大越好", "↑").replace("越小越好", "↓")

    labels = [pretty(s) for s in d["评分"]]
    cols = ["#c0392b" if s.startswith("8因子") else "#95a5a6" for s in d["评分"]]
    b.barh(labels, d["单笔%"], color=cols, height=0.62)
    b.axvline(rnd, color="#7f8c8d", ls="--", lw=1.3)
    b.axvline(real, color="#c0392b", ls="-", lw=2.0)
    b.axvline(best_rnd, color="#e67e22", ls=":", lw=1.4)
    b.text(real - 0.025, len(d) * 0.30, f"现行评分\n{real:.2f}%", fontsize=9.5,
           color="#c0392b", fontweight="bold", va="center", ha="right",
           path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])
    b.text(rnd - 0.02, 0.1, f"随机均值 {rnd:.2f}%", fontsize=8.5, color="#7f8c8d",
           ha="right", va="bottom",
           path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])
    b.text(best_rnd + 0.015, len(d) * 0.55, f"随机500次最好 {best_rnd:.2f}%", fontsize=8.5,
           color="#e67e22", va="center",
           path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])
    b.set_xlim(0, 1.62)
    b.set_xlabel("同日 top1 的单笔收益 %（474 个交易日，同一池同一规则）", fontsize=10.5)
    b.set_title("② 把评分换掉：最好的单因子也只有 0.83%", fontsize=11.5, pad=10)
    for y, v in enumerate(d["单笔%"]):
        b.text(v + 0.015, y, f"{v:.2f}", va="center", fontsize=8.5, color="#333",
               path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])
    b.grid(axis="x", ls=":", color="#ccc", alpha=0.7)
    b.set_axisbelow(True)
    b.tick_params(axis="y", labelsize=8.8)

    fig.text(0.5, 0.012,
             "口径：feat234 特征表，池=次日低开可成交 且 当日跌 3~6% 且 收盘位置≤0.32 且 价格3~150 且 市值≤600亿 且 换手2~12%；"
             "oracle 为每日池内收益最高者，需预知未来、不可实现",
             ha="center", fontsize=8.2, color="#666")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(ROOT / "第235轮_评分到底值多少钱.png", dpi=150)
    print("图1 ok")


if __name__ == "__main__":
    fig1()
