"""第232轮配图：alpha 衰减、日内结构无效、卖出时点对比"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP"]
plt.rcParams["axes.unicode_minus"] = False


def fig_decay():
    d = pd.read_csv(ROOT / "output" / "w232d_decay.csv")
    ks = [f"d{k}" for k in range(1, 7)]
    m = [d[k].mean() for k in ks]
    wr = [(d[k] > 0).mean() * 100 for k in ks]
    fig, a1 = plt.subplots(figsize=(11, 5.8))
    cols = ["#c0392b" if v > 0 else "#7f8c8d" for v in m]
    bars = a1.bar(range(1, 7), m, width=0.62, color=cols)
    a1.axhline(0, color="#333", lw=1)
    a1.set_xticks(range(1, 7))
    a1.set_xticklabels([f"第{i}天" for i in range(1, 7)])
    a1.set_ylabel("当日贡献 %（收盘→收盘）")
    a1.set_title("alpha 衰减：正贡献只到第 2 天，第 3 天转负", fontsize=12)
    for x, v, w in zip(range(1, 7), m, wr):
        a1.annotate(f"{v:+.2f}%\n胜率{w:.0f}%", (x, v),
                    xytext=(0, 10 if v > 0 else -26), textcoords="offset points",
                    ha="center", fontsize=9, color="#2c3e50")
    a1.grid(alpha=0.2, axis="y")
    a1.set_ylim(min(m) * 1.9, max(m) * 1.45)
    # 现行持仓区间标注
    a1.axvspan(0.55, 2.45, color="#c0392b", alpha=0.07)
    a1.annotate("现行持仓区间\nT+1 开盘买 · T+2 收盘卖", (1.5, max(m) * 1.28),
                ha="center", fontsize=10, color="#c0392b")
    fig.text(0.5, 0.015, "现行单笔 1.34%（裸口径）= 入场价白拿低开 0.61% + T+1 日内 0.85 + T+2 0.48。"
             "改成「T 日收盘买、T+1 收盘卖」只赚 0.22%——少赚的 1.11pp ≈ 低开 0.61 + 第2天 0.48",
             ha="center", fontsize=9.5, color="#444")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(ROOT / "第232轮_alpha衰减.png", dpi=150)
    print("已写 第232轮_alpha衰减.png")


def fig_feat():
    d = pd.read_csv(ROOT / "output" / "w232_feat.csv")
    cols = ["am", "am_vol", "low_at", "tail30", "tail_vol", "pm", "vwap_dev"]
    names = ["早盘涨跌", "早盘量占比", "最低点位置", "尾盘30分", "尾盘量占比", "午后涨跌", "收盘偏离VWAP"]
    base = d["ret"].mean()
    mat = []
    for c in cols:
        s = d[[c, "ret"]].dropna().sort_values(c)
        s["b"] = pd.qcut(s[c], 5, labels=False, duplicates="drop")
        mat.append(list(s.groupby("b")["ret"].mean().values))
    mat = np.array(mat)

    fig, a1 = plt.subplots(figsize=(11.5, 5.8))
    w = 0.16
    xs = np.arange(len(cols))
    colors = ["#c0392b", "#d98880", "#95a5a6", "#7fb3d5", "#2471a3"]
    for j in range(5):
        a1.bar(xs + (j - 2) * w, mat[:, j], width=w, color=colors[j],
               label=f"第{j+1}档" + ("（最低）" if j == 0 else "（最高）" if j == 4 else ""))
    a1.axhline(base, color="#333", ls="--", lw=1.2)
    a1.annotate(f"全部样本 {base:.2f}%", (-0.45, base), xytext=(0, 6),
                textcoords="offset points", ha="left", fontsize=9, color="#333")
    a1.annotate("单档尖峰（如尾盘量占比第4档）不是规律，\n是 559 笔被切成 5 份后每份约 112 笔的噪声",
                (4, mat[4].max()), xytext=(24, -6), textcoords="offset points",
                fontsize=8.5, color="#7f8c8d",
                arrowprops=dict(arrowstyle="->", color="#7f8c8d", lw=1))
    a1.set_xticks(xs)
    a1.set_xticklabels(names, fontsize=10)
    a1.set_ylabel("单笔收益 %")
    a1.set_title("信号日的日内路径，与次日收益没有关系（7 个特征 |t| 全 < 1.1）", fontsize=12)
    a1.legend(fontsize=8, ncol=5, loc="upper center", frameon=False,
              bbox_to_anchor=(0.5, -0.10))
    a1.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(ROOT / "第232轮_日内结构无效.png", dpi=150)
    print("已写 第232轮_日内结构无效.png")


def fig_exit():
    d = pd.read_csv(ROOT / "output" / "w232c_exit.csv")
    fig, a1 = plt.subplots(figsize=(11.5, 5.6))
    xs = range(len(d))
    cols = ["#c0392b" if "现行" in s else "#7f8c8d" for s in d["规则"]]
    a1.bar(xs, d["单笔"], width=0.58, color=cols)
    a1.axhline(d.loc[0, "单笔"], color="#c0392b", ls="--", lw=1.1, alpha=0.5, zorder=0)
    a1.set_xticks(list(xs))
    a1.set_xticklabels([s.replace("(", "\n(") for s in d["规则"]], fontsize=8.5)
    a1.set_ylabel("单笔收益 %")
    a1.set_ylim(0, max(d["单笔"]) * 1.32)
    a1.set_title("有没有「T+1 收盘时提前跑」更划算的时候？没有", fontsize=12)
    for x, (v, dv) in enumerate(zip(d["单笔"], d["Δ单笔"])):
        a1.annotate(f"{v:.2f}%\n{dv:+.2f}", (x, v), xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=8.5, color="#2c3e50", zorder=5,
                    path_effects=[pe.withStroke(linewidth=3, foreground="white")])
    a1.grid(alpha=0.2, axis="y")
    fig.text(0.5, 0.015, "T+1 收盘亏的 212 笔：T+2 开盘卖 −2.06% vs 收盘卖 −1.68% —— "
             "亏了就跑，跑在当天最低点上", ha="center", fontsize=9.5, color="#444")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(ROOT / "第232轮_卖出时点.png", dpi=150)
    print("已写 第232轮_卖出时点.png")


if __name__ == "__main__":
    fig_decay()
    fig_feat()
    fig_exit()
