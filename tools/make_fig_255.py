"""第255轮: "评分顶尖"这条路的价格表

① 严格化阶梯: 8 个方案的年化/卡玛/笔数 —— 越严越差, 单笔越高的代价
② 评分这把尺子: 五档单笔收益无单调 + 右尾贡献(前 5% 笔 = 51% 总收益)
"""
from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RED, DARK, GREY, LIGHT, BLUE = "#c0392b", "#2c3e50", "#95a5a6", "#d5dbdf", "#2e6da4"
plt.rcParams["axes.unicode_minus"] = False
for fp in ("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
           "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
           "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"):
    if Path(fp).exists():
        font_manager.fontManager.addfont(fp)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()
        break
plt.rcParams["figure.dpi"] = 130

# 2500 日, 正式口径(含补位), tools/bt_strict_255.py
ROWS = [
    ("A 254现状", 96.54, -17.21, 5.61, 557, 1.48),
    ("B 退回253", 96.38, -17.21, 5.60, 545, 1.52),
    ("C 只做21+", 84.80, -17.21, 4.93, 481, 1.56),
    ("D 只做31+", 75.92, -17.21, 4.41, 422, 1.63),
    ("E 只做41+", 79.08, -17.21, 4.59, 384, 1.79),
    ("F 关补位", 91.06, -17.41, 5.23, 507, 1.46),
    ("G 评分门1.2", 67.95, -25.27, 2.69, 469, 1.34),
    ("H 评分门1.4", 48.41, -27.74, 1.74, 393, 1.25),
]


def fig_ladder():
    names = [r[0] for r in ROWS]
    ann = [r[1] for r in ROWS]
    dd = [r[2] for r in ROWS]
    cal = [r[3] for r in ROWS]
    n = [r[4] for r in ROWS]

    fig, ax = plt.subplots(figsize=(11.2, 5.4))
    x = np.arange(len(names))
    cols = [BLUE, RED] + [GREY] * 4 + ["#8e44ad", "#8e44ad"]
    bars = ax.bar(x, ann, color=cols, width=0.62, zorder=2)
    for i, (b, a, c) in enumerate(zip(bars, ann, cal)):
        ax.text(b.get_x() + b.get_width() / 2, a + 2.2, f"{a:.1f}%",
                ha="center", va="bottom", fontsize=9.5, color=DARK, fontweight="bold")
        ax.text(b.get_x() + b.get_width() / 2, 3, f"卡玛 {c:.2f}",
                ha="center", va="bottom", fontsize=8.5, color="white")
        ax.text(b.get_x() + b.get_width() / 2, -9.5, f"{n[i]}笔",
                ha="center", va="top", fontsize=8, color="#5d6d7e")
    ax.axhline(96.54, color=BLUE, lw=1.0, ls=":", zorder=1)
    ax.text(len(names) - 0.4, 98.4, "现状 96.5%", ha="right", va="bottom",
            fontsize=8.5, color=BLUE)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9.5)
    ax.set_ylabel("年化收益率 %", fontsize=10)
    ax.set_ylim(-18, 112)
    ax.set_title("把「更严」的每个旋钮单独拧一遍：2500 日，含补位正式口径",
                 fontsize=12.5, color=DARK, pad=12, loc="left", fontweight="bold")
    ax.grid(axis="y", color=LIGHT, lw=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.annotate("本轮唯一动作：\n零成本退掉 254 的放宽", xy=(1, 96.38), xytext=(2.35, 55),
                fontsize=9.2, color=RED, ha="left",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.2,
                                connectionstyle="arc3,rad=0.15"))
    ax.annotate("评分门槛越严\n收益塌、回撤还放大 8pp", xy=(6.98, 71), xytext=(5.72, 86),
                fontsize=9.2, color="#8e44ad", ha="left", va="bottom",
                arrowprops=dict(arrowstyle="->", color="#8e44ad", lw=1.2))
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "第255轮_严格化阶梯.png"), bbox_inches="tight")
    plt.close(fig)


def fig_ruler():
    o = pickle.load(open(os.path.join(ROOT, "output/w255_strict_2500.pkl"), "rb"))
    tr = [z for z in o if z["name"].startswith("A")][0]["trades"]
    d = tr[(tr["pos_frac"].fillna(0) > 0) & tr["ret_pct"].notna()].copy()
    d["score"] = pd.to_numeric(d["score"], errors="coerce")
    d["ret"] = pd.to_numeric(d["ret_pct"], errors="coerce")
    d = d[d["score"].notna()]
    d["q"] = pd.qcut(d["score"], 5, labels=["Q1 低", "Q2", "Q3", "Q4", "Q5 高"])
    g = d.groupby("q", observed=True)["ret"].agg(["count", "mean"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.9),
                                   gridspec_kw={"width_ratios": [1, 1.05]})
    xs = np.arange(len(g))
    cols = [GREY] * 4 + [RED]
    ax1.bar(xs, g["mean"], color=cols, width=0.6, zorder=2)
    for i, (v, c) in enumerate(zip(g["mean"], g["count"])):
        ax1.text(i, v + 0.03, f"{v:.2f}%", ha="center", va="bottom",
                 fontsize=10, fontweight="bold", color=DARK)
        ax1.text(i, 0.05, f"{c}笔", ha="center", va="bottom", fontsize=8.2, color="white")
    ax1.axhline(d["ret"].mean(), color=BLUE, ls="--", lw=1.1,
                label=f"全体均值 {d['ret'].mean():.2f}%")
    ax1.legend(loc="upper center", bbox_to_anchor=(0.52, 1.0), fontsize=8.6,
               frameon=False)
    ax1.set_xticks(xs)
    ax1.set_xticklabels(g.index, fontsize=9.5)
    ax1.set_ylabel("单笔收益均值 %", fontsize=10)
    ax1.set_ylim(0, max(g["mean"]) * 1.22)
    ax1.set_title(f"按评分五档切：最高分档不是最好的（相关 {d['score'].corr(d['ret']):+.3f}）",
                  fontsize=11, color=DARK, loc="left", pad=10)
    ax1.grid(axis="y", color=LIGHT, lw=0.8, zorder=0)
    for s in ("top", "right"):
        ax1.spines[s].set_visible(False)

    s = d["ret"].sort_values(ascending=False).reset_index(drop=True)
    cum = s.cumsum() / s.sum() * 100
    frac = (np.arange(len(s)) + 1) / len(s) * 100
    ax2.plot(frac, cum, color=RED, lw=2.2, zorder=3)
    ax2.fill_between(frac, 0, cum, color=RED, alpha=0.08)
    ax2.plot([0, 100], [0, 100], color=GREY, ls=":", lw=1.1, zorder=1)
    ax2.text(64, 52, "如果每笔贡献相同\n（45° 线）", fontsize=8.6, color=GREY)
    k = int(round(len(s) * 0.05))
    ax2.scatter([5], [cum.iloc[k - 1]], color=DARK, s=42, zorder=5)
    ax2.annotate(f"前 5%（{k} 笔）贡献 {cum.iloc[k-1]:.0f}%",
                 xy=(5, cum.iloc[k - 1]), xytext=(11, 28), fontsize=9.4, color=DARK,
                 arrowprops=dict(arrowstyle="->", color=DARK, lw=1.1))
    hit = int(np.argmax(cum.values >= 100)) + 1
    ax2.scatter([hit / len(s) * 100], [100], color=BLUE, s=44, zorder=5)
    ax2.annotate(f"到第 {hit} 笔（{(hit)/len(s)*100:.0f}%）覆盖 100%",
                 xy=(hit / len(s) * 100, 100), xytext=(20, 62), fontsize=9.0, color=BLUE,
                 arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.1))
    peak = int(np.argmax(cum.values))
    ax2.scatter([(peak + 1) / len(s) * 100], [cum.iloc[peak]], color=RED, s=36, zorder=5)
    ax2.annotate(f"峰值在第 {peak+1} 笔（{(peak+1)/len(s)*100:.0f}%）：累计 {cum.iloc[peak]:.0f}%\n"
                 f"——最后 {100-(peak+1)/len(s)*100:.0f}% 的笔把它亏回 100%",
                 xy=(93, 104), xytext=(30, 128), fontsize=9.0, color=DARK,
                 arrowprops=dict(arrowstyle="->", color=DARK, lw=1.1,
                                 connectionstyle="arc3,rad=-0.18"))
    ax2.set_xlabel("按单笔收益从高到低排序的成交笔数占比 %", fontsize=9.6)
    ax2.set_ylabel("累计贡献总收益 %", fontsize=10)
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 178)
    ax2.set_title("钱是从右尾来的，不是从「选得准」来的", fontsize=11,
                  color=DARK, loc="left", pad=10)
    ax2.grid(color=LIGHT, lw=0.8, zorder=0)
    for sp in ("top", "right"):
        ax2.spines[sp].set_visible(False)

    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "第255轮_评分这把尺子.png"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_ladder()
    fig_ruler()
    print("done")
