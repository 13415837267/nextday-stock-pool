#!/usr/bin/env python3
"""第222轮 图: T+0 ETF 替代可转债的判决。"""
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

RED, GREEN, BLUE, GRAY, ORANGE = "#d62728", "#2ca02c", "#1f77b4", "#7f7f7f", "#ff7f0e"


def panel1(ax: plt.Axes) -> None:
    """T+0 四类 ETF 的 gap -> 日内收益。"""
    d = pd.read_parquet(ROOT / "data" / "etf" / "daily.parquet")
    d = d.sort_values(["code", "date"]).reset_index(drop=True)
    g = d.groupby("code", observed=True, sort=False)
    d["pc"] = g["close"].shift(1)
    d["gap"] = d["open"] / d["pc"] - 1
    d["intr"] = d["close"] / d["open"] - 1
    d = d.dropna(subset=["pc"])
    d = d[d["gap"].between(-0.05, 0.03)]

    edges = np.array([-0.05, -0.03, -0.02, -0.015, -0.01, -0.007, -0.004,
                      -0.002, 0, 0.002, 0.004, 0.007, 0.01, 0.015, 0.02, 0.03])
    mids = (edges[:-1] + edges[1:]) / 2 * 100

    style = {"qdii": ("跨境QDII", RED, "o"), "bond": ("债券ETF", BLUE, "s"),
             "gold": ("黄金ETF", ORANGE, "^"), "money": ("货币ETF", GRAY, "D")}
    for cls, (lab, c, m) in style.items():
        s = d[d["cls"] == cls].copy()
        s["b"] = pd.cut(s["gap"], edges)
        agg = s.groupby("b", observed=True).agg(v=("intr", "mean"), n=("intr", "size"))
        agg = agg[agg["n"] >= 100]
        ax.plot(mids[:len(agg)] if False else [iv.mid * 100 for iv in agg.index],
                agg["v"] * 1e4, marker=m, ms=4, lw=1.6, color=c, label=lab)

    ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.5)
    ax.axvline(0, color="k", lw=0.8, ls=":", alpha=0.5)
    # 成本带: 低价ETF一个tick就要 10~20bp
    ax.axhspan(-20, 20, color=GRAY, alpha=0.13, zorder=0)
    ax.text(3.05, 27, "灰带 = 1~2 个 tick 的\n往返成本", fontsize=7, color="#555",
            ha="right", va="bottom")
    ax.set_xlabel("T 日开盘缺口 gap (%)", fontsize=9)
    ax.set_ylabel("日内收益 close/open − 1 (bp)", fontsize=9)
    ax.set_title("① T+0 四类 ETF：跨境是反向动量，其余在成本带内", fontsize=10.5, pad=8)
    ax.legend(fontsize=7.5, loc="upper left", framealpha=0.92)
    ax.grid(alpha=0.25)
    ax.set_xlim(-5.2, 3.2)
    ax.set_ylim(-55, 70)


def panel2(ax: plt.Axes) -> None:
    """股票ETF: 流动性 x gap 的净收益热力。"""
    mat = np.array([[212, 122, 74, 41],
                    [4, 34, 50, 21],
                    [-82, 27, -12, 23],
                    [-112, -72, -16, -11]], dtype=float)
    rows = ["成交额<1亿\n(中位0.00亿)", "1~5亿\n(中位2.0亿)",
            "5~20亿\n(中位8.9亿)", ">20亿\n(中位33亿)"]
    cols = ["跌超3%", "−3~−2%", "−2~−1.5%", "−1.5~−1%"]

    vmax = 120
    im = ax.imshow(np.clip(mat, -vmax, vmax), cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(cols)), cols, fontsize=8.5)
    ax.set_yticks(range(len(rows)), rows, fontsize=8)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            ax.text(j, i, f"{v:+.0f}", ha="center", va="center", fontsize=10,
                    fontweight="bold", color="white" if abs(v) > 60 else "black")
    ax.set_title("② 股票ETF 的低开反转只活在僵尸样本里（净 bp）", fontsize=10.5, pad=8)
    ax.text(0.5, -0.19, "成交额<1亿的 2.7 万笔“大赚”，是买卖价差跳动与除权造成的假缺口；\n"
                        "真正能成交的 >20 亿宽基，深低开一律为负",
            transform=ax.transAxes, ha="center", fontsize=7.8, color="#444")


def panel3(ax: plt.Axes) -> None:
    """约束 -> 年化上限阶梯。统一 1500 日口径(≈近6年), 与 200% 测算同源。"""
    labels = ["现行·无杠杆\n无可转债", "＋可转债T+0\n填空仓期", "＋1.53× 两融杠杆"]
    vals = [111.8, 135.7, 200.0]
    mdd = [-17.7, -17.3, -25.7]
    cols = [GRAY, BLUE, RED]
    x = np.arange(3)
    bars = ax.bar(x, vals, width=0.55, color=cols, alpha=0.85, edgecolor="k", lw=0.6)
    for b, v, m in zip(bars, vals, mdd):
        ax.text(b.get_x() + b.get_width() / 2, v + 5, f"{v:.0f}%",
                ha="center", fontsize=11, fontweight="bold")
        ax.text(b.get_x() + b.get_width() / 2, v / 2 + 6, f"回撤\n{m:.0f}%",
                ha="center", va="center", fontsize=8.5, color="white", fontweight="bold")
    ax.axhline(200, color=RED, ls="--", lw=1, alpha=0.6)
    ax.text(-0.42, 205, "200% 目标线", fontsize=8, color=RED, ha="left")
    ax.set_xticks(x, labels, fontsize=8.5)
    ax.set_ylabel("1500 日口径年化 (%)", fontsize=9)
    ax.set_title("③ 200% 需要哪一级权限", fontsize=10.5, pad=8)
    ax.grid(axis="y", alpha=0.25)
    # 门槛标注
    for i, t in enumerate(["门槛 0", "2年经验+10万", "50万+6个月"]):
        ax.text(i, 9, t, ha="center", fontsize=7.5, color="#333" if i == 0 else "white",
                fontweight="bold")
    ax.set_ylim(0, 228)


def main() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2))
    panel1(axes[0])
    panel2(axes[1])
    panel3(axes[2])
    fig.suptitle("第222轮：可转债权限作废后，T+0 ETF 能不能顶上？—— 三条路全否",
                 fontsize=13, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0.02, 1, 0.945])
    out = ROOT / "第222轮_ETF能否替代可转债.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print("saved", out)


if __name__ == "__main__":
    main()
