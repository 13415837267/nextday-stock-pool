"""第261轮配图: 一年长什么样 —— 滚动收益分布 / 逐年 / 水下曲线"""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]

for fp in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
           "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
           "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"):
    if Path(fp).exists():
        font_manager.fontManager.addfont(fp)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

RED, GREEN, DARK, GREY = "#c0392b", "#1e8449", "#2c3e50", "#95a5a6"
BLUE = "#2e6da4"


def main() -> None:
    d = pickle.load(open(ROOT / "output/w259_fbgate_3200.pkl", "rb"))
    A = [x for x in d if x["name"].startswith("A ")][0]
    nav = np.array([x["nav"] for x in A["equity"]], dtype=float)
    ds = pd.to_datetime([x["d"] for x in A["equity"]])
    e = pd.Series(nav, index=ds)
    dd = (e / e.cummax() - 1) * 100

    W = 252
    g = np.array([nav[i + W] / nav[i] - 1 for i in range(len(nav) - W)]) * 100

    yr = (e.resample("YE").last() / e.resample("YE").last().shift(1) - 1) * 100
    yr = yr.dropna()
    yr.index = yr.index.year

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.6),
                             gridspec_kw={"width_ratios": [1.15, 1.0, 1.45]})

    # ---- 1 滚动一年收益分布 ----
    ax = axes[0]
    ax.hist(g, bins=60, color=BLUE, alpha=0.75, edgecolor="white", linewidth=0.4)
    med, p10, p5 = np.median(g), np.percentile(g, 10), np.percentile(g, 5)
    ax.axvline(med, color=RED, lw=2)
    ax.text(med + 12, ax.get_ylim()[1] * 0.88, f"中位 {med:.0f}%", color=RED,
            fontsize=11, fontweight="bold", va="top")
    for v in (p10, p5):
        ax.axvline(v, color=GREY, ls="--", lw=1.1)
    ax.axvline(0, color=DARK, lw=1, ls=":")
    ax.text(430, ax.get_ylim()[1] * 0.82,
            f"p25 {np.percentile(g,25):.0f}%\np10 {p10:.0f}%\np5 {p5:.0f}%",
            fontsize=9.2, color="#6b7785", va="top", ha="left", linespacing=1.5)
    ax.set_xlim(-40, 640)
    ax.set_title("随便挑一年, 收益是多少\n(3200 日里每个 252 日窗口, 共 2948 个)", fontsize=11.5)
    ax.set_xlabel("这一年赚多少 (%)", fontsize=10)
    ax.set_ylabel("窗口数", fontsize=10)
    ax.grid(axis="y", alpha=0.25)

    # ---- 2 逐年 ----
    ax = axes[1]
    cols = [RED if v >= 0 else GREEN for v in yr.values]
    bars = ax.bar(range(len(yr)), yr.values, color=cols, width=0.72, alpha=0.9)
    ax.set_xticks(range(len(yr)))
    ax.set_xticklabels([str(y)[2:] for y in yr.index], fontsize=9)
    ax.axhline(0, color=DARK, lw=1)
    for bb, v in zip(bars, yr.values):
        ax.text(bb.get_x() + bb.get_width() / 2, v + 12, f"{v:.0f}", ha="center",
                va="bottom", fontsize=8.2, color=DARK)
    ax.set_ylim(-40, max(yr.values) * 1.16)
    ax.set_title("按自然年\n(2017 / 2018 / 2019 连续三年 1% / 19% / 20%)", fontsize=11.5)
    ax.set_xlabel("年 (20xx)", fontsize=10)
    ax.set_ylabel("年度收益 (%)", fontsize=10)
    ax.grid(axis="y", alpha=0.25)

    # ---- 3 水下 ----
    ax = axes[2]
    ax.fill_between(dd.index, dd.values, 0, color=GREEN, alpha=0.35, lw=0)
    ax.plot(dd.index, dd.values, color=GREEN, lw=1.3)
    ax.axhline(0, color=DARK, lw=1)
    ax.set_ylim(-26, 2)
    ax.set_title("离上一次新高有多远\n(最长 503 天没创新高: 2017-07 ~ 2018-11)", fontsize=11.5)
    ax.set_ylabel("回撤 (%)", fontsize=10)
    ax.grid(alpha=0.25)
    i0 = dd.idxmin()
    ax.annotate("最深 −22.9%\n2015-07-07", (i0, dd.min()), textcoords="offset points",
                xytext=(16, 12), fontsize=9, color=GREEN, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=GREEN, lw=0.9))

    fig.text(0.5, -0.035,
             "口径: 3200 日回测(2013-09 ~ 2026-11), 730 笔, 单吊满仓, T+1 开盘买 / T+2 收盘卖。"
             "收益已扣双边 0.1% 滑点与费用。",
             ha="center", fontsize=8.8, color="#5d6d7e")

    fig.tight_layout()
    out = ROOT / "第261轮_一年长什么样.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print("saved", out)


if __name__ == "__main__":
    main()
