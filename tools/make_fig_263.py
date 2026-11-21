"""第263轮配图: edge 时间稳定性 + 剩余行情维度审计"""

from __future__ import annotations

import json
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
    res = json.load(open(ROOT / "output/exp_edge_263.json"))
    df = pd.read_csv(ROOT / "output/w262_trades.csv")
    df = df[df["ret_pct"].notna()].copy()
    df["ret_pct"] = df["ret_pct"].astype(float)
    df["buy_dt"] = pd.to_datetime(df["buy_date"].astype(str), format="%Y%m%d")
    df = df.sort_values(["buy_dt", "buy_date"]).reset_index(drop=True)

    fig, axes = plt.subplots(1, 4, figsize=(17.6, 4.4),
                             gridspec_kw={"width_ratios": [1.35, 1.0, 1.25, 1.15]})

    # ---- 1 滚动 100 笔单笔均值 ----
    ax = axes[0]
    roll = pd.DataFrame(res["rolling"])
    roll["d"] = pd.to_datetime(roll["d"])
    ax.plot(roll["d"], roll["mean"], color=BLUE, lw=1.4)
    ax.axhline(res["overall"]["mean_ret"], color=RED, lw=1.2, ls="--")
    ax.axhline(0, color=GREY, lw=0.8, ls=":")
    ax.text(roll["d"].iloc[2], res["overall"]["mean_ret"] + 0.22,
            f"全样本 {res['overall']['mean_ret']:.2f}%", color=RED, fontsize=9)
    ax.text(roll["d"].iloc[2], -2.4, "2017 / 2019 两个枯水年", color=DARK, fontsize=8.5)
    ax.set_title("滚动 100 笔单笔均值\n(前357笔 +1.06% → 后357笔 +1.41%, 不衰减)",
                 fontsize=10.5)
    ax.set_ylabel("单笔收益 %", fontsize=10)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(alpha=0.25)

    # ---- 2 gap 五档 ----
    ax = axes[1]
    gq = res["gap_quintiles"]
    labels = [f"q{q['q']}\n[{q['lo']:+},{q['hi']:+}]" for q in gq]
    vals = [q["mean_ret"] for q in gq]
    cols = [GREEN if q["lo"] < 0 else RED for q in gq]
    bars = ax.bar(range(len(vals)), vals, color=cols, width=0.68, alpha=0.9)
    ax.axhline(res["overall"]["mean_ret"], color=GREY, lw=0.9, ls="--")
    for bb, v in zip(bars, vals):
        ax.text(bb.get_x() + bb.get_width() / 2, v + 0.06, f"{v:.2f}",
                ha="center", va="bottom", fontsize=8.4, color=DARK)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, fontsize=7.8)
    ax.set_title("T+1 隔夜缺口五档单笔\n(深低开 +1.84 → 高开 +0.75, 已由仓位规则捕获)",
                 fontsize=10.5)
    ax.set_ylabel("单笔收益 %", fontsize=10)
    ax.grid(axis="y", alpha=0.25)

    # ---- 3 逐年 ----
    ax = axes[2]
    yr = res["yearly"]
    ys = sorted(yr.keys())
    yvals = [yr[y]["mean_ret"] for y in ys]
    cols3 = [RED if v >= 0 else GREEN for v in yvals]
    bars = ax.bar(range(len(ys)), yvals, color=cols3, width=0.72, alpha=0.9)
    ax.axhline(0, color=DARK, lw=1)
    for bb, v, y in zip(bars, yvals, ys):
        ax.text(bb.get_x() + bb.get_width() / 2, v + 0.12, f"{v:+.1f}",
                ha="center", va="bottom", fontsize=7.6, color=DARK)
    ax.set_xticks(range(len(ys)))
    ax.set_xticklabels([str(y)[2:] for y in ys], fontsize=8)
    ax.set_title("逐年单笔均值\n(2017 胜率 34%、2019 胜率 39% 是两个枯水年)", fontsize=10.5)
    ax.set_ylabel("单笔收益 %", fontsize=10)
    ax.grid(axis="y", alpha=0.25)

    # ---- 4 卖出日开盘 vs 收盘 ----
    ax = axes[4 - 1] if False else axes[3]
    s = df[df["sell_px_open"].notna() & df["sell_px_close"].notna()].copy()
    s["open_ret"] = (s["sell_px_open"] / s["buy_px"] - 1) * 100
    s["close_ret"] = (s["sell_px_close"] / s["buy_px"] - 1) * 100
    drift = (s["close_ret"] - s["open_ret"]).to_numpy()
    ax.hist(drift, bins=45, color=GREEN, alpha=0.7, edgecolor="white", linewidth=0.4)
    ax.axvline(0, color=DARK, lw=1)
    ax.axvline(drift.mean(), color=RED, lw=1.4, ls="--")
    ax.text(drift.mean() + 0.25, ax.get_ylim()[1] * 0.9,
            f"均值 {drift.mean():+.2f}%\n收盘更优 {res['sell']['n_close_better']}/"
            f"{res['sell']['n']}", color=RED, fontsize=8.8, va="top")
    ax.set_title("T+2 开盘卖 vs 收盘卖\n(收盘卖每笔多赚 0.70pp)", fontsize=10.5)
    ax.set_xlabel("收盘卖 - 开盘卖 (pp)", fontsize=9)
    ax.set_ylabel("笔数", fontsize=10)
    ax.grid(axis="y", alpha=0.25)

    fig.text(0.5, -0.04,
             "口径: 714 笔已平仓交易(2014-07 ~ 2026-11), 单吊满仓, T+1 开盘买 / T+2 收盘卖。"
             "Mann-Kendall tau=-0.010 (p=0.68), OLS 斜率 +0.1pp/千笔 (p=0.91) —— edge 无显著趋势。",
             ha="center", fontsize=8.6, color="#5d6d7e")

    fig.tight_layout()
    out = ROOT / "第263轮_edge不衰减.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print("saved", out)


if __name__ == "__main__":
    main()
