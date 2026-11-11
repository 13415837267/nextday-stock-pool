# -*- coding: utf-8 -*-
"""第199轮配图: 波动率维度是真的, 但它不可交易"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
OUT = Path("/workspace/第199轮_回撤和收益长在同一个地方.png")


def trades(days: int, with_f: bool = True) -> pd.DataFrame:
    t = pd.read_csv(ROOT / "output" / f"exp_factor_197_trades_{days}.csv",
                    dtype={"buy": str, "sell": str})
    t["buy"] = t.buy.str.replace("-", "")
    m = pd.read_csv(ROOT / "output" / "exp_mkt_199.csv", dtype={"d": str}).set_index("d")
    idx = list(m.index)
    pos = {d: i for i, d in enumerate(idx)}
    t["sig"] = [idx[pos[b] - 1] if b in pos and pos[b] > 0 else None for b in t.buy]
    t = t[t.sig.notna()].copy()
    f = m.reindex(t.sig)
    t["vp"] = f.vol_pct.values
    return t[t.vp.notna()].copy()


def fig():
    fig = plt.figure(figsize=(13.2, 10.2))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.05, 1.0, 1.15], hspace=0.72, wspace=0.28)

    # (a) 双窗口: 波动率分位 → 单笔均值
    ax = fig.add_subplot(gs[0, :])
    bins = [0, .2, .4, .6, .8, 1.01]
    labs = ["Q1\n最低", "Q2", "Q3", "Q4", "Q5\n最高"]
    for k, (days, c, mk) in enumerate([(1500, "#c0392b", "o"), (3000, "#7f8c8d", "s")]):
        t = trades(days)
        t["b"] = pd.cut(t.vp, bins, labels=labs)
        g = t.groupby("b", observed=False).ret.agg(["mean", "count"])
        x = np.arange(len(g)) + (k - .5) * .22
        ax.bar(x, g["mean"], .2, color=c, alpha=.85,
               label=f"{days} 日窗口（{int(g['count'].sum())} 笔）")
        for xi, yi, ni in zip(x, g["mean"], g["count"]):
            ax.text(xi, yi + .06, f"{yi:.2f}%\n{ni}笔", ha="center", va="bottom", fontsize=7.6,
                    color=c)
    ax.set_xticks(np.arange(len(labs)))
    ax.set_xticklabels(labs)
    ax.set_ylabel("单笔均值（%）")
    ax.set_ylim(0, 2.95)
    ax.set_title("(a) 市场 20 日波动率分位 → 主策略单笔期望：双窗口单调上升", fontsize=11.5, loc="left")
    ax.legend(fontsize=8.5, frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=.25, ls=":")
    ax.set_xlabel("信号日全市场 20 日已实现波动率的历史分位（250 日滚动）", fontsize=9)

    # (b) 三种用法: 年化差(pp)
    ax = fig.add_subplot(gs[1, 0])
    uses = ["低波空仓\n(vp<0.10)", "低波降仓\n(vp<0.10→0.5)", "低波换副策略\n(0.10/0.40)"]
    v1500 = [-18.7, -9.5, 6.3]
    v3000 = [-8.9, -4.5, 1.2]
    x = np.arange(3)
    ax.bar(x - .18, v1500, .34, color="#c0392b", alpha=.85, label="1500 日")
    ax.bar(x + .18, v3000, .34, color="#7f8c8d", alpha=.85, label="3000 日")
    ax.axhline(0, color="k", lw=.8)
    for xi, yi in list(zip(x - .18, v1500)) + list(zip(x + .18, v3000)):
        ax.text(xi, yi + (0.6 if yi > 0 else -0.6), f"{yi:+.1f}", ha="center",
                va="bottom" if yi > 0 else "top", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(uses, fontsize=8.5)
    ax.set_ylabel("年化差 vs 现行（pp）")
    ax.set_ylim(-24, 11)
    ax.set_title("(b) 三条路：两条是纯损失，换腿也只是微增", fontsize=11, loc="left")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=.25, ls=":")

    # (c) 回撤期间 vp 永远在高位
    ax = fig.add_subplot(gs[1, 1])
    m = pd.read_csv(ROOT / "output" / "exp_mkt_199.csv", dtype={"d": str}).set_index("d")
    for days, f, c in [(1500, "backtest", "#c0392b"), (3000, "backtest_3000_bak", "#2c3e50")]:
        d = json.loads((ROOT / "output" / f"{f}.json").read_text(encoding="utf-8"))
        eq = pd.Series({x["d"].replace("-", ""): x["nav"] for x in d["equity"]}).sort_index()
        dd = eq / eq.cummax() - 1
        i = dd.idxmin()
        peak = eq[:i].idxmax()
        vp = m.vol_pct.reindex([peak, i]).values
        ax.plot([0, 1], vp, "o-", color=c, lw=2, ms=7,
                label=f"{days} 日最大回撤（{dd.min()*100:.1f}%）")
        for xx, yy in zip([0, 1], vp):
            ax.text(xx, yy - .075, f"{yy:.2f}", fontsize=8.5, color=c, ha="center", va="top")
    ax.axhline(.4, color="#27ae60", ls="--", lw=1)
    ax.text(.52, .30, "vp=0.40 高波线", fontsize=8, color="#27ae60", ha="center")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["回撤起点", "回撤谷底"], fontsize=9)
    ax.set_xlim(-.28, 1.28)
    ax.set_ylim(0, 1)
    ax.set_ylabel("波动率分位 vp")
    ax.set_title("(c) 两次最大回撤，全程 vp≥0.4", fontsize=11, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    ax.grid(axis="y", alpha=.25, ls=":")

    # (d) 净值 + vp 阴影
    ax = fig.add_subplot(gs[2, :])
    d = json.loads((ROOT / "output" / "backtest.json").read_text(encoding="utf-8"))
    eq = pd.Series({x["d"].replace("-", ""): x["nav"] for x in d["equity"]}).sort_index()
    vp = m.vol_pct.reindex(eq.index).ffill()
    ax.plot(range(len(eq)), eq.values, color="#c0392b", lw=1.9, label="主策略净值（对数轴）")
    ax.set_yscale("log")
    ax.set_ylabel("净值（倍）")
    ax2 = ax.twinx()
    ax2.fill_between(range(len(vp)), 0, vp.values, color="#3498db", alpha=.10, lw=0)
    ax2.plot(range(len(vp)), vp.values, color="#3498db", lw=.9, alpha=.75, label="波动率分位 vp")
    ax2.set_ylim(0, 1.02)
    ax2.set_ylabel("vp", color="#3498db", labelpad=2)
    ax2.tick_params(axis="y", colors="#3498db")
    step = max(1, len(eq) // 9)
    ax.set_xticks(range(0, len(eq), step))
    ax.set_xticklabels([eq.index[i][:4] + "-" + eq.index[i][4:6] for i in range(0, len(eq), step)],
                       fontsize=8.5)
    ax.set_title("(d) 1500 日：净值台阶几乎都踩在蓝色高波区（vp 高）上", fontsize=11, loc="left")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8.5, frameon=False, loc="upper left")
    ax.grid(alpha=.2, ls=":")

    fig.suptitle("第 199 轮：波动率维度是真的，但回撤和收益长在同一个地方",
                 fontsize=14.5, y=.985, x=.012, ha="left", fontweight="bold")
    fig.text(.012, .004,
             "口径：信号日（T 日收盘）算全市场 20 日已实现波动率，取其 250 日滚动分位 vp；"
             "逐笔收益来自 197 轮回测（含双边成本），无未来函数。\n"
             "换腿 = vp≤0.10 全副策略、vp≥0.40 回主策略，每次切换扣 0.3%；"
             "块自助 1000 次（block=20 日）给 P(≤0)=0.48。",
             fontsize=7.8, color="#555")
    fig.savefig(OUT, dpi=170, bbox_inches="tight", facecolor="white")
    print("wrote", OUT)


if __name__ == "__main__":
    fig()
