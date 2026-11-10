# -*- coding: utf-8 -*-
"""第185轮作图 + 年度分解: 百倍是不是靠某一年赌对的

图1 组合净值(对数轴): 100% 主 / 60-40 / 100% 副, 三条曲线 + 各自最大回撤区间标注
图2 权重网格: 横轴回撤、纵轴倍数(对数), 标出"百倍线"与 60/40 点
附: 主/副策略分年度收益表(print), 用于回答"是不是某一年撞大运"
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
plt.rcParams["font.family"] = ["Noto Serif CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = ROOT / "output"
WORK = ROOT.parent


def load(name):
    d = json.loads((OUT / f"{name}.json").read_text(encoding="utf-8"))
    s = pd.Series({str(x["d"]).replace("-", ""): float(x["nav"]) for x in d["equity"]})
    return s.sort_index(), d["metrics"]


def yearly(nav: pd.Series) -> dict:
    out = {}
    for y, g in nav.groupby(nav.index.str[:4]):
        if len(g) > 5:
            out[y] = round(float(g.iloc[-1] / g.iloc[0] - 1) * 100, 1)
    return out


def main():
    nm, mm = load("backtest")
    ns, ms = load("satellite_bt")
    idx = nm.index.intersection(ns.index)
    a, b = nm.reindex(idx).ffill(), ns.reindex(idx).ffill()
    ra, rb = a.pct_change().fillna(0), b.pct_change().fillna(0)
    c60 = (1 + 0.6 * ra + 0.4 * rb).cumprod()
    c60 = c60 / c60.iloc[0]

    print("=== 分年度收益(%) ===")
    ya, yb, yc = yearly(a), yearly(b), yearly(c60)
    print(f"{'年':6s} {'主策略':>9s} {'60/40':>9s} {'副策略':>9s}")
    for y in sorted(ya):
        print(f"{y:6s} {ya[y]:>9.1f} {yc.get(y, float('nan')):>9.1f} "
              f"{yb.get(y, float('nan')):>9.1f}")

    # 滚动 252 日起点
    r = a.pct_change().fillna(0)
    roll = (1 + r).rolling(252).apply(np.prod, raw=True) - 1
    roll = roll.dropna() * 100
    print(f"\n主策略滚动 252 日: 最差 {roll.min():.1f}%  中位 {roll.median():.1f}%  "
          f"最好 {roll.max():.1f}%  亏钱起点 {(roll < 0).sum()}/{len(roll)}")

    # ---- 图1 ----
    fig, ax = plt.subplots(figsize=(12.6, 6.4))
    x = pd.to_datetime(idx, format="%Y%m%d")
    for s, lab, col, lw in ((a, f"100% 主策略  {mm['总收益率%']/100+1:,.0f}×",
                             "#c0392b", 2.1),
                            (c60, "60/40 组合  116×", "#2c7a4b", 2.1),
                            (b, f"100% 副策略  {ms['总收益率%']/100+1:,.2f}×",
                             "#2c6fa8", 1.6)):
        s = s / s.iloc[0]
        ax.plot(x, s, color=col, lw=lw, label=lab)
        dd = s / s.cummax() - 1
        i = int(np.argmin(dd.values))
        ax.scatter([x[i]], [s.iloc[i]], color=col, s=22, zorder=5)
        ax.annotate(f"{dd.iloc[i]*100:.1f}%", (x[i], s.iloc[i]),
                    textcoords="offset points", xytext=(6, -12),
                    fontsize=8.5, color=col)
    ax.set_yscale("log")
    ax.set_ylabel("净值(对数轴, 起点=1)")
    ax.set_title("第185轮：两条腿的复利与回撤（2020-08 ~ 2026-11，1500 个交易日）", fontsize=13)
    ax.grid(alpha=0.25, ls="--", which="both")
    ax.legend(fontsize=10, loc="upper left")
    ax.text(0.99, 0.02, "圆点 = 各自最大回撤位置", transform=ax.transAxes,
            ha="right", fontsize=8.5, color="#666")
    fig.tight_layout()
    p1 = WORK / "第185轮_组合净值.png"
    fig.savefig(p1, dpi=160)
    print(f"\n{p1}")

    # ---- 图2 ----
    d = json.loads((OUT / "combo.json").read_text(encoding="utf-8"))
    g = d["grid"]
    fig, ax = plt.subplots(figsize=(9.6, 6.0))
    xs = [abs(x["回撤%"]) for x in g]
    ys = [x["倍数"] for x in g]
    ws = [x["w"] for x in g]
    sc = ax.scatter(xs, ys, c=ws, cmap="RdYlGn", s=90, edgecolor="#333", zorder=4)
    for xi, yi, w in zip(xs, ys, ws):
        ax.annotate(f"{w}%", (xi, yi), textcoords="offset points", xytext=(7, 5), fontsize=8.5)
    ax.axhline(100, color="#c0392b", ls="--", lw=1.4)
    ax.text(21.6, 108, "百倍线", color="#c0392b", fontsize=9, ha="right")
    ax.set_yscale("log")
    ax.set_ylim(6.5, 900)
    ax.set_xlim(11.5, 23.5)
    ax.set_xlabel("最大回撤(绝对值 %)")
    ax.set_ylabel("期末倍数(对数轴)")
    ax.set_title("权重网格：横轴是回撤代价，纵轴是复利结果\n"
                 "红色虚线以上 = 百倍；最左侧 60% 档仍在线以上", fontsize=12)
    ax.grid(alpha=0.25, ls="--", which="both")
    fig.colorbar(sc, ax=ax, label="主策略权重 %")
    fig.tight_layout()
    p2 = WORK / "第185轮_权重网格.png"
    fig.savefig(p2, dpi=160)
    print(p2)


if __name__ == "__main__":
    main()
