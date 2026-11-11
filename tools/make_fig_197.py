# -*- coding: utf-8 -*-
"""第197轮配图: 主策略的钱从哪来(alpha/beta) + 组合权重跨窗口。"""
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
OUT = ROOT / "output"
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Sans CJK SC", "WenQuanYi Zen Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

C_MAIN, C_SUB, C_RED, C_GREY = "#1f4e79", "#c0504d", "#c00000", "#8c8c8c"


def fig1(ax1, ax2):
    d = pd.read_csv(OUT / "exp_factor_197_trades_1500.csv")
    f = json.loads((OUT / "exp_factor_197_1500.json").read_text(encoding="utf-8"))
    x, y = d["mkt"].values, d["ret"].values

    # (a) 散点 + 全样本/上行/下行回归
    ax1.axhline(0, color=C_GREY, lw=0.6, ls=":")
    ax1.axvline(0, color=C_GREY, lw=0.6, ls=":")
    ax1.scatter(x, y, s=14, color=C_MAIN, alpha=0.35, edgecolors="none", zorder=2)
    xs = np.linspace(x.min(), x.max(), 50)
    b1, b0 = f["mkt"]["beta"], f["mkt"]["alpha_daily_pct"] / 100
    ax1.plot(xs, (b0 + b1 * xs / 100) * 100, color=C_RED, lw=2.0, zorder=4,
             label=f"全样本 β={b1:.2f}　α={b0*100:+.2f}%/笔")
    for mask, col, lab, ls in ((x > 0, "#2e7d32", "上涨日 β=1.00", "--"),
                               (x < 0, "#ef6c00", "下跌日 β=0.21", "--")):
        X = np.column_stack([np.ones(mask.sum()), x[mask] / 100])
        bb, *_ = np.linalg.lstsq(X, y[mask] / 100, rcond=None)
        ax1.plot(xs, (bb[0] + bb[1] * xs / 100) * 100, color=col, lw=1.6, ls=ls, zorder=3, label=lab)
    ax1.set_xlabel("同期市场等权收益（T+1 开盘 → T+2 收盘，%）", fontsize=9)
    ax1.set_ylabel("这一笔的收益（%）", fontsize=9)
    ax1.set_title("(a) 近六年：上涨 β=1.00、下跌 β=0.21", fontsize=10.5, loc="left")
    ax1.legend(fontsize=8, loc="upper left", framealpha=0.9)
    ax1.grid(alpha=0.25, ls="--", lw=0.5)
    lo, hi = np.percentile(x, [0.5, 99.5])
    ax1.set_xlim(lo - 0.4, hi + 0.4)

    # (b) 两个窗口的 β 分解
    d3 = pd.read_csv(OUT / "exp_factor_197_trades_3000.csv")
    x3, y3 = d3["mkt"].values, d3["ret"].values

    def beta_of(xx, yy, mask):
        X = np.column_stack([np.ones(mask.sum()), xx[mask] / 100])
        bb, *_ = np.linalg.lstsq(X, yy[mask] / 100, rcond=None)
        return bb[1]

    cats = ["全样本", "上涨日", "下跌日"]
    v1 = [beta_of(x, y, np.ones(len(x), bool)), beta_of(x, y, x > 0), beta_of(x, y, x < 0)]
    v3 = [beta_of(x3, y3, np.ones(len(x3), bool)), beta_of(x3, y3, x3 > 0), beta_of(x3, y3, x3 < 0)]
    xi = np.arange(3)
    w = 0.34
    b1 = ax2.bar(xi - w / 2, v1, w, color=C_MAIN, label="近 1500 日")
    b3 = ax2.bar(xi + w / 2, v3, w, color=C_SUB, label="全史 3000 日")
    for bars in (b1, b3):
        for r in bars:
            ax2.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.03,
                     f"{r.get_height():.2f}", ha="center", fontsize=8.5, fontweight="bold",
                     color=bars is b1 and C_MAIN or C_SUB)
    ax2.axhline(1.0, color=C_GREY, lw=0.8, ls=":")
    ax2.text(2.42, 1.03, "β=1（跟着大盘走）", fontsize=7.5, color=C_GREY, ha="right")
    ax2.set_xticks(xi)
    ax2.set_xticklabels(cats, fontsize=9)
    ax2.set_ylabel("对市场等权收益的 β", fontsize=9)
    ax2.set_ylim(0, 1.18)
    ax2.set_title("(b) 下跌保护只在近六年成立：0.21 vs 0.72", fontsize=10.5, loc="left")
    ax2.legend(fontsize=8.5, loc="upper left")
    ax2.grid(axis="y", alpha=0.25, ls="--", lw=0.5)


def fig2(ax1, ax2):
    d1 = json.loads((OUT / "exp_combo_197_1500.json").read_text(encoding="utf-8"))
    d3 = json.loads((OUT / "exp_combo_197_3000.json").read_text(encoding="utf-8"))
    ws, c1, c3, a1, a3 = [], [], [], [], []
    for k, v in d1["grid"].items():
        ws.append(int(k.split("/")[0]) / 100)
        c1.append(v["卡玛比率"])
        a1.append(v["年化收益率%"])
    for k, v in d3["grid"].items():
        c3.append(v["卡玛比率"])
        a3.append(v["年化收益率%"])
    order = np.argsort(ws)
    ws = np.array(ws)[order]
    c1, c3 = np.array(c1)[order], np.array(c3)[order]
    a1, a3 = np.array(a1)[order], np.array(a3)[order]

    ax1.plot(ws * 100, c1, "o-", color=C_MAIN, lw=2.0, ms=5, label="近 1500 日")
    ax1.plot(ws * 100, c3, "s-", color=C_SUB, lw=2.0, ms=5, label="全史 3000 日")
    ax1.axvline(60, color=C_RED, lw=1.0, ls="--", alpha=0.7)
    ax1.annotate("60/40：近六年卡玛 7.56\n全史只有 3.54", xy=(60, 7.56), xytext=(16, 6.95),
                 fontsize=8.5, color=C_RED, ha="left",
                 arrowprops=dict(arrowstyle="->", color=C_RED, lw=1.0,
                                 connectionstyle="arc3,rad=0.15"))
    ax1.annotate("全史最优点\n仍是 100% 主策略", xy=(99, 4.04), xytext=(74, 5.35),
                 fontsize=8.5, color=C_SUB, ha="center",
                 arrowprops=dict(arrowstyle="->", color=C_SUB, lw=1.0))
    ax1.set_xlabel("账户里分给主策略的资金比例（%），其余给副策略", fontsize=9)
    ax1.set_ylabel("卡玛比率", fontsize=9)
    ax1.set_title("(a) 副策略只在近六年提升卡玛，全史是拖累", fontsize=10.5, loc="left")
    ax1.legend(fontsize=8.5, loc="lower left")
    ax1.grid(alpha=0.25, ls="--", lw=0.5)
    ax1.set_xticks([0, 20, 40, 60, 80, 100])
    ax1.set_ylim(0, 8.6)

    # (b) 年度收益
    def load(p):
        dd = json.loads((OUT / p).read_text(encoding="utf-8"))
        return pd.Series({str(z["d"]).replace("-", "")[:8]: float(z["nav"]) for z in dd["equity"]})
    m, s = load("backtest.json"), load("satellite_bt.json")
    df = pd.DataFrame({"m": m, "s": s}).dropna()
    yrs = sorted(set(df.index.str[:4]))
    am = [(df.loc[df.index.str[:4] == y, "m"].iloc[-1] / df.loc[df.index.str[:4] == y, "m"].iloc[0] - 1) * 100
          for y in yrs]
    as_ = [(df.loc[df.index.str[:4] == y, "s"].iloc[-1] / df.loc[df.index.str[:4] == y, "s"].iloc[0] - 1) * 100
           for y in yrs]
    xi = np.arange(len(yrs))
    w = 0.38
    ax2.bar(xi - w / 2, am, w, color=C_MAIN, label="主策略")
    ax2.bar(xi + w / 2, as_, w, color=C_SUB, label="副策略")
    for i, v in enumerate(am):
        ax2.text(i - w / 2, v + (6 if v >= 0 else -22), f"{v:.0f}", ha="center", fontsize=7.5,
                 color=C_MAIN, fontweight="bold")
    ax2.axhline(0, color="#333", lw=0.8)
    ax2.set_xticks(xi)
    ax2.set_xticklabels(yrs, fontsize=8)
    ax2.set_ylabel("当年收益（%）", fontsize=9)
    ax2.set_title("(b) 逐年：没有哪一年靠单笔暴利撑住，2020 是残缺年", fontsize=10.5, loc="left")
    ax2.legend(fontsize=8.5, loc="upper right")
    ax2.grid(axis="y", alpha=0.25, ls="--", lw=0.5)
    ax2.set_ylim(-40, 290)


def main():
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.4))
    fig1(axes[0][0], axes[0][1])
    fig2(axes[1][0], axes[1][1])
    fig.suptitle("第 197 轮：近六年是「凸」，全史是一半市场 β；副策略也只在近六年加分",
                 fontsize=13, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    p = ROOT.parent / "第197轮_主策略的钱从哪来.png"
    fig.savefig(p, dpi=145)
    print("写出", p)


if __name__ == "__main__":
    main()
