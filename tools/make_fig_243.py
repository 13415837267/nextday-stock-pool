"""第 243 轮配图: 选股窗口从 60 日拉到 250 日 —— 卡玛 3.38 -> 5.39 -> 仓位 70% 时 5.81

① 窗口曲线: 60~300 日全在 4.2~5.4, 是平台不是孤峰
② 仓位曲线: 20~85% 全在 4.3~5.8, 取 70% 时年化 137% 且回撤 -23.6%（比基准还小）
③ 分年: 11 年 11 正
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

WIN = [(60, 5.05, -33.12), (90, 5.44, -31.76), (120, 5.08, -33.38), (150, 4.65, -37.15),
       (180, 4.23, -40.01), (200, 4.68, -33.38), (250, 5.39, -29.65), (300, 5.09, -29.65),
       (360, 4.07, -35.13)]
POS = [(20, 101.52, 4.30), (30, 108.44, 4.60), (40, 115.47, 4.89), (50, 122.60, 5.20),
       (70, 137.16, 5.81), (85, 148.32, 5.63), (100, 159.67, 5.39)]
YR = [(2016, 1.82, 4), (2017, 1.14, 25), (2018, 1.69, 44), (2019, 1.36, 16),
      (2020, 1.27, 12), (2021, 1.86, 5), (2022, 0.72, 36), (2023, 1.50, 8),
      (2024, 3.89, 31), (2025, 4.76, 3), (2026, 3.50, 8)]
COOL = [(0, 5.39, -29.65, 29, 75), (5, 4.07, -31.63, 13, 122), (10, 3.99, -32.74, 8, 149),
        (20, 4.52, -28.65, 5, 176), (40, 4.42, -28.65, 5, 186), (60, 4.42, -28.65, 5, 186)]
BASE = (88.0, -24.24, 3.63)


def main():
    fig = plt.figure(figsize=(15.6, 5.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.0, 1.15], wspace=0.30,
                          left=0.055, right=0.955, top=0.845, bottom=0.205)

    # ---------- ① 窗口 ----------
    a = fig.add_subplot(gs[0, 0])
    xs = np.arange(len(WIN))
    cm = [w[1] for w in WIN]
    dd = [-w[2] for w in WIN]
    cols = ["#c0392b" if w[0] == 250 else "#95a5a6" for w in WIN]
    a.bar(xs, cm, color=cols, width=0.66, label="卡玛比率（左轴）")
    for x, c in zip(xs, cm):
        a.text(x, c + 0.08, f"{c:.2f}", ha="center", fontsize=8.4, color="#333",
               path_effects=[pe.withStroke(linewidth=2.6, foreground="white")])
    a.axhline(BASE[2], color="#2c3e50", ls="--", lw=1.6, label=f"现行 60日 {BASE[2]:.2f}")
    a.set_ylim(3.2, 6.3)
    a.set_ylabel("卡玛比率", fontsize=9.5)
    a.set_xlabel("选股窗口（距前 N 日最高价最远）", fontsize=9.5)
    a.set_xticks(xs)
    a.set_xticklabels([str(w[0]) for w in WIN], fontsize=8.4)
    a.set_title("① 窗口 60→300 日：卡玛 4.2~5.4 是平台\n"
                "红柱=取 250 日（回撤最低、末段最强）\n"
                "9 个窗口里 8 个三段全正 → 不是孤峰",
                fontsize=10.5, pad=6)
    a.grid(axis="y", ls=":", color="#ccc", alpha=0.6)
    a.set_axisbelow(True)
    a.legend(fontsize=8.2, loc="upper left", framealpha=0.92)
    # 右侧阴影: 平台范围
    a.axvspan(-0.5, 7.5, color="#f1c40f", alpha=0.10, zorder=0)

    b2 = a.twinx()
    b2.plot(xs, dd, "s--", color="#2980b9", lw=1.6, ms=4.5, label="最大回撤 %（右轴）")
    b2.set_ylim(24, 44)
    b2.set_ylabel("最大回撤 %（绝对值）", fontsize=9.5, color="#2980b9")
    b2.tick_params(axis="y", colors="#2980b9", labelsize=8.5)
    b2.legend(fontsize=8.2, loc="lower right", framealpha=0.92)

    # ---------- ② 仓位 ----------
    b = fig.add_subplot(gs[0, 1])
    px = np.arange(len(POS))
    ann = [p[1] for p in POS]
    kk = [p[2] for p in POS]
    a2 = [p[0] for p in POS]
    b.bar(px, ann, color=["#c0392b" if x == 70 else "#95a5a6" for x in a2], width=0.66)
    b.axhline(BASE[0], color="#2c3e50", ls="--", lw=1.6)
    b.text(len(POS) - 0.55, BASE[0] + 4, f"基准 {BASE[0]:.0f}%", fontsize=8.2, color="#2c3e50",
           ha="right", path_effects=[pe.withStroke(linewidth=2.4, foreground="white")])
    for x, v in zip(px, ann):
        b.text(x, v + 6, f"{v:.0f}%", ha="center", fontsize=8.5, color="#333",
               bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.88))
    b.set_ylim(58, 192)
    b.set_ylabel("年化 %（左轴）", fontsize=9.5)
    b.set_xlabel("补位仓位（单吊一只，占资金比例）", fontsize=9.5)
    b.set_xticks(px)
    b.set_xticklabels([f"{p[0]}%" for p in POS], fontsize=8.4)
    b.set_title("② 70% 仓位：年化 137%，回撤 −23.6%\n（比基准 −24.2% 还小）", fontsize=10.8, pad=8)
    b.grid(axis="y", ls=":", color="#ccc", alpha=0.6)
    b.set_axisbelow(True)

    b3 = b.twinx()
    b3.plot(px, kk, "o--", color="#2980b9", lw=1.9, ms=5.5, label="卡玛（右轴）")
    for x, v in zip(px, kk):
        dy = -0.38 if x < 2 else 0.17
        b3.text(x, v + dy, f"{v:.2f}", ha="center", fontsize=8.3, color="#2980b9",
                bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.88))
    b3.axhline(BASE[2], color="#2980b9", ls=":", lw=1.3, alpha=0.8)
    b3.set_ylim(3.9, 6.5)
    b3.set_ylabel("卡玛比率（右轴）", fontsize=9.5, color="#2980b9")
    b3.tick_params(axis="y", colors="#2980b9", labelsize=8.5)
    b3.legend(fontsize=8.2, loc="lower right", framealpha=0.92)

    # ---------- ③ 冷却期 ----------
    c = fig.add_subplot(gs[0, 2])
    cx = np.arange(len(COOL))
    kc = [x[1] for x in COOL]
    cd_ = [x[2] for x in COOL]
    tp = [x[3] for x in COOL]
    c.bar(cx, kc, color=["#95a5a6" if x[0] != 20 else "#c0392b" for x in COOL], width=0.64)
    for x, v in zip(cx, kc):
        c.text(x, v + 0.07, f"{v:.2f}", ha="center", fontsize=8.4, color="#333",
               bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="none", alpha=0.85))
    c.axhline(BASE[2], color="#2c3e50", ls="--", lw=1.6, label=f"基准 {BASE[2]:.2f}")
    c.set_ylim(3.15, 6.7)
    c.set_ylabel("卡玛比率（左轴）", fontsize=9.5)
    c.set_xlabel("同一只票的冷却期（交易日）", fontsize=9.5)
    c.set_xticks(cx)
    c.set_xticklabels([("无" if x[0] == 0 else str(x[0])) for x in COOL], fontsize=8.6)
    c.set_title("③ 加 20 日冷却：卡玛 5.39→4.52\n"
                "\"连买同一只 12 次\"变成 1 次；它用 176 只票、回撤还小 1pp",
                fontsize=10.3, pad=7)
    c.grid(axis="y", ls=":", color="#ccc", alpha=0.6)
    c.set_axisbelow(True)
    c.legend(fontsize=8.2, loc="upper right", framealpha=0.92)

    c2 = c.twinx()
    c2.plot(cx, tp, "o-", color="#e67e22", lw=1.9, ms=5.5, label="前 5 只票占全部次数 %")
    for x, v in zip(cx, tp):
        if x != 0:
            continue
        c2.text(x, v + 2.6, f"{v:.0f}%", ha="center", fontsize=8.5, color="#b35c00",
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.9))
    c2.set_ylim(-1, 38)
    c2.set_ylabel("前 5 票占比 %", fontsize=9.2, color="#b35c00")
    c2.tick_params(axis="y", colors="#b35c00", labelsize=8.5)
    c2.legend(fontsize=7.9, loc="upper center", framealpha=0.92)


    fig.text(0.5, 0.062,
             "口径：2016-08~2026-11，2466 个交易日。主策略=每日一只、T+1 开盘买→T+2 收盘卖"
             "（558 笔，年化 88.0%/回撤 −24.2%/卡玛 3.63）。补位只在主策略空仓且资金空闲的 1159 天触发，共 192 次。\n"
             "选股=「距前 250 日最高价（不含当日）最远」，单笔已扣 0.10% 往返成本；"
             "① 只换排序窗口，② ③ 固定 250 日；③ 为满仓口径，实际方案取 70% 仓 + 20 日冷却。",
             ha="center", fontsize=8.3, color="#555", linespacing=1.7)

    p = ROOT / "第243轮_把选股窗口拉到250日.png"
    fig.savefig(p, dpi=170, facecolor="white")
    print(f"saved {p}")


if __name__ == "__main__":
    main()
