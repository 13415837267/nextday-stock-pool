"""第二百四十轮配图: 目标函数换成"空仓日"之后 —— 挖到的条件、以及北交所这个坑"""
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


def main() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17.6, 6.3),
                             gridspec_kw={"width_ratios": [1.28, 1.0, 1.0]})
    a, b, c = axes

    # ---------- ① 卡玛增量: 含北交所 vs 仅主板 ----------
    rows = [
        ("W7 破位补位\n满仓·含北交所", 2.44, "#c0392b"),
        ("W7 破位补位\n满仓·仅主板", -1.32, "#7f8c8d"),
        ("W7 破位补位\n30%仓·仅主板", 0.74, "#27ae60"),
        ("B1 暴跌抄底\n满仓·含北交所", 1.29, "#d98880"),
        ("B1 暴跌抄底\n满仓·仅主板", 0.02, "#bdc3c7"),
    ]
    ys = np.arange(len(rows))[::-1]
    for y, (lab, v, col) in zip(ys, rows):
        a.barh(y, v, color=col, height=0.58)
        a.text(v + (0.09 if v >= 0 else -0.09), y, f"{v:+.2f}", va="center",
               ha="left" if v >= 0 else "right", fontsize=10, fontweight="bold", color="#333")
    a.axvline(0, color="#333", lw=1.2)
    a.set_yticks(ys)
    a.set_yticklabels([r[0] for r in rows], fontsize=9)
    a.set_xlim(-2.0, 3.1)
    a.set_xlabel("卡玛比率相对基准的增量（基准：2021+ 为 6.63，2019+ 为 4.95）", fontsize=9.5)
    a.set_title("① 两个“发现”的超额，在剔除北交所后基本消失\n"
                "3 万本金开不了北交所（50 万门槛），而“市值最小”选股必然选到它",
                fontsize=11, pad=8)
    a.set_ylim(-0.7, len(rows) - 0.4)
    a.grid(axis="x", ls=":", color="#ccc", alpha=0.6)
    a.set_axisbelow(True)

    # ---------- ② 仓位曲线 ----------
    w = np.array([0.0, 0.3, 0.5, 0.7, 1.0])
    ann = np.array([156.31, 173.87, 185.71, 197.60, 215.48])
    dd = np.array([-23.59, -23.59, -26.62, -32.42, -40.57])
    cml = np.array([6.63, 7.37, 6.98, 6.09, 5.31])
    b.plot(w * 100, ann, "o-", color="#c0392b", lw=2, ms=6, label="年化收益 %")
    for x, y in zip(w * 100, ann):
        b.text(x, y + 6, f"{y:.0f}", ha="center", fontsize=9, color="#c0392b")
    b.set_xlabel("补位日投入的仓位 %（剩余资金空着）", fontsize=9.5)
    b.set_ylabel("年化收益 %", fontsize=9.5, color="#c0392b")
    b.set_ylim(138, 228)
    b2 = b.twinx()
    b2.plot(w * 100, -dd, "s--", color="#2c3e50", lw=1.8, ms=5, label="最大回撤 %（右轴）")
    for x, y, k in zip(w * 100, -dd, cml):
        ha = "right" if x > 90 else "center"
        b2.text(x, y + 2.2, f"卡玛 {k:.2f}", fontsize=8.3, color="#2980b9", ha=ha,
                path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])
    b2.set_ylabel("回撤 %（绝对值）", fontsize=9.5, color="#2c3e50")
    b2.set_ylim(16, 52)
    b.axvline(30, color="#27ae60", ls=":", lw=1.2, alpha=0.8)
    b.text(52, 152, "30% 仓位：回撤不变\n卡玛 7.37（基准 6.63）", fontsize=8.7, color="#27ae60",
           va="bottom", ha="left", linespacing=1.7,
           path_effects=[pe.withStroke(linewidth=2.8, foreground="white")])
    b.set_title("② 这条规则不是“加满就好”\n满仓时回撤从 −24% 炸到 −41%，30~50% 才是对的", fontsize=11, pad=8)
    b.grid(ls=":", color="#ccc", alpha=0.55)
    b.set_axisbelow(True)
    lines = b.get_lines() + b2.get_lines()
    b2.legend(lines, [l.get_label() for l in lines], fontsize=7.9, loc="upper left",
              framealpha=0.92, bbox_to_anchor=(0.0, 1.0))

    # ---------- ③ 分年 ----------
    yrs = ["2021", "2022", "2023", "2024", "2025", "2026"]
    s = np.array([10.30, 54.57, 8.29, 37.31, 10.24, 10.05])
    n = np.array([5, 38, 8, 31, 3, 8])
    xs = np.arange(len(yrs))
    c.bar(xs, s, color="#c0392b", width=0.6)
    for x, (v, k) in enumerate(zip(s, n)):
        c.text(x, v + 1.5, f"+{v:.0f}", ha="center", fontsize=9.5, fontweight="bold", color="#333")
        c.text(x, 1.2, f"{k}次", ha="center", fontsize=8.2, color="white")
    c.set_xticks(xs)
    c.set_xticklabels(yrs, fontsize=9.5)
    c.set_ylim(0, 64)
    c.set_ylabel("该年补位累计贡献（百分点）", fontsize=9.5)
    c.set_title("③ 主板 3 只·满仓口径：6 年全正\n但 2022/2024 两年贡献 88%，熊市才是它的主场", fontsize=11, pad=8)
    c.grid(axis="y", ls=":", color="#ccc", alpha=0.6)
    c.set_axisbelow(True)

    fig.text(0.5, 0.015,
             "规则：全市场等权指数距 20 日线 ≤ −3.51%　且　个股收盘距前 60 日最高 ≤ −24.46%　→　按流通市值从小到大买 3 只，"
             "T+1 开盘买 T+2 收盘卖｜93 次·单笔 +1.41%·胜率 64.5%",
             ha="center", fontsize=8.6, color="#666")
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(ROOT / "第240轮_破位日补位.png", dpi=150, facecolor="white")
    print("saved")


if __name__ == "__main__":
    main()
