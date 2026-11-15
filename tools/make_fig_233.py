"""第233轮 · 配图：2500 天的漏斗、缺口门槛的倒 U、23 个状态变量的全噪声"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP"]
plt.rcParams["axes.unicode_minus"] = False
ROOT = Path(__file__).resolve().parents[1]


def fig_funnel() -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.6),
                                 gridspec_kw={"width_ratios": [1.05, 1]})
    layers = [("全部交易日", 2500, "#34495e", "100%"),
              ("风控放行的日子", 1339, "#2980b9", "53.6%（砍掉 45%）"),
              ("真的买了票的日子", 559, "#27ae60", "22.4%（其中 36 笔是前一日信号、次日风控已关仍执行）")]
    for i, (name, v, c, note) in enumerate(layers):
        y = len(layers) - 1 - i
        a1.barh([y], [v], color=c, height=0.62)
        a1.text(v / 2, y, f"{v} 日", ha="center", va="center", fontsize=13,
                color="white", weight="bold")
        a1.text(v + 45, y, f"{name}", va="center", fontsize=11, color="#2c3e50")
        a1.text(v + 45, y - 0.26, note, va="center", fontsize=9, color="#7f8c8d")
    a1.set_xlim(0, 3750)
    a1.set_ylim(-0.6, 2.6)
    a1.set_yticks([])
    a1.set_xticks([0, 500, 1000, 1500, 2000, 2500])
    a1.set_xlabel("交易日数")
    a1.set_title("2500 天里，只有 559 天真的买了票", fontsize=12.5)
    a1.spines[["top", "right", "left"]].set_visible(False)
    a1.grid(axis="x", alpha=0.22)

    # 右：skip 原因
    d = pd.DataFrame({"原因": ["缺口不进可买区间", "评分低于 1.0 门槛", "停牌 / 一字板"],
                      "次数": [1319, 594, 6],
                      "颜色": ["#2980b9", "#8e44ad", "#95a5a6"]})
    a2.barh(d["原因"], d["次数"], color=d["颜色"], height=0.5)
    for i, v in enumerate(d["次数"]):
        a2.text(v + 25, i, f"{v} 次", va="center", fontsize=10)
    a2.set_xlim(0, 1600)
    a2.set_title("被放弃的到底是什么原因", fontsize=12.5)
    a2.set_xlabel("放弃次数（同一信号日可能多条）")
    a2.invert_yaxis()
    a2.grid(axis="x", alpha=0.22)

    fig.text(0.5, 0.02, "持仓 2 天、放行 1339 天 ⇒ 笔数理论上限 ≈ 670；实际 559 已用掉 83%。"
             "剩下 17% 只能靠买更差的票来换，实测 −21pp。",
             ha="center", fontsize=10, color="#444")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(ROOT / "第233轮_2500天的漏斗.png", dpi=150)
    print("图1 ok")


def fig_gap() -> None:
    d = pd.read_csv(ROOT / "output" / "w233d_gap.csv")
    m = {"gap<=-1.0(收紧)": -1.0, "gap<=-0.5(收紧)": -0.5, "基准(现行)": 0.0,
         "gap<=+0.5": 0.5, "gap<=+1.0": 1.0, "gap<=+2.0": 2.0,
         "gap<=+3.0": 3.0, "gap<=+5.0": 5.0}
    d = d[d["方案"].isin(m)].copy()
    d["x"] = d["方案"].map(m)
    d = d.sort_values("x")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.4))
    a1.plot(d["x"], d["年化"], "-o", color="#c0392b", lw=2.2, ms=7)
    a1.set_ylabel("年化 %")
    a1.set_xlabel("买入端缺口上限（%；0 = 必须低开，正数 = 允许高开）")
    a1.set_title("年化：现行值就是峰", fontsize=12)
    a1.grid(alpha=0.25)
    for x, y in zip(d["x"], d["年化"]):
        a1.annotate(f"{y:.1f}", (x, y), xytext=(0, 12), textcoords="offset points",
                    ha="center", fontsize=9, color="#c0392b")
    i = int(np.argmax(d["年化"].values))
    a1.plot([d["x"].values[i]], [d["年化"].values[i]], "*", ms=24, color="#f1c40f",
            markeredgecolor="#c0392b", markeredgewidth=1.2, zorder=5)
    a1.set_ylim(10, 100)

    d2 = d[d["x"] >= -0.5].copy()      # −1% 档离群（单笔 1.86%），单独在图注说明
    a2b = a2.twinx()
    a2.bar(d2["x"], d2["笔数"], width=0.34, color="#2980b9", alpha=0.85, label="笔数（左轴）")
    a2.set_ylabel("笔数", color="#2980b9")
    a2.tick_params(axis="y", colors="#2980b9")
    a2b.plot(d2["x"], d2["单笔"], "-^", color="#27ae60", lw=1.8, ms=7, label="单笔 %（右轴）")
    a2b.set_ylabel("单笔收益 %", color="#27ae60")
    a2b.tick_params(axis="y", colors="#27ae60")
    a2.set_xlabel("买入端缺口上限（%；0 = 必须低开）")
    a2.set_title("笔数在 0 处最高；放宽不增笔数、只降单笔", fontsize=12)
    a2.grid(alpha=0.2)
    for x, v in zip(d2["x"], d2["笔数"]):
        a2.annotate(f"{v}", (x, v / 2), ha="center", va="center", fontsize=9.5,
                    color="white", weight="bold")
    h1, l1 = a2.get_legend_handles_labels()
    h2, l2 = a2b.get_legend_handles_labels()
    a2.legend(h1 + h2, l1 + l2, loc="upper center", fontsize=9, framealpha=0.9)
    a2.set_ylim(0, 760)

    fig.text(0.5, 0.02, "收紧到 −1%（未画入右图）：单笔升到 1.856%（+48%）但笔数只剩 139（−75%），年化 23.7%。"
             "放宽到 +2%：笔数反而变少（持仓互斥的挤占），单笔掉 0.15pp。",
             ha="center", fontsize=9.5, color="#444")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(ROOT / "第233轮_缺口门槛倒U.png", dpi=150)
    print("图2 ok")


def fig_noise() -> None:
    a = pd.read_csv(ROOT / "output" / "w233_probe.csv")
    b = pd.read_csv(ROOT / "output" / "w233_senti.csv")
    a["类"] = "市场状态"
    b["类"] = "情绪指标"
    d = pd.concat([a[["变量", "pearson", "类"]], b[["变量", "pearson", "类"]]])
    d = d.dropna().sort_values("pearson", ascending=False).reset_index(drop=True)
    fig, a1 = plt.subplots(figsize=(9.6, 7.2))
    cols = {"市场状态": "#2980b9", "情绪指标": "#8e44ad"}
    ys = np.arange(len(d))
    a1.barh(ys, d["pearson"], color=[cols[c] for c in d["类"]], height=0.62)
    a1.set_yticks(ys)
    a1.set_yticklabels(d["变量"], fontsize=9.5)
    a1.invert_yaxis()
    a1.axvline(0, color="#333", lw=1)
    for x in (-0.1, 0.1):
        a1.axvline(x, color="#e74c3c", ls="--", lw=1, alpha=0.55)
    a1.set_xlim(-0.17, 0.17)
    a1.set_xlabel("与单笔收益的相关系数 r")
    for y, r in zip(ys, d["pearson"]):
        a1.text(r + (0.006 if r >= 0 else -0.006), y, f"{r:+.3f}", va="center",
                ha="left" if r >= 0 else "right", fontsize=8.5, color="#444")
    a1.set_title("23 个「环境变量」全部落在 ±0.12 以内", fontsize=13)
    a1.text(0.102, -0.8, "|r| = 0.1", fontsize=9, color="#e74c3c")
    a1.grid(axis="x", alpha=0.2)
    fig.text(0.5, 0.025, "更关键的是分年：这批变量的五分位极差 0.7~2.2pp，来自 559 笔切成 5 份（每份 112 笔）的抽样噪声；"
             "10 个年度里没有一个能稳定同向（最好 6/10）。", ha="center", fontsize=9.5, color="#444")
    fig.tight_layout(rect=[0, 0.045, 1, 1])
    fig.savefig(ROOT / "第233轮_23个环境变量全噪声.png", dpi=150)
    print("图3 ok")


if __name__ == "__main__":
    only = sys.argv[1] if len(sys.argv) > 1 else ""
    if only in ("", "1"):
        fig_funnel()
    if only in ("", "2"):
        fig_gap()
    if only in ("", "3"):
        fig_noise()
