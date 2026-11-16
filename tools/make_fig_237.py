"""第二百三十七轮配图: 挖出来的条件, 拆成 beta 和 alpha"""
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
OUT = ROOT / "output"

LABEL = {
    "C1 bias10 & bias20": "C1  反弹中继（10日强/20日弱）",
    "C2 bias10 & ret10": "C2  急跌后首日反弹",
    "C3 pct & rel_pct": "C3  大盘大跌日抗跌",
    "C4 rel_pct & dd_m_pct": "C4  抗跌+单日跌穿20日回撤",
    "C5 rel_pct & vol_x_pct": "C5  抗跌+高波动中下跌",
    "C6 bias20 & ret10": "C6  20日线附近+10日仍涨",
}


def main() -> None:
    d = pd.read_csv(OUT / "w237c_summary.csv")
    d["lab"] = d["cond"].map(lambda s: LABEL.get(s, s))
    d = d.sort_values("y", ascending=True)

    fig, (a, b) = plt.subplots(1, 2, figsize=(13.8, 6.6),
                               gridspec_kw={"width_ratios": [1.45, 1]})

    ys = np.arange(len(d))
    alpha = d["y_ex"].values
    beta = d["y"].values - alpha
    a.barh(ys, beta, color="#c8ccd0", height=0.6, label="市场 beta（同一天随便买也有的）")
    a.barh(ys, alpha, left=beta, color="#c0392b", height=0.6, label="选股 alpha（扣掉当日市场均值）")
    a.axvline(1.178, color="#2c3e50", ls="--", lw=1.6)
    a.text(2.50, len(d) - 1.5, "现行\n1.18%", fontsize=8.5, color="#2c3e50", va="center", ha="left",
           path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])
    a.axvline(0.235, color="#7f8c8d", ls=":", lw=1.3)
    a.text(0.26, -0.62, "全市场 0.235%", fontsize=8, color="#7f8c8d",
           path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])
    for y, (t, al) in enumerate(zip(d["y"], alpha)):
        a.text(t + 0.04, y, f"{t:.2f}", va="center", fontsize=9, color="#333")
        a.text(al / 2 + (t - al), y, f"{al:.2f}", va="center", ha="center", fontsize=8.2,
               color="white", fontweight="bold")
    a.set_yticks(ys)
    a.set_yticklabels(d["lab"], fontsize=9)
    a.set_xlim(0, 2.95)
    a.set_ylim(-0.8, len(d) - 0.25)
    a.set_xlabel("单笔收益 %（T+1 开盘买 → T+2 收盘卖）", fontsize=10)
    a.set_title("① 6 个条件看着 1.3~2.4%，拆开只剩 0.4~0.7% 是选股", fontsize=11.5, pad=10)
    a.legend(fontsize=8.5, loc="lower right", framealpha=0.92)
    a.grid(axis="x", ls=":", color="#ccc", alpha=0.6)
    a.set_axisbelow(True)

    # 右: 覆盖天数 vs alpha
    b.scatter(d["days"], d["y_ex"], s=d["n"] / 260, color="#c0392b", alpha=0.75,
              edgecolor="white", linewidth=1.2, zorder=3)
    b.axhline(1.178, color="#2c3e50", ls="--", lw=1.4)
    b.axvline(559, color="#2c3e50", ls=":", lw=1.2)
    b.text(600, 1.22, "现行策略\n559 天 / 1.18%", fontsize=8.5, color="#2c3e50",
           path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])
    for _, r in d.iterrows():
        tag = r["cond"].split()[0]
        off = {"C3": (-16, 10), "C4": (4, 12), "C5": (16, -14)}.get(tag, (0, 11))
        b.annotate(tag, (r["days"], r["y_ex"]), textcoords="offset points",
                   xytext=off, ha="center", fontsize=9, fontweight="bold", color="#7b241c")
    b.set_xlabel("条件覆盖的交易日数（共 4015 天）", fontsize=10)
    b.set_ylabel("选股 alpha %", fontsize=10)
    b.set_title("② 新条件 alpha 更低，但覆盖天数多 2~3 倍", fontsize=11.5, pad=10)
    b.set_xlim(0, 1950)
    b.set_ylim(0.15, 1.55)
    b.grid(ls=":", color="#ccc", alpha=0.6)
    b.set_axisbelow(True)

    fig.text(0.5, 0.012,
             "口径：817 万可执行样本（次日低开可成交、非停牌非 ST、上市满 250 日）；alpha = 单笔收益 − 当日全市场均值；"
             "圆点大小 = 样本量",
             ha="center", fontsize=8.2, color="#666")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(ROOT / "第237轮_挖出来的条件值多少.png", dpi=150)
    print("ok")


if __name__ == "__main__":
    main()
