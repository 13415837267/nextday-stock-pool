"""第264轮配图: 单吊策略仓位风险预算 —— 年化 / 回撤 / 卡玛 随仓位的变化"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    d = json.load(open(ROOT / "output/exp_risk_264.json"))
    g = d["grid_rf0"]
    f = [x["f"] for x in g]
    cagr = [x["cagr_pct"] for x in g]
    dd = [x["maxdd_pct"] for x in g]
    calmar = [x["cagr_pct"] / abs(x["maxdd_pct"]) for x in g]

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.6))

    ax = axes[0]
    ax.plot(f, cagr, "-o", color=BLUE, lw=2, ms=5)
    ax.scatter([1.0], [cagr[-1]], color=RED, zorder=5, s=60)
    ax.annotate(f"满仓 {cagr[-1]:.0f}%", (1.0, cagr[-1]),
                textcoords="offset points", xytext=(-8, 8),
                color=RED, fontweight="bold", fontsize=10)
    ax.annotate(f"20% 仓 {cagr[1]:.0f}%", (0.2, cagr[1]),
                textcoords="offset points", xytext=(-40, -22),
                color=GREY, fontsize=9,
                arrowprops=dict(arrowstyle="-", color=GREY, lw=0.8))
    ax.set_title("年化收益 vs 仓位\n(降仓是几乎线性的砍收益)", fontsize=11.5)
    ax.set_xlabel("投入比例 f", fontsize=10)
    ax.set_ylabel("年化收益 (%)", fontsize=10)
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.plot(f, dd, "-o", color=GREEN, lw=2, ms=5)
    ax.scatter([1.0], [dd[-1]], color=RED, zorder=5, s=60)
    ax.annotate(f"满仓 {dd[-1]:.1f}%", (1.0, dd[-1]),
                textcoords="offset points", xytext=(-8, -16),
                color=RED, fontweight="bold", fontsize=10)
    ax.axhline(0, color=DARK, lw=1)
    ax.set_title("最大回撤 vs 仓位\n(线性缩放, 满仓 -27%)", fontsize=11.5)
    ax.set_xlabel("投入比例 f", fontsize=10)
    ax.set_ylabel("最大回撤 (%)", fontsize=10)
    ax.grid(alpha=0.25)

    ax = axes[2]
    ax.plot(f, calmar, "-o", color=DARK, lw=2, ms=5)
    ax.scatter([1.0], [calmar[-1]], color=RED, zorder=5, s=60)
    ax.annotate(f"满仓卡玛 {calmar[-1]:.2f}\n(风险调整后最优)", (1.0, calmar[-1]),
                textcoords="offset points", xytext=(-70, 10),
                color=RED, fontweight="bold", fontsize=10)
    ax.annotate(f"20% 仓卡玛 {calmar[1]:.2f}", (0.2, calmar[1]),
                textcoords="offset points", xytext=(-10, 18),
                color=GREY, fontsize=9,
                arrowprops=dict(arrowstyle="-", color=GREY, lw=0.8))
    ax.set_title("卡玛比率 vs 仓位\n(满仓 = 最优, 降仓反而变差)", fontsize=11.5)
    ax.set_xlabel("投入比例 f", fontsize=10)
    ax.set_ylabel("卡玛 (年化/回撤)", fontsize=10)
    ax.grid(alpha=0.25)

    fig.text(0.5, -0.04,
             "口径: output/backtest_long.json 满仓单吊净额日度曲线(3000 日, 2014-07 ~ 2026-11, 714 笔, 年化 138%)。"
             "f=投入策略的比例, 其余 1-f 留现金(0 收益)。Kelly 全/半/四分之一 = 4.99 / 2.49 / 1.25, 均 ≥100%。",
             ha="center", fontsize=8.6, color="#5d6d7e")

    fig.tight_layout()
    out = ROOT / "第264轮_仓位风险预算.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print("saved", out)


if __name__ == "__main__":
    main()
