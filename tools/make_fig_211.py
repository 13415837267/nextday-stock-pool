"""第211轮配图: 左=逼近250日新高的"倒U", 右=等权分散的年度复利(依赖牛市).

左图: 右侧池内按 dist_ph250 细分, 赢家率(ret10>20%)随"逼近新高"单调上升,
  但 ret10 均值在 -1~0 见顶、突破过头(>3%)后回落 —— 追高的甜点在"临界点"而非"已破位"。
右图: 核心池(逼近新高+温和量+低波+无涨停)等权分散的近似年化, 红=正 绿=负,
  说明它是顺风信号: 2014/2015/2019~2021 赚钱, 2011/2018/2022/2024 稳定为负。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

C1, C2, C3 = "#2F6F9F", "#C0392B", "#7F8C8D"


def main() -> None:
    df = pd.read_parquet(OUT / "panel_all_205.parquet",
                         columns=["d", "pct", "mom60", "dist_ph20", "dist_ph250",
                                  "vr", "vol20", "lu5", "ret10"])
    df = df.dropna(subset=["ret10"])
    df["d"] = df["d"].astype(str)

    right = df[(df["pct"] > 0) & (df["mom60"] > 0) & (df["dist_ph20"] > -5)].copy()
    nd = right["d"].nunique()
    right["k"] = pd.cut(right["dist_ph250"], [-999, -30, -15, -5, -3, -1, 0, 1, 3, 999],
                        labels=["<-30", "-30~-15", "-15~-5", "-5~-3", "-3~-1", "-1~0", "0~1", "1~3", ">3"])
    g = right.groupby("k", observed=True)
    labels = list(g.size().index)
    win = [g.ret10.apply(lambda x: (x > 20).mean() * 100)[k] for k in labels]
    mu = [g.ret10.mean()[k] for k in labels]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15.6, 6.0),
                                  gridspec_kw={"width_ratios": [1.12, 1]})

    # ---- 左图: 双轴(赢家率柱 + ret10均值线) ----
    xs = np.arange(len(labels))
    bars = ax.bar(xs, win, width=0.62, color="#9BB8D3", edgecolor="none", label="未来10日涨20%+的概率（左轴）")
    for x, w in zip(xs, win):
        ax.text(x, w / 2, f"{w:.1f}", ha="center", va="center", fontsize=9, color="#1F3A52")
    ax.set_ylabel("赢家率 ret10>20% (%)", fontsize=10)
    ax.set_ylim(0, 13)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8.6)

    axr = ax.twinx()
    axr.plot(xs, mu, color=C2, marker="o", lw=2, ms=5, label="未来10日平均收益（右轴）")
    for x, m in zip(xs, mu):
        axr.annotate(f"{m:+.1f}", (x, m), textcoords="offset points", xytext=(0, 9),
                     ha="center", fontsize=8.8, color=C2)
    axr.axhline(0, color=C3, lw=0.8, ls="--")
    axr.set_ylabel("ret10 平均收益 (%)", fontsize=10)
    axr.set_ylim(-1.5, 3.3)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axr.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9.2, framealpha=0.9)
    ax.set_xlabel("距 250 日高点 dist_ph250（右侧池：收阳 + 趋势向上 + 距 20 日高点 > −5%）", fontsize=10)
    ax.set_title("追高的甜点在“临界点”：越逼近新高概率越高，突破过头收益反而回落",
                 fontsize=11.6, fontweight="bold", pad=10)

    # ---- 右图: 等权分散年度复利 ----
    core = df[(df["pct"] > 0) & (df["mom60"] > 0) & (df["dist_ph20"] > -5)
              & df["dist_ph250"].between(-5, 0)
              & df["vr"].between(1.0, 2.0) & (df["vol20"] < 3.0) & (df["lu5"] == 0)].copy()
    core["yr"] = core["d"].str[:4]
    batch = core.groupby("d")["ret10"].mean()
    byyr = batch.groupby(batch.index.str[:4]).mean()
    daily = byyr / 100 / 10
    ann = (1 + daily) ** 250 - 1
    years = sorted(byyr.index)
    annv = [ann[y] * 100 for y in years]
    cols = [C2 if a >= 0 else "#2E8B57" for a in annv]
    ax2.bar(years, annv, color=cols, width=0.66)
    for y, a in zip(years, annv):
        ax2.text(y, a + (3 if a >= 0 else -5), f"{a:+.0f}", ha="center",
                 va="bottom" if a >= 0 else "top", fontsize=8.2, color="#333")
    ax2.axhline(0, color=C3, lw=0.9)
    ax2.set_xticks(years)
    ax2.set_xticklabels(years, rotation=45, fontsize=8.4)
    ax2.set_ylabel("近似年化收益 (%)", fontsize=10)
    ax2.set_title("等权分散也救不了：它是顺风信号，熊市年稳定为负",
                  fontsize=11.6, fontweight="bold", pad=10)
    ax2.text(0.02, 0.96, f"复利总倍数 ≈ 6.6×", transform=ax2.transAxes,
             fontsize=10, color=C1, va="top", fontweight="bold")

    fig.tight_layout()
    fig.savefig(OUT / "第211轮_右侧追高的真实画像.png", dpi=130, bbox_inches="tight")
    print("saved", OUT / "第211轮_右侧追高的真实画像.png")


if __name__ == "__main__":
    main()
