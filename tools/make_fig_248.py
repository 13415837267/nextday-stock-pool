"""第248轮 图: 让位 / 逐年 / 净值 / 重叠日散点"""

from __future__ import annotations

import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
COST = 0.30
RED, DARK, GREY, BLUE, ORANGE = "#c0392b", "#2c3e50", "#7f8c8d", "#2980b9", "#d68910"


def metrics(r: pd.Series) -> tuple[float, float, float]:
    nav = (1 + r / 100).cumprod()
    ann = nav.iloc[-1] ** (244 / len(r)) - 1
    dd = (nav / nav.cummax() - 1).min()
    return round(ann * 100, 2), round(dd * 100, 2), round(ann * 100 / abs(dd * 100), 2)


def main() -> None:
    nav = pd.read_csv(OUT / "w248_nav_cmp.csv", index_col=0)
    nav.index = nav.index.astype(int)
    tr = pd.read_csv(OUT / "w232_trades.csv")
    cal = sorted(int(f.split("/")[-1].replace(".csv.gz", ""))
                 for f in glob.glob(str(ROOT / "data" / "panel" / "*.csv.gz")))
    cal = [d for d in cal if d >= tr.signal.min()]
    base = pd.Series(0.0, index=cal)
    for dt, g in tr.groupby("signal"):
        base[int(dt)] = g.ret.mean()

    f2 = pd.read_csv(OUT / "w248_final_S2.csv")
    ov = pd.read_csv(OUT / "w248_overlap_detail.csv")

    fig, ax = plt.subplots(2, 2, figsize=(15.4, 10.4))

    # ---- 1 逐年 ----
    a1 = ax[0, 0]
    y0 = nav.S0.groupby(nav.index // 10000).apply(lambda x: (1 + x / 100).prod() - 1) * 100
    y2 = nav.S2.groupby(nav.index // 10000).apply(lambda x: (1 + x / 100).prod() - 1) * 100
    ks = list(y0.index)
    dif = (y2 - y0).values
    ypos = np.arange(len(ks))[::-1]
    cols = [RED if d_ > 0 else BLUE for d_ in dif]
    a1.barh(ypos, dif, 0.62, color=cols, alpha=0.9)
    a1.axvline(0, color=DARK, lw=1)
    a1.set_yticks(ypos)
    a1.set_yticklabels([str(k) for k in ks], fontsize=9)
    a1.set_xlabel("248 − 247 的年度收益差（百分点）")
    a1.set_title("逐年差：5 年改善 / 5 年变差（2016 不变），钱从 2024 挪到 2018·2020",
                 fontsize=12, color=DARK)
    a1.grid(axis="x", alpha=0.25)
    for yp, d_, k in zip(ypos, dif, ks):
        a1.text(d_ + (1.6 if d_ > 0 else -1.6), yp, f"{d_:+.0f}  ({y0[k]:.0f}→{y2[k]:.0f})",
                va="center", ha="left" if d_ > 0 else "right", fontsize=7.5, color=DARK)
    a1.set_xlim(min(dif) * 1.45, max(dif) * 1.55)

    # ---- 2 净值 ----
    a2 = ax[0, 1]
    for s, nm, cl, lw in ((base, "主策略单独 (3.57)", GREY, 1.4),
                          (nav.S0, "247 定稿 (5.41)", BLUE, 1.8),
                          (nav.S2, "248 定稿 (5.76)", RED, 2.0)):
        s = pd.Series(s.values, index=pd.to_datetime(s.index.astype(str)))
        a2.plot(s.index, (1 + s / 100).cumprod(), color=cl, lw=lw, label=nm)
    a2.set_yscale("log")
    a2.set_title("净值（对数轴）：钱从 2024 挪到 2018·2020，风险来源没变", fontsize=12, color=DARK)
    a2.legend(fontsize=8, loc="upper left")
    a2.grid(alpha=0.25)

    # ---- 3 方案对比 ----
    a3 = ax[1, 0]
    names = ["主策略\n单独", "S0\n247粗占用", "S1\n只改占用口径", "S2\n+重叠日让位",
             "S2 压力\n深低开归零"]
    kama = [3.57, 5.41, 5.48, 5.76, 4.12]
    cols = [GREY, GREY, BLUE, RED, ORANGE]
    bars = a3.bar(range(5), kama, 0.6, color=cols)
    a3.set_xticks(range(5))
    a3.set_xticklabels(names, fontsize=9)
    a3.set_ylabel("卡玛")
    a3.set_ylim(0, 7)
    a3.set_title("两步各值多少：口径 +0.07，让位 +0.28", fontsize=12, color=DARK)
    a3.axhline(5.41, color=GREY, ls=":", lw=1)
    for b, v in zip(bars, kama):
        a3.text(b.get_x() + b.get_width() / 2, v + 0.12, f"{v:.2f}", ha="center",
                fontsize=11, fontweight="bold", color=DARK)
    a3.grid(axis="y", alpha=0.25)

    # ---- 4 重叠日散点 ----
    a4 = ax[1, 1]
    dp = ov[ov.路径 == "深低开"]
    bp = ov[ov.路径 == "破位"]
    lim = (-14, 12)
    a4.plot(lim, lim, color=GREY, ls="--", lw=1)
    a4.axhline(0, color=GREY, lw=0.6)
    a4.axvline(0, color=GREY, lw=0.6)
    a4.scatter(dp.主策收益, dp.补位收益, s=52, color=RED, alpha=0.75,
               edgecolor="white", label=f"深低开 {len(dp)} 天（补位赢 {(dp.补位收益>dp.主策收益).mean()*100:.0f}%）")
    a4.scatter(bp.主策收益, bp.补位收益, s=52, color=BLUE, alpha=0.85, marker="s",
               edgecolor="white", label=f"破位 {len(bp)} 天（都亏，补位只 30% 仓）")
    a4.set_xlim(*lim)
    a4.set_ylim(-17, 33)
    a4.set_xlabel("主策略那笔的收益 %")
    a4.set_ylabel("补位那笔的收益 %")
    a4.set_title("48 天两边都想要：对角线以上 = 让位划算", fontsize=12, color=DARK)
    a4.legend(fontsize=8, loc="upper left")
    a4.grid(alpha=0.25)
    a4.text(0.97, 0.03, f"均值差 +{ov.补位收益.sub(ov.主策收益).mean():.2f}pp  t="
            f"{ov.补位收益.sub(ov.主策收益).mean()/(ov.补位收益.sub(ov.主策收益).std()/np.sqrt(len(ov))):.2f}",
            transform=a4.transAxes, ha="right", fontsize=9, color=DARK)

    fig.suptitle("第248轮  主策略与补位的资金调度：谁先做、钱什么时候回来",
                 fontsize=15, fontweight="bold", color=DARK, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(ROOT / "第248轮_资金调度.png", dpi=150, bbox_inches="tight")
    print("saved 第248轮_资金调度.png")

    print(f"S0 {metrics(nav.S0)}  S2 {metrics(nav.S2)}")


if __name__ == "__main__":
    main()
