"""第247轮 图: 口径修正 / 深低开扩容 / 限价单证伪 / 路径分仓面"""

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
    c = pd.read_csv(OUT / "w247_picks.csv")
    tr = pd.read_csv(OUT / "w232_trades.csv")
    tr["signal"] = tr["signal"].astype(int)
    cal = sorted(int(f.split("/")[-1].replace(".csv.gz", ""))
                 for f in glob.glob(str(ROOT / "data" / "panel" / "*.csv.gz")))
    cal = np.array([d for d in cal if d >= tr.signal.min()])
    base = pd.Series(0.0, index=cal)
    for dt, g in tr.groupby("signal"):
        base[int(dt)] = g["ret"].mean()

    # ---- 仓位面 ----
    p1s = [0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    p2s = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 1.00]
    pv = pd.DataFrame(index=[f"{p:.0%}" for p in p1s],
                      columns=[f"{p:.0%}" for p in p2s], dtype=float)
    dv = pv.copy()
    for P1 in p1s:
        for P2 in p2s:
            sim = base.copy()
            for path, P in (("破位", P1), ("深低开", P2)):
                s = c[c.path == path]
                v = pd.Series((s.y.values - COST) * P, index=s.trade_date.values).sort_index()
                sim[v.index] = sim[v.index] + v.values
            a_, d_, cm_ = metrics(sim)
            pv.loc[f"{P1:.0%}", f"{P2:.0%}"] = cm_
            dv.loc[f"{P1:.0%}", f"{P2:.0%}"] = d_
    pv.to_csv(OUT / "w247_fig_kama.csv", encoding="utf-8-sig")

    # ---- 图 ----
    fig, ax = plt.subplots(2, 2, figsize=(15.4, 10.4))
    fig.suptitle("第247轮  修掉一个把卡玛抬高 1.1 的未来信息, 再从别的方向补回来",
                 fontsize=17, y=0.985)

    # ① 三个数字
    a0 = ax[0, 0]
    labels = ["245 报的\n(漏每年 1 月)\n198 笔", "246 报的\n(提前知道谁低开)\n256 笔",
              "同规则\n去掉未来信息\n247 笔", "247 定稿\n(深低开扩到前 10)\n290 笔"]
    vals = [5.49, 5.16, 4.03, 5.44]
    cols = [GREY, GREY, BLUE, RED]
    bars = a0.bar(range(4), vals, color=cols, width=0.62,
                  edgecolor="white", linewidth=1.4)
    for i, (b, v) in enumerate(zip(bars, vals)):
        a0.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v + 0.09),
                    ha="center", fontsize=15, fontweight="bold", color=cols[i])
    a0.plot([1.7, 2.3], [4.95, 4.95], color="#1a1a1a", lw=1.0, ls=":")
    a0.annotate("", xy=(2, 4.06), xytext=(2, 5.10),
                arrowprops=dict(arrowstyle="->", color=RED, lw=2.2))
    a0.annotate("−1.13", (2.12, 4.56), fontsize=14, color=RED, fontweight="bold")
    a0.set_xticks(range(4))
    a0.set_xticklabels(labels, fontsize=9.5)
    a0.set_ylabel("卡玛 (各自最优仓位)", fontsize=11)
    a0.set_ylim(0, 6.1)
    a0.axhline(3.57, color="#95a5a6", lw=1.2, ls="--")
    a0.annotate("仅主策略 3.57", (3.42, 3.57), fontsize=9, color="#7f8c8d",
                va="bottom", ha="right")
    a0.set_title("① 246 的 5.16 里, 有 1.13 是「提前知道明天谁低开」撑起来的",
                 fontsize=12.5, loc="left")
    a0.grid(axis="y", alpha=0.25)

    # ② 深低开 N
    a1 = ax[0, 1]
    g = pd.read_csv(OUT / "w247j_grid.csv")
    x = np.arange(len(g))
    b1 = a1.bar(x - 0.19, g.deep_bar, 0.36, color=RED, label="深低开单笔收益(%)")
    a1.set_ylabel("单笔收益 (%)", fontsize=11, color=RED)
    a1.tick_params(axis="y", labelcolor=RED)
    a1.set_xticks(x)
    a1.set_xticklabels([f"N={int(n)}" for n in g.N], fontsize=10)
    for i, v in enumerate(g.deep_bar):
        a1.annotate(f"{v:.2f}", (i - 0.19, v + 0.08), ha="center", fontsize=9,
                    color=RED, fontweight="bold")
    ax1b = a1.twinx()
    ax1b.plot(x + 0.19, g["卡玛"], "o-", color=DARK, lw=2.2, ms=7, label="整体卡玛")
    ax1b.set_ylabel("卡玛", fontsize=11, color=DARK)
    ax1b.set_ylim(3.6, 5.9)
    for i, v in enumerate(g["卡玛"]):
        ax1b.annotate(f"{v:.2f}", (i + 0.19, v + 0.07), ha="center", fontsize=9.5,
                      color=DARK, fontweight="bold")
    a1.set_title("② 深低开这条路原来只盯第 1 名(14 次), 放开到前 10 名变 61 次, 卡玛 4.03→4.94",
                 fontsize=12, loc="left")
    a1.set_ylim(0, 7.2)
    a1.grid(axis="y", alpha=0.2)

    # ③ 限价单证伪
    a2 = ax[1, 0]
    gd = pd.read_csv(OUT / "w247d_grid.csv")
    gd = gd[gd.X < 999]
    xs = gd["X"].values
    a2.plot(xs, gd["单笔"], "o-", color=RED, lw=2.4, ms=7, zorder=4)
    a2.axhline(2.32, color=DARK, lw=1.5, ls="--", zorder=2)
    a2.annotate("不挂单, 收盘卖 = 2.32%", (2.2, 2.30), fontsize=10.5, color=DARK,
                ha="left", va="top", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=DARK, alpha=0.92))
    for xi, yi in zip(xs, gd["单笔"]):
        if xi in (3, 5, 8, 12, 15):
            a2.annotate(f"{yi:.2f}", (xi, yi - 0.14), fontsize=10, color=RED,
                        ha="center", va="top", fontweight="bold")
    a2.set_xlabel("T+2 挂限价卖单的触发线 (+X%)   ← 越往左越早兑现", fontsize=11)
    a2.set_ylabel("单笔收益 (%)", fontsize=11)
    a2.set_xticks(list(xs))
    a2.set_xticklabels([f"+{int(v)}%" for v in xs], fontsize=10)
    a2.set_xlim(1.6, 16.2)
    a2.set_ylim(-0.35, 2.55)
    a2.set_title("③ 提前兑现是错的: 挂 +5% 有 42% 会成交, 单笔却从 2.32 掉到 0.79",
                 fontsize=12, loc="left")
    a2.grid(alpha=0.25)

    # ④ 分仓面
    a3 = ax[1, 1]
    arr = pv.values.astype(float)
    im = a3.imshow(arr, cmap="RdYlGn", aspect="auto", vmin=3.6, vmax=6.3)
    a3.set_xticks(range(len(p2s)))
    a3.set_xticklabels([f"{p:.0%}" for p in p2s], fontsize=10)
    a3.set_yticks(range(len(p1s)))
    a3.set_yticklabels([f"{p:.0%}" for p in p1s], fontsize=10)
    a3.set_xlabel("深低开日仓位", fontsize=11)
    a3.set_ylabel("破位日仓位", fontsize=11)
    for i in range(len(p1s)):
        for j in range(len(p2s)):
            a3.text(j, i, f"{arr[i, j]:.2f}", ha="center", va="center",
                    fontsize=9.5, color="#1a1a1a")
    ci = p1s.index(0.30)
    cj = p2s.index(0.50)
    a3.plot(cj, ci, "*", ms=24, color="#1a1a1a", mec="white", mew=1.5)
    a3.annotate("定稿 30%/50%\n5.44", (cj, ci), (cj - 1.9, ci - 1.05), fontsize=10,
                fontweight="bold", color="#1a1a1a",
                arrowprops=dict(arrowstyle="->", color="#1a1a1a", lw=1.6),
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#1a1a1a", alpha=0.92))
    a3.set_title("④ 分仓面: 深低开仓一路升高(它是个股独立崩盘, 与大盘暴跌不重叠), "
                 "取 50% 不取端点", fontsize=11.5, loc="left")
    cb = plt.colorbar(im, ax=a3, fraction=0.045)
    cb.set_label("卡玛", rotation=270, labelpad=13)

    fig.tight_layout(rect=[0, 0, 1, 0.962])
    p = OUT / "第247轮_修掉未来信息.png"
    fig.savefig(p, dpi=155, facecolor="white")
    print(f"saved {p}")


if __name__ == "__main__":
    main()
