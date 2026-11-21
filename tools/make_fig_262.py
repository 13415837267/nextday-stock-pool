"""第262轮配图: 风格背离 / 资金集中度 门控全否"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
    res = json.load(open(ROOT / "output/exp_style_262.json"))
    sty = pd.read_csv(ROOT / "output/w262_style_daily.csv", index_col=0)
    sty.index = pd.to_datetime(sty.index, format="%Y%m%d")

    fig, axes = plt.subplots(2, 2, figsize=(14.6, 8.8))

    ax = axes[0, 0]
    ax.plot(sty.index, sty["s60"], color=BLUE, lw=1.0)
    ax.axhline(0, color=DARK, lw=0.9)
    ax.axvspan(pd.Timestamp("2017-07-06"), pd.Timestamp("2018-11-21"),
               color=GREY, alpha=0.18, lw=0)
    ax.annotate("2017-07 ~ 2018-11\n503 天不创新高", (pd.Timestamp("2017-11-01"), 38),
                fontsize=9, color=DARK, ha="center")
    ax.set_title("大票/成交额加权 相对 全市场等权 的 60 日累计超额 (pp)\n"
                 "2017 年全年均值 25.5, 与 2015/2016/2020 同一区间", fontsize=11)
    ax.set_ylabel("s60 (pp)", fontsize=9)
    ax.grid(alpha=0.25)

    ax = axes[0, 1]
    qs = res["main_quintiles"]["s60"]
    vals = [q["mean_ret"] for q in qs]
    cols = [GREEN if v < res["baseline"]["mean_ret"] else RED for v in vals]
    ax.bar(range(5), vals, color=cols, alpha=0.85, width=0.66)
    ax.axhline(res["baseline"]["mean_ret"], color=DARK, ls="--", lw=1.2)
    ax.text(4.45, res["baseline"]["mean_ret"] - 0.12, "全样本 1.23%", ha="right",
            va="top", fontsize=9, color=DARK)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=9, color=DARK)
    ax.set_xticks(range(5))
    ax.set_xticklabels(["最低\n(钱在离开大票)", "2", "3", "4", "最高\n(钱在挤进大票)"], fontsize=8.5)
    ax.set_title(f"主腿按信号日 s60 分五档, 单笔收益没有单调关系\n"
                 f"spearman = {res['corr']['s60']['spearman']:+.2f}, "
                 f"2017 年 42% 的交易在最低档", fontsize=11)
    ax.set_ylabel("单笔平均收益 (%)", fontsize=9)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 0]
    base = res["baseline"]["log_sum_pct"]
    vs = sorted(res["variants"], key=lambda x: x["stats"]["log_sum_pct"] - base)
    delta = [v["stats"]["log_sum_pct"] - base for v in vs]
    ax.barh(range(len(delta)), delta, color=GREEN, alpha=0.8, height=0.82)
    ax.axvline(0, color=DARK, lw=1)
    ax.set_yticks([])
    ax.set_xlabel("相对基线, 逐笔复利对数收益差 (pp)", fontsize=9)
    ax.set_title(f"{len(delta)} 个门控 (7 变量 × 分位/扩张 z × 全砍/半仓)\n"
                 f"最好的一个仍少 {abs(max(delta)):.0f}pp, 全部为负", fontsize=11)
    ax.grid(axis="x", alpha=0.25)
    ax.text(max(delta) - 2, len(delta) - 1.5, f"最好 {max(delta):.0f}pp", ha="right",
            fontsize=9, color=DARK)
    ax.text(min(delta) + 2, 1.5, f"最差 {min(delta):.0f}pp", ha="left", fontsize=9, color=DARK)

    ax = axes[1, 1]
    yrs = res["yearly"]
    xs = [v["s60"] for v in yrs.values() if v["s60"] is not None]
    ys = [v["mean_ret"] for v in yrs.values() if v["s60"] is not None]
    names = [int(y) for y, v in yrs.items() if v["s60"] is not None]
    ax.scatter(xs, ys, s=42, color=BLUE, alpha=0.8, zorder=3)
    for x, y, n in zip(xs, ys, names):
        ax.annotate(str(n)[2:], (x, y), textcoords="offset points", xytext=(4, 4),
                    fontsize=8.5, color=DARK)
    ax.axhline(0, color=DARK, lw=0.9, ls=":")
    i17 = names.index(2017)
    ax.scatter([xs[i17]], [ys[i17]], s=90, facecolors="none", edgecolors=RED, lw=1.6, zorder=4)
    ax.set_title("按自然年: 风格强弱解释不了收益\n"
                 "2017 年 s60=25.5 (中位附近) 却是唯一负年", fontsize=11)
    ax.set_xlabel("当年平均 s60 (pp)", fontsize=9)
    ax.set_ylabel("当年单笔平均收益 (%)", fontsize=9)
    ax.grid(alpha=0.25)

    fig.text(0.5, 0.005,
             "口径: 3000 日回测 (2014-07 ~ 2026-11), 714 笔已平仓交易; 风格变量只用信号日收盘前数据; "
             "门控阈值为扩张分位, 无前视。",
             ha="center", fontsize=8.6, color="#5d6d7e")

    fig.tight_layout(rect=(0, 0.02, 1, 1))
    out = ROOT / "第262轮_风格背离.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print("saved", out)


if __name__ == "__main__":
    main()
