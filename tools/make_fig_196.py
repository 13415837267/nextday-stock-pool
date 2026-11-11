# -*- coding: utf-8 -*-
"""第 196 轮配图: 名义 1.0x 的真实敞口 + 空仓日的市场质量。

四格:
  (a) 逐笔目标仓位分布 —— 1.5 那 71 笔是杠杆旁路留下的证据
  (b) A/B/C 三变体净值(对数) —— 去掉透支后的真实代价
  (c) 主策略在场日 vs 空仓日, 市场等权基准的日均收益
  (d) 副策略在"主策略在场日" vs "主策略空档日"的日均收益
"""
from __future__ import annotations

import copy
import json
import os
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest import prepare_context, run_backtest   # noqa: E402
from src.config import ROOT, load_config                # noqa: E402

OUT = ROOT / "output"
FIG = Path("/workspace/第196轮_名义无杠杆与空仓的价值.png")

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Sans CJK SC", "WenQuanYi Zen Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

C_MAIN, C_RED, C_GREEN, C_GRAY = "#1f3b57", "#c0392b", "#2d6a4f", "#8a8a8a"


def _run_variants(days: int) -> dict:
    os.environ["BT_FULL_TRADES"] = "1"
    base = load_config()
    ctx = prepare_context(base, days)
    out = {}
    for name, mut in (("A", None), ("B", "off"), ("C", "cap")):
        cfg = copy.deepcopy(base)
        cfg["execution"]["pos_frac_cap"] = 0        # config 已落地 1.0, A 必须显式关掉才是 A
        if mut == "off":
            cfg["execution"]["mkt_boost_tiers"] = []
        elif mut == "cap":
            cfg["execution"]["pos_frac_cap"] = 1.0
        res = run_backtest(cfg, days, ctx)
        out[name] = res
    return out


def main() -> None:
    days = 1500
    data = json.loads((OUT / f"exp_lev_196_{days}.json").read_text(encoding="utf-8"))["variants"]
    res = _run_variants(days)
    attrib = json.loads((OUT / "exp_attrib_196_1500.json").read_text(encoding="utf-8"))

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.6))
    fig.subplots_adjust(left=0.065, right=0.975, top=0.90, bottom=0.08, hspace=0.42, wspace=0.24)

    # (a) 敞口分布
    ax = axes[0, 0]
    pf = [float(t.get("pos_frac") or 0) for t in res["A"]["trades_all"] if t.get("ret_pct") is not None]
    cnt = Counter(round(x, 2) for x in pf)
    xs = sorted(cnt)
    ys = [cnt[x] for x in xs]
    cols = [C_RED if x > 1.0001 else C_MAIN for x in xs]
    ax.bar([str(x) for x in xs], ys, color=cols, width=0.72)
    for i, (x, y) in enumerate(zip(xs, ys)):
        if y >= 8:
            ax.text(i, y + 2, str(y), ha="center", va="bottom", fontsize=8.5,
                    color=C_RED if x > 1.0001 else C_MAIN)
    ax.set_ylim(0, max(ys) * 1.30)
    n_over = sum(1 for x in pf if x > 1.0001)
    ax.annotate(f"目标仓位 1.5：{n_over} 笔（{n_over / len(pf):.0%}）\n引擎自动开 0.5 倍融资",
                xy=(len(xs) - 1, cnt[1.5] * 1.01), xytext=(0.15, max(ys) * 1.12),
                fontsize=9, color=C_RED, ha="left",
                arrowprops=dict(arrowstyle="->", color=C_RED, lw=1.2,
                                connectionstyle="arc3,rad=-0.18"))
    ax.set_title("(a) leverage=1.0 的账户，逐笔目标仓位分布", fontsize=11.5, color=C_MAIN, pad=8)
    ax.set_ylabel("笔数", fontsize=9.5)
    ax.set_xlabel("目标仓位（占总资产）", fontsize=9.5)
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # (b) 净值
    ax = axes[0, 1]
    lab = {"A": ("A 现行：恐慌日 ×1.5（可透支）", C_RED),
           "B": ("B 关闭恐慌日加仓", C_GRAY),
           "C": ("C ×1.5 但截断到 1.0（采纳）", C_GREEN)}
    for k in ("A", "B", "C"):
        eq = res[k]["equity"]
        nav = np.array([x["nav"] for x in eq], dtype=float) / eq[0]["nav"]
        key = {"A": "A_现行_boost1.5可透支", "B": "B_关闭boost", "C": "C_boost截断到1.0"}[k]
        v = data[key]
        ax.plot(nav, lw=2.0 if k == "A" else 1.6, color=lab[k][1],
                label=f"{lab[k][0]}\n{v['总收益率%'] / 100:.0f}×　年化 {v['年化收益率%']:.1f}%　卡玛 {v['卡玛比率']}")
    ax.set_yscale("log")
    ax.axhline(1.0, color="#cccccc", lw=0.8, ls=":")
    ax.set_title("(b) 同一批信号，只差“能不能透支”", fontsize=11.5, color=C_MAIN, pad=8)
    ax.set_ylabel("净值倍数（对数）", fontsize=9.5)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.22, lw=0.6)
    ax.legend(fontsize=8.2, loc="upper left", frameon=False, labelspacing=0.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # (c) 市场日均：在场 vs 空仓
    ax = axes[1, 0]
    A = attrib["A_择时是否躲开下跌"]
    vals = [A["在场日_市场日均%"], A["空仓日_市场日均%"]]
    ax.bar(["有持仓\n661 天", "空仓\n837 天"], vals, color=[C_GREEN, C_RED], width=0.5)
    for i, v in enumerate(vals):
        ax.text(i, v + (0.012 if v >= 0 else -0.012), f"{v:+.3f}%",
                ha="center", va="bottom" if v >= 0 else "top", fontsize=11,
                color=C_GREEN if v >= 0 else C_RED, fontweight="bold")
    ax.axhline(0, color="#444444", lw=0.9)
    ax.set_ylim(-0.11, 0.22)
    ax.set_title("(c) 主策略在场日 / 空仓日的市场等权日均收益", fontsize=11.5, color=C_MAIN, pad=8)
    ax.set_ylabel("市场日均收益（%）", fontsize=9.5)
    ax.tick_params(labelsize=9)
    ax.text(0.5, 0.90, f"胜率 {A['在场日_市场胜率%']:.0f}%  vs  {A['空仓日_市场胜率%']:.0f}%",
            transform=ax.transAxes, ha="center", fontsize=9.5, color=C_MAIN)
    ax.grid(axis="y", alpha=0.22, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # (d) 副策略日均：重叠 vs 空档
    ax = axes[1, 1]
    B = attrib["B_副策略在空档期"]
    vals = [B["重叠日_副日均%"], B["空档日_副日均%"]]
    ax.bar(["主策略也在场\n516 天", "主策略空档\n200 天"], vals, color=[C_MAIN, "#d9a441"], width=0.5)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:+.3f}%", ha="center", va="bottom", fontsize=11,
                color=C_MAIN if i == 0 else "#8a6a1f", fontweight="bold")
    ax.axhline(0, color="#444444", lw=0.9)
    ax.set_ylim(-0.08, 0.58)
    ax.set_title("(d) 副策略的日收益，按“主策略当天在不在场”切分", fontsize=11.5, color=C_MAIN, pad=8)
    ax.set_ylabel("副策略日均收益（%）", fontsize=9.5)
    ax.tick_params(labelsize=9)
    ax.text(0.5, 0.88, "逐笔：冲突 95 笔 +2.90%　/　独占空档 11 笔 −0.49%",
            transform=ax.transAxes, ha="center", fontsize=9.2, color=C_MAIN)
    ax.grid(axis="y", alpha=0.22, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.suptitle("第 196 轮：leverage=1.0 并没有关掉杠杆；“空仓”也不是浪费",
                 fontsize=14.5, color=C_MAIN, y=0.965)
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=155, facecolor="white")
    print("写出", FIG)


if __name__ == "__main__":
    main()
