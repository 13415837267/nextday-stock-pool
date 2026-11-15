"""第二百三十四轮配图"""
from __future__ import annotations

import sys
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

CN = {"pct": "当日涨跌幅", "amp": "当日振幅", "close_pos": "收盘位置", "low_tail": "下影线",
      "up_shadow": "上影线", "gap": "当日缺口", "intraday": "日内涨跌", "vr": "量比",
      "turnover_rate": "换手率", "amt_yi": "成交额", "amt_ratio": "额比20日",
      "circ_mv": "流通市值", "bias5": "5日乖离", "bias10": "10日乖离",
      "bias20": "20日乖离", "bias60": "60日乖离", "dist_high60": "距60日高点",
      "dist_low60": "距60日低点", "dist_high20": "距20日高点", "ret3": "3日涨幅",
      "ret5": "5日涨幅", "ret10": "10日涨幅", "ret20": "20日涨幅", "vol20": "20日波动",
      "maxdd20": "20日回撤", "dn_streak": "连跌天数", "up_streak": "连涨天数",
      "rel_pct": "相对大盘", "rs20": "20日相对强度", "breadth": "上涨家数占比",
      "idx_dist": "指数距MA20", "listed": "上市天数"}


def fig1() -> None:
    d = pd.read_csv(OUT / "w234_uni_lowopen.csv")
    d["一致率"] = d.apply(
        lambda r: max(r["正年数"], r["年数"] - r["正年数"]) / r["年数"], axis=1)
    d["名"] = d["特征"].map(lambda x: CN.get(x, x))
    d = d[(d["合计n"] > 3e6) & (d["年数"] >= 12)].sort_values("极差", key=lambda s: s.abs())
    fig, a = plt.subplots(figsize=(10.2, 7.4))
    cols = ["#c0392b" if c >= 0.9 else ("#e67e22" if c >= 0.7 else "#95a5a6") for c in d["一致率"]]
    a.barh(d["名"], d["极差"], color=cols, height=0.66)
    a.axvline(0, color="#333", lw=1)
    for y, (v, c, n) in enumerate(zip(d["极差"], d["一致率"], d["年数"])):
        a.text(v + (0.02 if v >= 0 else -0.02), y, f"{v:+.2f}   一致 {int(round(c*n))}/{n}",
               va="center", ha="left" if v >= 0 else "right", fontsize=8.5, color="#333")
    a.set_xlim(-1.62, 0.62)
    a.set_xlabel("最低档 与 最高档 的次日收益差（百分点）")
    a.set_title("全市场 817 万个「低开」样本：哪些特征真的分得开次日收益", fontsize=13)
    from matplotlib.patches import Patch
    a.legend(handles=[Patch(color="#c0392b", label="17 年里 ≥90% 同向（稳）"),
                      Patch(color="#e67e22", label="70%~90% 同向"),
                      Patch(color="#95a5a6", label="<70% 同向（不稳）")],
             loc="lower right", fontsize=9)
    a.grid(axis="x", alpha=0.22)
    fig.text(0.5, 0.015, "口径：T 日收盘算特征，T+1 低开时以开盘价买入、T+2 收盘卖出。"
             "「距20日高点 / 20日回撤 / 60日乖离」都是 17/17 年同向——它们是同一件事：离前高越远，次日反弹越强。",
             ha="center", fontsize=9, color="#444")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(ROOT / "第234轮_全市场特征排序.png", dpi=150)
    print("图1 ok", flush=True)


def fig2() -> None:
    d = pd.read_csv(OUT / "w234_curve_pct_lowopen.csv")
    d = d[d["n"] >= 3000]
    fig, a = plt.subplots(figsize=(10.4, 5.6))
    a.axvspan(-6, -3, color="#f1c40f", alpha=0.22, label="策略实际选择的区间")
    a.axhline(0, color="#333", lw=0.9)
    a.plot(d["mid"], d["y"], "-o", color="#c0392b", lw=2.2, ms=4.5, label="T+1开盘买 → T+2收盘卖")
    a.plot(d["mid"], d["y1"], "--s", color="#2980b9", lw=1.2, ms=3.5, alpha=0.75,
           label="只持 1 天（T+1 收盘卖）")
    a.set_xlabel("信号日涨跌幅（%）")
    a.set_ylabel("平均收益（%）")
    a.set_title("涨跌幅与次日收益不是单调关系，是两头翘的 U 形", fontsize=13)
    a.legend(fontsize=9.5, loc="upper center")
    a.grid(alpha=0.25)
    a.set_xlim(-11.5, 11.5)
    a.annotate(f"跌 4% 附近\n{d[d['mid'].between(-4.5,-3.5)]['y'].mean():.2f}%",
               (-4, float(d[d["mid"].between(-4.5, -3.5)]["y"].mean())),
               textcoords="offset points", xytext=(-4, -34), ha="center", fontsize=9,
               color="#7d6608")
    a.annotate("平盘最差\n≈0.12%", (0.25, 0.127), textcoords="offset points",
               xytext=(34, -26), fontsize=9, color="#555",
               arrowprops=dict(arrowstyle="->", color="#555", lw=1))
    fig.text(0.5, 0.015, "所以「按涨跌幅排序取第 1 名」必然失败：排序只能取两端，"
             "而这两端恰恰都不是最好的。跌最多的端是 −0.03%，涨最多的端是 +0.38%，最优点在分布内部。",
             ha="center", fontsize=9.5, color="#444")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(ROOT / "第234轮_涨跌幅的U形.png", dpi=150)
    print("图2 ok", flush=True)


def fig3() -> None:
    tab = pd.read_csv(OUT / "w234_2d_high20_pct.csv", index_col=0)
    cnt = pd.read_csv(OUT / "w234_2d_high20_pct_n.csv", index_col=0)
    fig, a = plt.subplots(figsize=(9.8, 5.8))
    m = tab.where(cnt >= 8000)
    im = a.imshow(m.values, cmap="RdYlGn", vmin=-0.3, vmax=1.3, aspect="auto")
    a.set_xticks(range(len(tab.columns)))
    a.set_xticklabels(tab.columns, fontsize=9, rotation=0)
    a.set_yticks(range(len(tab.index)))
    a.set_yticklabels(tab.index, fontsize=9)
    a.set_xlabel("当日涨跌幅（%）")
    a.set_ylabel("距 20 日最高点（%）")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m.values[i, j]
            if np.isnan(v):
                a.text(j, i, "样本\n不足", ha="center", va="center", fontsize=7.5, color="#999")
            else:
                a.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=10,
                       color="#1a1a1a" if v < 0.85 else "white")
    a.set_title("控制住跌幅之后，「距 20 日高点」还有没有增量", fontsize=13)
    fig.colorbar(im, ax=a, shrink=0.82, label="次日平均收益 %")
    fig.text(0.5, 0.015, "看每一列内部：跌 5~3% 这一列，距高点 −8~−5% 是 0.51%，深跌 −16% 以下是 0.22%。"
             "差异真实存在但只有 0.3pp，远小于跌幅本身的贡献。",
             ha="center", fontsize=9.5, color="#444")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(ROOT / "第234轮_二维交叉.png", dpi=150)
    print("图3 ok", flush=True)


def fig4() -> None:
    f = OUT / "w234_single_lowopen.csv"
    d = pd.read_csv(f)
    d["名"] = d["特征"].map(lambda x: CN.get(x, x))
    d = d.sort_values("年化", ascending=False).head(10)
    fig, a = plt.subplots(figsize=(9.6, 5.4))
    lab = [f"{r['名']}（{r['方向']}）" for _, r in d.iterrows()]
    cols = ["#c0392b" if v > 0 else "#27ae60" for v in d["年化"]]
    a.barh(lab, d["年化"], color=cols, height=0.62)
    a.axvline(84.13, color="#2980b9", ls="--", lw=1.6)
    a.text(84.13, -0.9, "现有策略 84.1%", color="#2980b9", fontsize=10, ha="center")
    a.axvline(0, color="#333", lw=1)
    for y, v in enumerate(d["年化"]):
        a.text(v + 1.2, y, f"{v:.1f}%", va="center", fontsize=9)
    a.set_xlabel("年化 %（每天按该特征排序买第 1 名，全史 2500 日）")
    a.set_title("最好的单因子 51.6%，比现有组合低 32.5 个百分点", fontsize=13)
    a.invert_yaxis()
    a.set_xlim(-4, 96)
    a.grid(axis="x", alpha=0.22)
    fig.text(0.5, 0.015, "注意「60日乖离（大优）」有 44%、而策略给乖离负权重却是最优——同一特征在不同样本空间里方向相反，"
             "这正是单因子排序与真实策略的差别。", ha="center", fontsize=9.5, color="#444")
    fig.tight_layout(rect=[0, 0.045, 1, 1])
    fig.savefig(ROOT / "第234轮_单因子天花板.png", dpi=150)
    print("图4 ok", flush=True)




def fig5() -> None:
    b = pd.read_csv(OUT / "w234_board.csv")
    w = pd.read_csv(OUT / "w234_weight.csv")
    r = pd.read_csv(OUT / "w234_range.csv")
    b["组"] = "板块 / 市值 / 振幅"
    w["组"] = "评分权重"
    r["组"] = "选股区间"
    d = pd.concat([b[["方案", "Δ年化", "Δ笔数", "组"]],
                   w[["方案", "Δ年化", "Δ笔数", "组"]],
                   r[["方案", "Δ年化", "Δ笔数", "组"]]], ignore_index=True)
    d = d[d["Δ年化"] != 0].sort_values("Δ年化")
    d["名"] = d["方案"].str.replace(r"^[A-F]\d\s*", "", regex=True)
    fig, a = plt.subplots(figsize=(10.4, 8.2))
    cmap = {"板块 / 市值 / 振幅": "#2980b9", "评分权重": "#8e44ad", "选股区间": "#e67e22"}
    cols = [cmap[g] for g in d["组"]]
    a.barh(d["名"], d["Δ年化"], color=cols, height=0.68)
    a.axvline(0, color="#333", lw=1.2)
    for y, (v, dn) in enumerate(zip(d["Δ年化"], d["Δ笔数"])):
        a.text(v - 1.8, y, f"{v:+.1f}pp   笔数{dn:+d}", va="center", ha="right",
               fontsize=8.8, color="#333")
    a.set_xlim(min(d["Δ年化"]) - 26, 4)
    a.set_xlabel("年化变化（百分点，相对现行 84.13%）")
    a.set_title("全市场归纳出的 21 个改法，没有一个为正", fontsize=13.5)
    from matplotlib.patches import Patch
    a.legend(handles=[Patch(color=c, label=g) for g, c in cmap.items()],
             loc="center right", fontsize=9.5, bbox_to_anchor=(1.0, 0.42))
    a.grid(axis="x", alpha=0.22)
    a.tick_params(axis="y", labelsize=9.5)
    fig.text(0.5, 0.015, "最轻的一刀（区间下限 −6→−7，只放宽 1 个百分点）也要掉 9.2pp。"
             "原因不是特征不对，是「每天只买第 1 名」：筛掉谁，名额就让给第 2 名。",
             ha="center", fontsize=9.5, color="#444")
    fig.tight_layout(rect=[0, 0.035, 1, 1])
    fig.savefig(ROOT / "第234轮_21个改法全负.png", dpi=150)
    print("图5 ok", flush=True)




def fig6() -> None:
    y = pd.read_csv(OUT / "w234_gbm_year_lowopen.csv")
    imp = pd.read_csv(OUT / "w234_gbm_imp_lowopen.csv").head(12)
    imp["名"] = imp["特征"].map(lambda x: CN.get(x, x))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.4),
                                 gridspec_kw={"width_ratios": [1.05, 1]})
    a1b = a1.twinx()
    a1.bar(y["year"], y["q5-q1"], color="#2980b9", alpha=0.75, width=0.62,
           label="多头五分位超额（左轴）")
    a1.set_ylabel("多头 − 空头 五分位超额 %", color="#2980b9")
    a1.tick_params(axis="y", colors="#2980b9")
    a1b.plot(y["year"], y["ic"], "-o", color="#c0392b", lw=2, ms=6, label="IC（右轴）")
    a1b.set_ylabel("IC", color="#c0392b")
    a1b.tick_params(axis="y", colors="#c0392b")
    a1.set_title("全市场的可预测性在逐年衰减", fontsize=12)
    a1.set_ylim(0, 0.95)
    a1b.set_ylim(0, 0.105)
    a1.set_xticks(y["year"])
    a1.set_xticklabels(y["year"], rotation=45, fontsize=9)
    a1.grid(axis="y", alpha=0.2)
    for x, v in zip(y["year"], y["ic"]):
        a1b.annotate(f"{v:.3f}", (x, v), xytext=(0, 9), textcoords="offset points",
                     ha="center", fontsize=8, color="#c0392b", zorder=6,
                     path_effects=[pe.withStroke(linewidth=3, foreground="white")])
    h1, l1 = a1.get_legend_handles_labels()
    h2, l2 = a1b.get_legend_handles_labels()
    a1.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=9)

    a2.barh(imp["名"][::-1], imp["gain占比%"][::-1], color="#8e44ad", height=0.62)
    a2.set_xlabel("GBM 增益占比 %")
    a2.set_title("32 个特征没有一个是主导：最高仅 6.0%", fontsize=12)
    a2.set_xlim(0, 7.4)
    for yy, v in enumerate(imp["gain占比%"][::-1]):
        a2.text(v + 0.09, yy, f"{v:.1f}", va="center", fontsize=8.5)
    a2.grid(axis="x", alpha=0.2)
    a2.tick_params(axis="y", labelsize=9)

    fig.text(0.5, 0.015, "训练 2009-2017、验证 2018-2026。验证 IC 从 2018 年 0.093 降到 2026 年 0.026；"
             "而策略权重最大的「20 日乖离」在右图排倒数第三（2.9%）——权重不是按重要性分配的。",
             ha="center", fontsize=9.5, color="#444")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(ROOT / "第234轮_可预测性衰减.png", dpi=150)
    print("图6 ok", flush=True)


if __name__ == "__main__":
    only = sys.argv[1] if len(sys.argv) > 1 else ""
    if only in ("", "1"):
        fig1()
    if only in ("", "2"):
        fig2()
    if only in ("", "3"):
        fig3()
    if only in ("", "4"):
        fig4()
    if only in ("", "5"):
        fig5()
    if only in ("", "6"):
        fig6()
