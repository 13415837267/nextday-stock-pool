"""第254轮：四格判决图。

① universe 从 1,599 涨到 5,535 —— 同一个"11 只"在 18 年里严了 3.5 倍
② 2500 日：绝对只数 vs 达标率，6 组对照
③ 振幅因子：事后分档单调，但一进评分就崩（选择效应）
④ 连亏之后该做还是该停：连亏 2 笔后下一笔反而更好
"""
from __future__ import annotations

import glob
import gzip
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

RED, DARK, GREY, LIGHT = "#c0392b", "#2c3e50", "#95a5a6", "#d5dbdf"
plt.rcParams["axes.unicode_minus"] = False
for fp in ("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
           "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
           "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"):
    if Path(fp).exists():
        font_manager.fontManager.addfont(fp)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()
        break
plt.rcParams["figure.dpi"] = 130


def universe_series() -> pd.DataFrame:
    """各年 universe 中位(抽样 3 个交易日/年)。"""
    files = sorted(glob.glob(os.path.join(ROOT, "data/panel/*.csv.gz")))
    by_year = defaultdict(list)
    for f in files:
        by_year[os.path.basename(f)[:4]].append(f)
    rows = []
    for y in sorted(by_year):
        fs = by_year[y][:: max(1, len(by_year[y]) // 3)][:3]
        ns = []
        for f in fs:
            with gzip.open(f, "rt") as fh:
                ns.append(sum(1 for _ in fh) - 1)
        rows.append({"y": int(y), "univ": float(np.mean(ns))})
    return pd.DataFrame(rows)


def main() -> None:
    fig, ax = plt.subplots(2, 2, figsize=(13.6, 9.0))
    fig.subplots_adjust(hspace=0.38, wspace=0.26, top=0.90, bottom=0.09, left=0.07, right=0.97)

    # ---------- 1 universe ----------
    a1 = ax[0, 0]
    u = universe_series()
    a1.fill_between(u.y, u.univ, color=RED, alpha=0.13)
    a1.plot(u.y, u.univ, color=RED, lw=2.2, marker="o", ms=3.4)
    a1.set_ylabel("全市场股票数", color=RED)
    a1.tick_params(axis="y", labelcolor=RED)
    a1.set_ylim(0, 6400)
    a1.set_xticks(range(2009, 2027, 2))
    a2 = a1.twinx()
    strict = 11 / u.univ * 1e4
    a2.plot(u.y, strict, color=DARK, lw=1.8, ls="--", marker="s", ms=3.2)
    a2.set_ylabel("「11 只」的严格度（bp）", color=DARK)
    a2.tick_params(axis="y", labelcolor=DARK)
    a2.set_ylim(0, 80)
    a1.set_title("① 同一个「11 只」，2009 年是 69bp，2026 年只剩 20bp", fontsize=11.5, loc="left")
    a1.annotate(f"{u.univ.iloc[0]:.0f} 只", (u.y.iloc[0], u.univ.iloc[0]),
                textcoords="offset points", xytext=(8, -16), fontsize=8.8, color=RED)
    a1.annotate(f"{u.univ.iloc[-1]:.0f} 只", (u.y.iloc[-1], u.univ.iloc[-1]),
                textcoords="offset points", xytext=(-44, 6), fontsize=8.8, color=RED)
    a1.text(0.02, 0.06, "虚线：绝对门槛的严格度随上市数量翻倍而悄悄收紧 3.5 倍",
            transform=a1.transAxes, fontsize=8.4, color=GREY)

    # ---------- 2 三窗口 × 池深档 ----------
    a3 = ax[0, 1]
    src = {
        "1500 日": "output/w254_tiernofill_1500.pkl",
        "2500 日": "output/w254_tiernofill_2500.pkl",
        "3200 日": "output/w254_fullnofill_3200.pkl",
    }
    # 两个脚本里的方案名不同(254e 叫 "E 253+11-15半仓", 254f 叫 "E 21/16/11半/0"), 按前缀取
    plans = ["A 253前", "B 253定稿", "E "]
    short = ["A 253 前\n11-20 半仓", "B 253\n11-20 满仓", "E 254\n11-15 半仓"]
    cols = [GREY, "#8b6f47", RED]
    data = {}
    for lab, fp in src.items():
        lst = pickle.load(open(os.path.join(ROOT, fp), "rb"))
        row = []
        for k in plans:
            hit = [o for o in lst if o["name"].startswith(k)]
            row.append(hit[0]["m"]["卡玛比率"])
        data[lab] = row
    xs = np.arange(len(src))
    w = 0.25
    for j, (lab, c) in enumerate(zip(short, cols)):
        vals = [data[k][j] for k in src]
        a3.bar(xs + (j - 1) * w, vals, w * 0.92, color=c, label=lab)
        for x, v in zip(xs, vals):
            a3.text(x + (j - 1) * w, v + 0.10, f"{v:.2f}", ha="center", fontsize=8.6,
                    fontweight="bold", color=DARK)
    a3.set_xticks(xs)
    a3.set_xticklabels(list(src.keys()), fontsize=9.6)
    a3.set_ylabel("卡玛比率（主策略单独）")
    a3.set_ylim(0, 10.2)
    a3.legend(fontsize=7.8, loc="upper right", frameon=False, ncol=1)
    a3.set_title("② 三个窗口，E 全部最高：253 把 11-20 当成一档太粗", fontsize=11.5, loc="left")
    a3.text(0.02, 0.96, "含补位的正式口径下 B 与 E 打平（5.60 / 5.61）\n"
                        "差别在补位哑火的行情：2500 日回撤 -20.03% vs -17.41%",
            transform=a3.transAxes, fontsize=7.8, color=DARK, va="top")

    # ---------- 3 amp20 ----------
    a4 = ax[1, 0]
    am = pickle.load(open(os.path.join(ROOT, "output/w254_amp_2500.pkl"), "rb"))
    ks = [o["m"]["卡玛比率"] for o in am]
    xs2 = np.arange(len(am))
    a4.bar(xs2, ks, 0.62, color=[RED] + [GREY] * (len(am) - 1))
    for i, v in enumerate(ks):
        a4.text(i, v + 0.08, f"{v:.2f}", ha="center", fontsize=9.4,
                fontweight="bold", color=DARK)
    a4.set_xticks(xs2)
    a4.set_xticklabels([o["name"].replace("amp20=", "权重 ") for o in am], fontsize=9)
    a4.set_ylabel("卡玛比率")
    a4.set_ylim(0, 7.6)
    a4.set_title("③ 振幅因子：事后分档单调，一进评分就崩", fontsize=11.5, loc="left")

    # 右上角插图: 事后分档
    tail = pd.read_pickle(os.path.join(ROOT, "output/w254_tail.pkl"))
    tail = tail[tail.pool_n > 0].copy()
    tail["q"] = pd.qcut(tail.amp20, 4, labels=["Q1", "Q2", "Q3", "Q4"])
    g = tail.groupby("q", observed=True).ret_pct.mean()
    ins = a4.inset_axes([0.665, 0.60, 0.305, 0.31])
    ins.bar(range(4), g.values, 0.6, color="#7f8c8d")
    ins.set_xticks(range(4))
    ins.set_xticklabels(["", "", "", ""])
    ins.set_xlabel("左：低波 → 右：高波", fontsize=6.4, color=GREY, labelpad=1.5)
    ins.set_ylim(0, 3.4)
    ins.set_title("同一批票，事后按振幅四分", fontsize=7.0, color=GREY)
    for i, v in enumerate(g.values):
        ins.text(i, v + 0.07, f"{v:.2f}", ha="center", fontsize=7.2, color=DARK)
    ins.tick_params(labelsize=6)
    a4.text(0.02, 0.97, "事后：越波动越好（0.69→2.09%）\n事前：权重越大越差（5.60→3.36）",
            transform=a4.transAxes, fontsize=8.4, color=DARK, va="top")

    # ---------- 4 连亏 ----------
    a5 = ax[1, 1]
    # 253 定稿的 495 笔主策略(w254_tail.pkl 就是这批 + 波动字段), 不是 253 之前的 559 笔
    m = pd.read_pickle(os.path.join(ROOT, "output/w254_tail.pkl"))
    m = m.copy().sort_values("buy_date").reset_index(drop=True)
    st, cur = [], 0
    for x in (m.ret_pct < 0):
        cur = cur + 1 if x else 0
        st.append(cur)
    m["prev"] = pd.Series(st).shift(1).fillna(0).astype(int)
    grp = m.groupby("prev").agg(n=("ret_pct", "size"), avg=("ret_pct", "mean"),
                                win=("ret_pct", lambda s: (s > 0).mean() * 100))
    grp = grp[grp.n >= 15]
    xs3 = np.arange(len(grp))
    base = m.ret_pct.mean()
    a5.bar(xs3, grp.avg.values, 0.58,
           color=[RED if v >= base else GREY for v in grp.avg.values])
    for i, (v, n, w) in enumerate(zip(grp.avg.values, grp.n.values, grp.win.values)):
        a5.text(i, v + 0.06, f"{v:+.2f}%", ha="center", fontsize=9.2, color=DARK,
                fontweight="bold")
        a5.text(i, -0.30, f"{n} 笔\n胜率 {w:.0f}%", ha="center", fontsize=7.6, color=GREY)
    a5.set_xticks(xs3)
    a5.set_xticklabels([f"前面连亏 {i} 笔" for i in grp.index], fontsize=8.8)
    a5.set_ylabel("下一笔平均收益 %")
    a5.set_ylim(-0.6, 2.6)
    a5.axhline(base, color=DARK, ls=":", lw=1)
    a5.text(2.5, base + 0.32, f"全样本 {base:.2f}%", ha="center", fontsize=8, color=DARK)
    a5.set_title("④ 连亏之后该做还是该停：没有「越亏越该停」这回事", fontsize=11.5, loc="left")

    fig.suptitle("第 254 轮　池深档拆细（②）与两条被否掉的路（③④）",
                 fontsize=13.5, x=0.07, ha="left", y=0.965)
    out = os.path.join(ROOT, "output/第254轮_三条路.png")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print("→", out)


if __name__ == "__main__":
    main()
