"""第252轮: 补位池收窄到 3 只的判决图

四格:
  1. 池宽-卡玛横条: 3 只不是"更精", 但"换挑法"能把损失补回来
  2. 深低开 43 笔按名次分档: 为什么要跳过最崩的 3 只
  3. 破位 28 笔: 排名优先 vs 开盘最低, 逐笔对照
  4. 净值: 主策略 / 251 定稿 / 252 定稿
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"

RED, GREEN, BLUE, GOLD, GREY = "#D94A3D", "#2E9E62", "#3A7BD5", "#D4A017", "#8b949e"
DARK = "#22272e"
BG = "#ffffff"

for fp in ("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
           "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
           "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"):
    if Path(fp).exists():
        font_manager.fontManager.addfont(fp)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()
        break
plt.rcParams.update({"axes.unicode_minus": False, "figure.facecolor": BG,
                     "axes.facecolor": BG, "axes.edgecolor": "#d0d7de",
                     "grid.color": "#e8ecf0", "text.color": DARK,
                     "axes.labelcolor": DARK, "xtick.color": DARK, "ytick.color": DARK})


def load(fn):
    return {o["name"]: o for o in pickle.load(open(OUT / fn, "rb"))}


def main() -> None:
    A = load("w252_pool_2500.pkl")
    Bk = load("w252b_pick_2500.pkl")
    F = load("w252f_skip_2500.pkl")
    D = load("w252d_pct_2500.pkl")
    C = load("w252c_mix_2500.pkl")

    fig, ax = plt.subplots(2, 2, figsize=(13.6, 10.0))
    fig.suptitle("第252轮 · 补位池收到 3 只：卡玛 5.37 → 5.23，补位 71 → 51 笔",
                 fontsize=15, color=DARK, y=0.985)

    # ---------- 1 池宽-卡玛 ----------
    a1 = ax[0, 0]
    pts = [
        ("3只·跌幅榜候选", 163, 2.55, GREY),
        ("2只/2只", 36, 4.01, GREY),
        ("5只/5只", 56, 4.69, GREY),
        ("3只/3只·前3名", 47, 5.00, GOLD),
        ("3只/3只·跳3名 ★252", 51, 5.23, RED),
        ("3只/8只", 71, 5.28, BLUE),
        ("10只/8只·251定稿", 71, 5.37, BLUE),
        ("10只/8只·挑最低开", 71, 5.38, GREEN),
    ]
    ys = np.arange(len(pts))[::-1]
    vals = [v for _, _, v, _ in pts]
    a1.barh(ys, vals, 0.62, color=[c for _, _, _, c in pts], alpha=0.9)
    a1.axvline(4.52, color=GOLD, ls="--", lw=1.1, zorder=0)
    a1.set_yticks(ys)
    a1.set_yticklabels([f"{nm}  ·  {n}笔" for nm, n, _, _ in pts], fontsize=8.6)
    a1.set_xlabel("卡玛比率（虚线 = 主策略单独 4.52）")
    a1.set_title("池窄了不掉、池宽了更稳；唯一崩的是改用跌幅榜当候选", fontsize=11.5)
    for y, v in zip(ys, vals):
        a1.text(v + 0.04, y, f"{v:.2f}", va="center", fontsize=9, color=DARK)
    a1.set_xlim(0, 6.3)

    # ---------- 2 深低开按名次 ----------
    a2 = ax[0, 1]
    fl = Bk["B 251定稿 10/8"]["fill"]
    dd = fl[fl.fill_path == "深低开"].copy()
    dd["band"] = pd.cut(dd.fill_rk, [0, 3, 5, 8], labels=["1-3 名", "4-5 名", "6-8 名"])
    g = dd.groupby("band", observed=True).agg(n=("ret_pct", "size"), avg=("ret_pct", "mean"),
                                              win=("ret_pct", lambda s: (s > 0).mean() * 100))
    xs = np.arange(len(g))
    bars = a2.bar(xs, g["avg"], 0.5, color=[GREY, GOLD, GREEN], alpha=0.9)
    for x, (idx, r) in zip(xs, g.iterrows()):
        a2.text(x, r["avg"] + 0.12, f"{r['avg']:+.2f}%", ha="center", fontsize=10, color=DARK)
        a2.text(x, -0.35, f"{int(r['n'])} 笔 · 胜率 {r['win']:.0f}%", ha="center",
                fontsize=8.6, color=GREY)
    a2.axhline(0, color=DARK, lw=0.8)
    a2.set_xticks(xs)
    a2.set_xticklabels(list(g.index), fontsize=10)
    a2.set_ylabel("单笔收益 %")
    a2.set_ylim(-0.9, 3.8)
    a2.set_title("深低开 43 笔：距前高最远的前 3 名单笔只有 +0.43%", fontsize=11.5)

    # ---------- 3 破位 28 笔 两选法（配对散点，对角线之上=改进） ----------
    a3 = ax[1, 0]
    bb = Bk["B 251定稿 10/8"]["fill"]
    ss = C["S 10/8 只破位最低开"]["fill"]
    b1 = bb[bb.fill_path == "破位"][["buy_date", "ret_pct"]].rename(columns={"ret_pct": "rank_first"})
    s1 = ss[ss.fill_path == "破位"][["buy_date", "ret_pct"]].rename(columns={"ret_pct": "lowest"})
    m = b1.merge(s1, on="buy_date", how="inner").reset_index(drop=True)
    lim = (-13, 29)
    a3.plot(lim, lim, color=GREY, ls="--", lw=1)
    up = m[m["lowest"] > m["rank_first"]]
    dn = m[m["lowest"] <= m["rank_first"]]
    a3.scatter(up["rank_first"], up["lowest"], s=44, color=RED, alpha=0.85, zorder=3,
               label=f"改进了 {len(up)} 笔")
    a3.scatter(dn["rank_first"], dn["lowest"], s=44, color=GREY, alpha=0.75, zorder=3,
               label=f"没改进 {len(dn)} 笔")
    a3.set_xlim(*lim)
    a3.set_ylim(*lim)
    a3.set_xlabel("排名第一只达标 · 单笔收益 %")
    a3.set_ylabel("开盘最低的那只 · 单笔收益 %")
    a3.legend(fontsize=8.8, loc="upper left", frameon=False)
    a3.set_title("破位 28 笔：换成挑开盘最低 单笔 1.90% → 3.57%", fontsize=11.5)
    a3.text(0.98, 0.04, "对角线之上 = 这一笔换对了；均值虚线口径见正文",
            transform=a3.transAxes, ha="right", fontsize=8.2, color=GREY)

    # ---------- 4 净值 ----------
    a4 = ax[1, 1]
    for nm, o, c, ls in (("主策略单独", A["A 主策略单独"], GOLD, "-"),
                         ("251 定稿 10/8", Bk["B 251定稿 10/8"], GREY, "-"),
                         ("252 定稿 3/3", F["AA 3/3 第4-6名"], RED, "-")):
        e = pd.DataFrame(o["equity"])
        e["d"] = pd.to_datetime(e["d"])
        a4.plot(e["d"], e["nav"], color=c, lw=1.5, ls=ls, label=f"{nm}（卡玛 {o['m']['卡玛比率']}）")
    a4.set_yscale("log")
    a4.grid(True, which="major", alpha=0.5)
    a4.legend(fontsize=8.6, loc="upper left", frameon=False)
    a4.set_title("净值（对数轴，2500 个交易日）", fontsize=11.5)
    a4.set_ylabel("净值")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = OUT / "第252轮_补位池最多三个.png"
    fig.savefig(out, dpi=150, facecolor=BG)
    print("saved", out)


if __name__ == "__main__":
    main()
