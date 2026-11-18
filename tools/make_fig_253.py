"""第253轮: 池深分档改口径的判决图

四格:
  1. 559 笔按"信号日达标票数"分组: 池越窄, 次日越差 (单笔 + 胜率)
  2. 五组仓位档方案: 年化 / 卡玛
  3. 分半(前 2016-2021 / 后 2022-2026): 改善是不是只吃老样本
  4. 净值: A 现状 vs B 定稿
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
    d = load("w253a_pooltier_2500.pkl")
    A = d["A 现状 1.0/0.5/0.3"]

    tr = pd.DataFrame(A["trades"])
    tr = tr[tr.ret_pct.notna()]
    m = tr[~tr._fillin.fillna(False).astype(bool)].copy()

    fig, ax = plt.subplots(2, 2, figsize=(13.4, 8.6))
    fig.suptitle("第253轮：池深不是仓位旋钮，是择时信号", fontsize=15.5, y=0.975)

    # ---------- 1 池深分档 ----------
    a1 = ax[0, 0]
    # 只画 dip 池(剔除 pool_n=0 的备用池票, 它们没有"池深"这个概念)
    m = m[m.pool_n > 0].copy()
    bins = [0, 5, 10, 20, 40, 10 ** 6]
    labels = ["1-5 只", "6-10", "11-20", "21-40", "41+"]
    m["gb"] = pd.cut(m.pool_n, bins, labels=labels)
    g = m.groupby("gb", observed=True).agg(n=("ret_pct", "size"), avg=("ret_pct", "mean"),
                                           win=("ret_pct", lambda s: (s > 0).mean() * 100))
    cols = [RED if v < 0.3 else (GOLD if v < 1.1 else GREEN) for v in g["avg"]]
    a1.bar(range(len(g)), g["avg"], 0.62, color=cols)
    for i, (v, n) in enumerate(zip(g["avg"], g["n"])):
        _y = v + 0.10 if v >= 0 else v + 0.28
        a1.text(i, _y, f"{v:+.2f}%\n{n} 笔", ha="center", fontsize=9.0,
                color=DARK if v >= 0 else "white", fontweight="bold" if v < 0 else "normal")
    a1.axhline(0, color=DARK, lw=0.9)
    a1.set_xticks(range(len(g)))
    a1.set_xticklabels(g.index, fontsize=10)
    a1.set_ylabel("单笔收益 %")
    a1.set_ylim(-2.4, 2.9)
    a1.set_title(f"① {len(m)} 笔 dip 按信号日达标票数分组：池越窄，次日越差", fontsize=11.5, loc="left")
    a1b = a1.twinx()
    a1b.plot(range(len(g)), g["win"], "o--", color=BLUE, lw=1.5, ms=5)
    a1b.set_ylim(20, 75)
    a1b.set_ylabel("胜率 %", color=BLUE)
    a1b.tick_params(axis="y", colors=BLUE)
    a1b.grid(False)
    a1.text(0.02, 0.02, "窄池 → 孤立回调，没有共振可借；宽池 → 同形态票一起被错杀",
            transform=a1.transAxes, fontsize=8.4, color=GREY)

    # ---------- 2 五组方案 ----------
    a2 = ax[0, 1]
    names = ["A 现状 1.0/0.5/0.3", "B 1.0/1.0/0", "C 1.0/1.0/0.3", "D 1.0/0/0", "E 五档 41/21/11/6"]
    short = ["A 现状\n.5/.3", "B 定稿\n1/1/0", "C\n1/1/.3",
             "D\n1/0/0", "E 五档"]
    cagr = [d[n]["m"]["年化收益率%"] for n in names]
    km = [d[n]["m"]["卡玛比率"] for n in names]
    xs = np.arange(len(names))
    cols = [GREY, RED, BLUE, GOLD, GREEN]
    a2.bar(xs - 0.19, cagr, 0.38, color=cols, alpha=0.9)
    for i, v in enumerate(cagr):
        a2.text(xs[i] - 0.19, v + 1.4, f"{v:.1f}", ha="center", fontsize=9)
    a2.set_ylabel("年化 %")
    a2.set_ylim(70, 108)
    a2.set_xticks(xs)
    a2.set_xticklabels(short, fontsize=8.6)
    a2b = a2.twinx()
    a2b.plot(xs + 0.19, km, "o-", color=DARK, lw=1.6, ms=6)
    for i, v in enumerate(km):
        a2b.text(xs[i] + 0.19, v + 0.06, f"{v:.2f}", ha="center", fontsize=9, color=DARK)
    a2b.set_ylim(4.6, 6.0)
    a2b.set_ylabel("卡玛", color=DARK)
    a2b.grid(False)
    a2.set_title("② 最大回撤五组逐位相同(-17.21%)，年化的提升全落在卡玛上",
                 fontsize=11.5, loc="left")

    # ---------- 3 分半 ----------
    a3 = ax[1, 0]
    mid = pd.Timestamp("2021-12-31")

    def seg(nav, a=None, b=None):
        s = pd.Series({pd.to_datetime(x["d"]): x["nav"] for x in nav}).sort_index()
        if a is not None:
            s = s[s.index >= a]
        if b is not None:
            s = s[s.index <= b]
        r = (s.iloc[-1] / s.iloc[0]) ** (252 / len(s)) - 1
        dd = (s / s.cummax() - 1).min()
        return r * 100, (r / abs(dd) if dd else np.nan)

    vals = []
    for nm, lb in (("A 现状 1.0/0.5/0.3", "A 现状"), ("B 1.0/1.0/0", "B 定稿")):
        eq = d[nm]["equity"]
        r1, k1 = seg(eq, b=mid)
        r2, k2 = seg(eq, a=mid)
        vals.append((lb, r1, k1, r2, k2))
    xs = np.arange(4)
    heights = [vals[0][2], vals[1][2], vals[0][4], vals[1][4]]
    caps = [vals[0][1], vals[1][1], vals[0][3], vals[1][3]]
    cols3 = [GREY, RED, "#b9c0c7", "#e8a39c"]
    a3.bar(xs, heights, 0.62, color=cols3)
    for i, (h, c) in enumerate(zip(heights, caps)):
        a3.text(i, h + 0.28, f"卡玛 {h:.2f}", ha="center", fontsize=9.4,
                fontweight="bold", color=DARK)
        a3.text(i, h * 0.52, f"年化\n{c:.0f}%", ha="center", va="center",
                fontsize=8.8, color="white", fontweight="bold")
    a3.set_xticks(xs)
    a3.set_xticklabels(["前段\nA 现状", "前段\nB 定稿", "后段\nA 现状", "后段\nB 定稿"], fontsize=8.8)
    a3.set_ylabel("卡玛比率")
    a3.set_ylim(0, 13.2)
    a3.axvline(1.5, color=GREY, ls=":", lw=1.1)
    a3.text(0.75, 12.4, "2016-2021", ha="center", fontsize=10, color=GREY)
    a3.text(2.25, 12.4, "2022-2026", ha="center", fontsize=10, color=GREY)
    a3.set_title("③ 分半：两段都不吃亏，后段小跌是一笔运气", fontsize=11.5, loc="left")
    a3.text(0.0, -0.20,
            "后段 -0.70 来自 2023-04-07 龙建股份(-7.91%, 15 只池 / 评分 1.014)由半仓变满仓, 单笔运气",
            transform=a3.transAxes, ha="left", fontsize=8.0, color=GREY)

    # ---------- 4 净值 ----------
    a4 = ax[1, 1]
    for nm, lb, c, lw in (("A 现状 1.0/0.5/0.3", "A 现状 89.5% / 5.20", GREY, 1.3),
                          ("B 1.0/1.0/0", "B 定稿 96.4% / 5.60", RED, 1.9)):
        s = pd.Series({pd.to_datetime(x["d"]): x["nav"] for x in d[nm]["equity"]}).sort_index()
        a4.plot(s.index, s / s.iloc[0], color=c, lw=lw, label=lb)
    a4.set_yscale("log")
    a4.set_ylabel("净值（对数轴）")
    a4.legend(fontsize=9, loc="upper left", frameon=False)
    a4.set_title("④ 2500 日净值：同一批交易日，只改了“池窄时做不做、做多重”", fontsize=11.5, loc="left")
    a4.tick_params(axis="x", labelsize=8.5)

    fig.tight_layout(rect=[0, 0, 1, 0.955])
    out = OUT / "第253轮_池深不是仓位旋钮.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
