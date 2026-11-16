"""第 241 轮配图: 补位是 beta 规则 —— 买几只决定成败

① 5 种买法（满仓）的卡玛与回撤: 越分散, 卡玛越高且回撤越低
② 净值: 基准 / 最小1只满仓 / 市值分层40只满仓
③ 42 波补位日收益: 收益集中在少数几波
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "output" / "feat234"
OUT = ROOT / "output"
COST = 0.10
plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

COLS = ["ts_code", "trade_date", "y", "n1_ok", "pct", "circ_mv", "idx_dist",
        "dist_high60", "amt_yi", "is_paused", "is_st", "listed"]


def load() -> pd.DataFrame:
    parts = []
    for p in sorted(FEAT.glob("f*.parquet")):
        d = pd.read_parquet(p, columns=COLS)
        d = d[(d["is_paused"] == 0) & (d["is_st"] == 0) & (d["n1_ok"] == True)]      # noqa: E712
        d = d[d["pct"].between(-11.5, 11.5) & d["y"].between(-40, 60) & (d["listed"] >= 250)]
        d = d[d["ts_code"].str[:2].isin(["60", "00"])]
        if len(d):
            parts.append(d)
    return pd.concat(parts, ignore_index=True)


def metrics(r: pd.Series) -> tuple[float, float, float]:
    nav = (1 + r / 100).cumprod()
    ann = nav.iloc[-1] ** (244 / len(r)) - 1
    dd = (nav / nav.cummax() - 1).min()
    return ann * 100, dd * 100, ann * 100 / abs(dd * 100)


def main() -> None:
    d = load()
    cal = np.sort(d["trade_date"].unique())
    tr = pd.read_csv(OUT / "w232_trades.csv")
    cal = cal[cal >= tr["signal"].min()]
    tr = tr[tr["signal"].isin(cal)]
    busy = set()
    for _, r in tr.iterrows():
        busy.add(r["buy"])
        busy.add(r["sell"])
    sig = set(tr["signal"])
    free = set(x for x in cal if x not in busy and x not in sig)
    base = pd.Series(0.0, index=cal)
    for dt, g in tr.groupby("signal"):
        base[dt] = g["ret"].mean()
    nxt = {cal[i]: cal[i + 1] for i in range(len(cal) - 1)}
    nxt[cal[-1]] = cal[-1]

    s = d[(d["idx_dist"] <= -3.51) & (d["dist_high60"] <= -24.46)]
    days = []
    for dt in sorted(s["trade_date"].unique()):
        if dt not in free:
            continue
        bd = nxt.get(dt)
        sd = nxt.get(bd, bd)
        if bd in busy or sd in busy:
            continue
        days.append(dt)
    sub = {dt: g for dt, g in d[d["trade_date"].isin(days)].groupby("trade_date")}

    def strat(n_grp: int, per: int = 1) -> dict:
        out = {}
        for dt, g in sub.items():
            g2 = g.dropna(subset=["circ_mv"]).sort_values("circ_mv")
            if len(g2) < n_grp * per:
                out[dt] = g2["y"].mean() - COST
                continue
            sel = []
            for _, gg in g2.groupby(pd.qcut(g2["circ_mv"].rank(method="first"), n_grp, labels=False)):
                sel.append(gg.nlargest(per, "amt_yi")["y"].values)
            out[dt] = np.concatenate(sel).mean() - COST
        return out

    V = {
        "最小市值1只": {dt: g.nsmallest(1, "circ_mv")["y"].mean() - COST for dt, g in sub.items()},
        "成交额前10只": {dt: g.nlargest(min(10, len(g)), "amt_yi")["y"].mean() - COST for dt, g in sub.items()},
        "分层20只": strat(20, 1),
        "分层40只": strat(20, 2),
        "全市场等权": {dt: g["y"].mean() - COST for dt, g in sub.items()},
    }
    b = metrics(base)

    fig = plt.figure(figsize=(16.4, 5.7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.0, 1.0], wspace=0.34)

    # ---------- ① 卡玛 + 回撤 ----------
    a = fig.add_subplot(gs[0, 0])
    names = list(V)
    cms, dds, anns = [], [], []
    for k in names:
        v = pd.Series(V[k])
        sim = base.copy()
        sim[v.index] += v.values
        ann, dd, cm = metrics(sim)
        anns.append(ann)
        dds.append(-dd)
        cms.append(cm)
    x = np.arange(len(names))
    a.bar(x, cms, 0.58, color=["#c0392b", "#e67e22", "#2980b9", "#8e44ad", "#27ae60"])
    a.axhline(b[2], color="#2c3e50", ls="--", lw=1.7, label=f"基准 {b[2]:.2f}")
    for xx, (cm, an) in enumerate(zip(cms, anns)):
        a.text(xx, cm + 0.08, f"{cm:.2f}", ha="center", fontsize=9.4, fontweight="bold", color="#333")
        a.text(xx, 0.36, f"年化\n{an:.0f}%", ha="center", fontsize=7.8, color="white",
               fontweight="bold", linespacing=1.5)
    a.set_xticks(x)
    a.set_xticklabels(["最小市值\n1只", "成交额前\n10只", "市值分层\n20只", "市值分层\n40只", "全市场\n等权"], fontsize=8.8)
    a.set_ylim(0, 5.9)
    a.set_ylabel("卡玛比率（满仓）", fontsize=10)
    a.set_title("① 破位日买得越散，卡玛越高、回撤反而越小", fontsize=11.5, pad=9)
    a.legend(fontsize=8.6, loc="upper left", framealpha=0.92)
    a.grid(axis="y", ls=":", color="#ccc", alpha=0.6)
    a.set_axisbelow(True)

    a2 = a.twinx()
    a2.plot(x, dds, "o--", color="#34495e", lw=1.6, ms=5.5)
    for xx, dd in zip(x, dds):
        a2.text(xx, dd - 1.9, f"{dd:.0f}%", ha="center", fontsize=8.4, color="#34495e",
                path_effects=[pe.withStroke(linewidth=2.6, foreground="white")])
    a2.set_ylim(10, 52)
    a2.set_ylabel("最大回撤 %（虚线，越低越好）", fontsize=9, color="#34495e")
    a2.tick_params(axis="y", colors="#34495e", labelsize=8.5)

    # ---------- ② 净值 ----------
    bb = fig.add_subplot(gs[0, 1])
    v1 = pd.Series(V["最小市值1只"])
    sim1 = base.copy()
    sim1[v1.index] += v1.values
    v4 = pd.Series(V["分层40只"])
    sim4 = base.copy()
    sim4[v4.index] += v4.values
    for lab, ser, c, ls, lw in (("基准（主策略）", base, "#2c3e50", "--", 1.4),
                                ("＋最小市值1只 满仓", sim1, "#e67e22", "-", 1.5),
                                ("＋市值分层40只 满仓", sim4, "#8e44ad", "-", 1.9)):
        nav = (1 + ser / 100).cumprod().values
        bb.plot(np.arange(len(cal)), nav, color=c, ls=ls, lw=lw, label=lab)
    bb.set_yscale("log")
    bb.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}x"))
    bb.set_title("② 十年净值：补位只加在主策略空仓日", fontsize=11.5, pad=9)
    bb.set_ylabel("净值（对数轴）", fontsize=9.5)
    bb.set_xlabel("交易日序号（2016-08 → 2026-11）", fontsize=9)
    bb.legend(fontsize=8.2, loc="upper left", framealpha=0.92)
    bb.grid(ls=":", color="#ccc", alpha=0.5)
    bb.set_axisbelow(True)
    m4 = metrics(sim4)
    bb.text(0.97, 0.03,
            f"基准 {b[0]:.0f}% / {b[1]:.0f}% / {b[2]:.2f}\n分层40只 {m4[0]:.0f}% / {m4[1]:.0f}% / {m4[2]:.2f}",
            transform=bb.transAxes, fontsize=8.2, ha="right", va="bottom", color="#555",
            linespacing=1.6, bbox=dict(fc="white", ec="#ddd", alpha=0.92, pad=3))

    # ---------- ③ 42 波 ----------
    c = fig.add_subplot(gs[0, 2])
    v = pd.Series(V["分层40只"])
    dt_ = pd.to_datetime(pd.Series(v.index).astype(str))
    wave = (dt_.diff().dt.days > 10).cumsum().values
    wv = pd.DataFrame({"w": wave, "y": v.values}).groupby("w")["y"].sum().sort_values()
    cols = ["#c0392b" if x < 0 else "#27ae60" for x in wv.values]
    c.bar(range(len(wv)), wv.values, color=cols, width=0.82)
    c.axhline(0, color="#333", lw=0.9)
    c.set_title(f"③ 42 波下跌：最差 3 波 {wv.head(3).sum():+.0f}pp，最好 3 波 {wv.tail(3).sum():+.0f}pp",
                fontsize=11.5, pad=9)
    c.set_xlabel("每一波（间隔>10 天算新一波），按收益排序", fontsize=9)
    c.set_ylabel("这波补位合计收益（pp）", fontsize=9)
    c.grid(axis="y", ls=":", color="#ccc", alpha=0.6)
    c.set_axisbelow(True)
    run = mx = 0
    for yy in v.values:
        run = run + 1 if yy < 0 else 0
        mx = max(mx, run)
    c.annotate(f"最长连亏 {mx} 次\n单次最差 {v.min():.1f}%",
               (1.0, wv.values[0]), xytext=(36, 96), textcoords="offset points",
               fontsize=8.2, color="#c0392b", linespacing=1.6,
               arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.1,
                               connectionstyle="arc3,rad=-0.2"))

    fig.text(0.5, 0.005,
             "补位规则：等权指数距 20 日线 ≤ −3.51% 且 个股距前 60 日高 ≤ −24.46%，T+1 开盘买、T+2 收盘卖，只用主策略空仓且资金空闲的日子"
             "　·　扣 0.10% 成本　·　仅主板（60/00）　·　分层 = 按流通市值 20 等份，每份取成交额最大的 1~2 只",
             ha="center", fontsize=8.1, color="#666")

    out = ROOT / "第241轮_补位是beta所以怎么买.png"
    fig.savefig(out, dpi=155, bbox_inches="tight", facecolor="white")
    print("saved", out)
    print("基准", tuple(round(x, 2) for x in b))
    for k in names:
        v = pd.Series(V[k])
        sim = base.copy()
        sim[v.index] += v.values
        print(f"  {k:<8} {tuple(round(x, 2) for x in metrics(sim))}")


if __name__ == "__main__":
    main()
