"""第 242 轮配图: 单吊（每天一只、满仓）在补位上的真实上限

① 单吊满仓逐年: 11 年里 2 年亏
② 仓位是唯一有效旋钮: 年化一路升, 回撤一路炸
③ 入场缺口过滤: 好结果只活在一个格子里, 邻近参数就崩
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

COLS = ["ts_code", "trade_date", "y", "n1_ok", "n1_gap", "pct", "circ_mv", "idx_dist",
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


def met(r: pd.Series) -> tuple[float, float, float]:
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
    b = met(base)
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

    def pick(lo=-99.0, hi=99.0, w=1.0):
        v = {}
        for dt, g in sub.items():
            g2 = g.dropna(subset=["n1_gap", "dist_high60"])
            g2 = g2[(g2["n1_gap"] >= lo) & (g2["n1_gap"] <= hi)]
            if not len(g2):
                continue
            g2 = g2.sort_values(["dist_high60", "amt_yi"], ascending=[True, False])
            v[dt] = g2["y"].iloc[0] - COST
        v = pd.Series(v)
        sim = base.copy()
        sim[v.index] += v.values * w
        return v, sim, met(sim)

    v_all, sim_all, m_all = pick()
    fig = plt.figure(figsize=(16.4, 5.7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.0, 1.15], wspace=0.32)

    # ---------- ① 分年 ----------
    a = fig.add_subplot(gs[0, 0])
    p = v_all.reset_index()
    p.columns = ["signal", "y"]
    p["yr"] = p["signal"] // 10000
    g = p.groupby("yr")["y"].agg(["size", "sum"])
    cols = ["#c0392b" if x < 0 else "#27ae60" for x in g["sum"]]
    a.bar(range(len(g)), g["sum"], color=cols, width=0.66)
    a.axhline(0, color="#333", lw=0.9)
    for i, (n, sm) in enumerate(zip(g["size"], g["sum"])):
        a.text(i, sm + (3 if sm >= 0 else -3), f"{sm:+.0f}", ha="center",
               va="bottom" if sm >= 0 else "top", fontsize=8.3,
               color="#1e8449" if sm >= 0 else "#c0392b")
        a.text(i, -78, f"{n}次", ha="center", fontsize=7.4, color="#888")
    a.set_xticks(range(len(g)))
    a.set_xticklabels([str(x) for x in g.index], fontsize=8.4, rotation=0)
    a.set_ylim(-86, 108)
    a.set_ylabel("这一年的补位合计收益（pp）", fontsize=9.5)
    a.set_title(f"① 单吊满仓 11 年：{int((g['sum'] > 0).sum())} 正 {int((g['sum'] <= 0).sum())} 负",
                fontsize=11.5, pad=9)
    a.grid(axis="y", ls=":", color="#ccc", alpha=0.6)
    a.set_axisbelow(True)

    # ---------- ② 仓位 ----------
    bb = fig.add_subplot(gs[0, 1])
    ws = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0]
    anns, dds, cms = [], [], []
    for w in ws:
        _, _, m = pick(w=w)
        anns.append(m[0])
        dds.append(-m[1])
        cms.append(m[2])
    xs = [w * 100 for w in ws]
    bb.plot(xs, anns, "o-", color="#c0392b", lw=2.0, ms=5.5, label="年化 %（左轴）")
    for x, y in zip(xs, anns):
        bb.text(x, y + 5, f"{y:.0f}%", ha="center", fontsize=8.2, color="#c0392b",
                path_effects=[pe.withStroke(linewidth=2.4, foreground="white")])
    bb.axhline(b[0], color="#c0392b", ls=":", lw=1.3, alpha=0.7)
    bb.text(100, b[0] - 5.5, f"基准 {b[0]:.0f}%", fontsize=8, color="#c0392b", ha="right",
            path_effects=[pe.withStroke(linewidth=2.4, foreground="white")])
    bb.set_xlabel("补位仓位（%，单吊一只）", fontsize=9.5)
    bb.set_ylabel("年化 %", fontsize=9.5, color="#c0392b")
    bb.tick_params(axis="y", colors="#c0392b", labelsize=8.5)
    bb.set_ylim(85, 178)
    bb.set_title("② 仓位：年化一路升，卡玛一路降", fontsize=11.5, pad=9)
    bb.grid(ls=":", color="#ccc", alpha=0.55)
    bb.set_axisbelow(True)

    b3 = bb.twinx()
    b3.plot(xs, cms, "s--", color="#2980b9", lw=1.9, ms=5, label="卡玛（右轴）")
    for x, y in zip(xs, cms):
        b3.text(x, y - 0.16, f"{y:.2f}", ha="center", fontsize=8.2, color="#2980b9",
                path_effects=[pe.withStroke(linewidth=2.4, foreground="white")])
    b3.axhline(b[2], color="#2980b9", ls=":", lw=1.3, alpha=0.7)
    b3.text(101, b[2] + 0.06, f"基准 {b[2]:.2f}", fontsize=8, color="#2980b9", ha="right",
            path_effects=[pe.withStroke(linewidth=2.4, foreground="white")])
    b3.set_ylabel("卡玛比率", fontsize=9.5, color="#2980b9")
    b3.tick_params(axis="y", colors="#2980b9", labelsize=8.5)
    b3.set_ylim(2.9, 5.1)
    h1, l1 = bb.get_legend_handles_labels()
    h2, l2 = b3.get_legend_handles_labels()
    bb.legend(h1 + h2, l1 + l2, fontsize=8.2, loc="upper left", framealpha=0.92)
    bb.text(50, 88, "回撤随之从 −23.6% 炸到 −45.2%", fontsize=8.2, color="#7f8c8d",
            ha="center", path_effects=[pe.withStroke(linewidth=2.4, foreground="white")])

    # ---------- ③ gap 热力图 ----------
    c = fig.add_subplot(gs[0, 2])
    los = [-12, -10, -9, -8, -7, -6, -5]
    his = [-4, -3, -2]
    Z = np.zeros((len(los), len(his)))
    for i, lo in enumerate(los):
        for j, hi in enumerate(his):
            _, _, m = pick(lo, hi)
            Z[i, j] = m[2]
    im = c.imshow(Z, cmap="RdYlGn", vmin=2.2, vmax=5.6, aspect="auto")
    for i in range(len(los)):
        for j in range(len(his)):
            c.text(j, i, f"{Z[i, j]:.2f}", ha="center", va="center", fontsize=9.2,
                   color="#222", fontweight="bold" if Z[i, j] >= 5.3 else "normal")
    c.set_xticks(range(len(his)))
    c.set_xticklabels([f"≤{x}%" for x in his], fontsize=9)
    c.set_yticks(range(len(los)))
    c.set_yticklabels([f"{x}%" for x in los], fontsize=9)
    c.set_xlabel("竞价缺口上限", fontsize=9.5)
    c.set_ylabel("竞价缺口下限", fontsize=9.5)
    c.set_title("③ 入场缺口过滤：最高那一格是孤岛，不是规律", fontsize=11.5, pad=9)
    i0, j0 = np.unravel_index(Z.argmax(), Z.shape)
    c.add_patch(plt.Rectangle((j0 - 0.5, i0 - 0.5), 1, 1, fill=False, ec="#c0392b", lw=2.4))
    nb = [Z[i0, j] for j in range(len(his)) if j != j0] + \
         [Z[i, j0] for i in range(len(los)) if i != i0]
    c.text(2.62, 0.4, f"最高 {Z.max():.2f}\n相邻最低\n{min(nb):.2f}", fontsize=8.0, va="center",
           ha="left", color="#c0392b", linespacing=1.7,
           path_effects=[pe.withStroke(linewidth=2.6, foreground="white")])
    fig.colorbar(im, ax=c, fraction=0.045, pad=0.03).set_label("卡玛", fontsize=8.5)

    fig.text(0.5, 0.005,
             "补位规则：等权指数距 20 日线 ≤ −3.51% 且 个股距前 60 日高 ≤ −24.46%，T+1 开盘买单吊一只（距 60 日高最远），T+2 收盘卖，"
             "只用主策略空仓且资金空闲的日子　·　扣 0.10% 成本　·　仅主板（60/00）",
             ha="center", fontsize=8.1, color="#666")

    out = ROOT / "第242轮_单吊的代价与上限.png"
    fig.savefig(out, dpi=155, bbox_inches="tight", facecolor="white")
    print("saved", out)
    print("基准", tuple(round(x, 2) for x in b), " 单吊满仓", tuple(round(x, 2) for x in m_all))
    print(f"单吊满仓 {len(v_all)} 次 单笔{v_all.mean():+.3f}% 胜率{(v_all > 0).mean() * 100:.1f}%")


if __name__ == "__main__":
    main()
