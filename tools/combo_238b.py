"""第二百三十八轮 B: 补位方案的分年稳定性、组合、市场环境依赖

238 主表显示 C3/C4 补位 32 笔把年化从 86% 抬到 95~103% 且回撤不变,
C1 补位 393 笔年化 116~127% 但回撤从 -24% 炸到 -35~-48%。
本轮查清楚: ① 这些补位赚的钱集中在哪一年 ② 能不能组合 ③ 回撤恶化是否来自
"在大盘下跌日补位"(即推翻了风控)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "output" / "feat234"
OUT = ROOT / "output"
TRADES = OUT / "w232_trades.csv"
MKT = FEAT / "market.csv"

WINDOW = 2500
COST = 0.10
COLS = ["ts_code", "trade_date", "y", "n1_ok", "pct", "rel_pct", "bias10", "bias20",
        "ret10", "maxdd20", "vol20", "is_paused", "is_st", "listed", "circ_mv"]

CONDS = {
    "C1": lambda d: (d.bias10 > 3.148) & (d.bias20 < -4.582),
    "C2": lambda d: (d.bias10 > 3.148) & (d.ret10 < -5.874),
    "C3": lambda d: (d.pct < -1.816) & (d.rel_pct > 1.287),
    "C4": lambda d: (d.rel_pct > 1.287) & ((d.maxdd20 - d.pct) > 1.707),
    "C5": lambda d: (d.rel_pct > 1.287) & ((d.vol20 * d.pct) < -4.439),
    "C6": lambda d: (d.bias20 < -4.582) & d.ret10.between(-5.874, -1.72),
}
PRIORITY = ["C3", "C4", "C5", "C1", "C2", "C6"]


def load_all() -> pd.DataFrame:
    parts = []
    for p in sorted(FEAT.glob("f*.parquet")):
        d = pd.read_parquet(p, columns=COLS)
        d = d[(d["is_paused"] == 0) & (d["is_st"] == 0) & (d["n1_ok"] == True)]   # noqa: E712
        d = d[d["pct"].between(-11.5, 11.5) & d["y"].between(-40, 60) & (d["listed"] >= 250)]
        if len(d):
            parts.append(d)
    return pd.concat(parts, ignore_index=True)


def metrics(day_ret: pd.Series) -> tuple[float, float, float]:
    nav = (1 + day_ret / 100).cumprod()
    yrs = len(day_ret) / 244.0
    ann = nav.iloc[-1] ** (1 / yrs) - 1
    dd = (nav / nav.cummax() - 1).min()
    return ann * 100, dd * 100, (ann * 100 / abs(dd * 100) if dd else np.nan)


def build_picks(d, cal, busy, nxt, cmask, mode="mv"):
    """返回 date -> 补位单笔收益(%)"""
    out = {}
    for cname in [c for c in PRIORITY if c in cmask]:
        s = d[cmask[cname]]
        if not len(s):
            continue
        for dt in s["trade_date"].unique():
            if dt in out or dt in busy:
                continue
            b, sl = nxt[dt], nxt[nxt[dt]]
            if b in busy or sl in busy:
                continue
            g = s[s["trade_date"] == dt]
            if mode == "mv":
                v = g.loc[g["circ_mv"].idxmin(), "y"]
            else:
                v = g["y"].mean()
            out[dt] = v - COST
    return out


def main() -> None:
    d = load_all()
    cal = np.sort(d["trade_date"].unique())[-WINDOW:]
    nxt = {cal[i]: cal[i + 1] for i in range(len(cal) - 1)}
    nxt[np.int64(cal[-1])] = np.int64(cal[-1])
    d = d[d["trade_date"].isin(cal)].copy()

    tr = pd.read_csv(TRADES)
    tr = tr[tr["signal"].isin(cal)]
    base = pd.Series(0.0, index=cal)
    for dt, g in tr.groupby("signal"):
        base[dt] = g["ret"].mean()
    busy = set()
    for _, r in tr.iterrows():
        busy.add(r["buy"]); busy.add(r["sell"])

    cmask = {}
    for c, fn in CONDS.items():
        m = fn(d)
        # 条件内再要求"次日能买"已由 n1_ok 保证
        cmask[c] = m

    # ① 单条件分年贡献
    print("=" * 76)
    print("① 单条件补位: 分年收益差(补位日 vs 0)")
    rows = []
    for c in PRIORITY:
        picks = build_picks(d, cal, busy, nxt, {c: cmask[c]}, "mv")
        if not picks:
            continue
        s = pd.Series(picks)
        yr = (s.index // 10000).astype(int)
        g = s.groupby(yr).agg(["size", "sum"])
        rows.append({"条件": c, **{str(k): f"{int(v['size'])}笔/{v['sum']:+.0f}pp"
                                   for k, v in g.iterrows()}})
    print(pd.DataFrame(rows).fillna("-").to_string(index=False))

    # ② 组合: 优先级去重
    for name, cs in [("C3+C4+C5", ["C3", "C4", "C5"]),
                     ("C3+C4+C5+C1", ["C3", "C4", "C5", "C1"]),
                     ("全 6 个", PRIORITY)]:
        picks = build_picks(d, cal, busy, nxt, cmask, "mv")
        # build_picks 已按 PRIORITY 顺序, 但不受 cs 限制 -> 重做
        sub = {c: cmask[c] for c in cs}
        old = PRIORITY
        PRIORITY[:] = [c for c in old if c in cs]
        picks = build_picks(d, cal, busy, nxt, sub, "mv")
        PRIORITY[:] = old
        sim = base.copy()
        s = pd.Series(picks)
        sim[s.index] = s.values
        ann, dd, cml = metrics(sim)
        print(f"\n{name}: 补位 {len(s)} 笔 单笔 {s.mean():+.3f}%  "
              f"年化 {ann:.2f}%  回撤 {dd:.2f}%  卡玛 {cml:.2f}  "
              f"(基准 {metrics(base)[0]:.2f}/{metrics(base)[1]:.2f}/{metrics(base)[2]:.2f})")

    # ③ 补位日的市场环境
    mk = pd.read_csv(MKT).rename(columns={"trade_date": "date"}).set_index("date")
    mk["ret"] = mk["med_pct"]
    mk["above"] = (mk["idx_ma20_dist"] > 0).astype(float)
    print("\n" + "=" * 76)
    print("③ 补位日的大盘状态(补位日 vs 全部交易日)")
    allp = mk.reindex(cal)
    for c in ["C1", "C3", "C5", "C6"]:
        picks = build_picks(d, cal, busy, nxt, {c: cmask[c]}, "mv")
        if not picks:
            continue
        idx = pd.Index(list(picks.keys()))
        z = mk.reindex(idx)
        print(f"{c}: {len(idx)} 日  20日线上占比 {z['above'].mean()*100:.0f}% "
              f"(全期 {allp['above'].mean()*100:.0f}%)  "
              f"当日大盘中位 {z['ret'].median():+.2f}% (全期 {allp['ret'].median():+.2f}%)")

    # ④ 加一层市场过滤: 只在指数 20 日线上补位
    print("\n" + "=" * 76)
    print("④ 补位 + 市场过滤(指数 20 日线上才补)")
    for c in ["C1", "C3", "C5"]:
        picks = build_picks(d, cal, busy, nxt, {c: cmask[c]}, "mv")
        idx = pd.Index(list(picks.keys()))
        keep = [x for x in idx if allp["above"].get(x, 0) == 1]
        p2 = {k: v for k, v in picks.items() if k in set(keep)}
        if not p2:
            continue
        sim = base.copy()
        s = pd.Series(p2)
        sim[s.index] = s.values
        ann, dd, cml = metrics(sim)
        print(f"{c}+过滤: {len(s)} 笔 单笔 {s.mean():+.3f}%  年化 {ann:.2f}%  "
              f"回撤 {dd:.2f}%  卡玛 {cml:.2f}")

    # ⑤ bootstrap: 32 笔到底靠不靠谱
    print("\n" + "=" * 76)
    print("⑤ C3/C4 补位的 bootstrap(有放回重抽 5000 次, 看年化区间)")
    for c in ["C3", "C4"]:
        picks = build_picks(d, cal, busy, nxt, {c: cmask[c]}, "mv")
        v = np.array(list(picks.values()))
        if len(v) < 5:
            continue
        rng = np.random.default_rng(238)
        anns = []
        for _ in range(3000):
            idx = rng.integers(0, len(v), len(v))
            sim = base.copy()
            sim[pd.Index(list(picks.keys()))] = v[idx]
            anns.append(metrics(sim)[0])
        anns = np.array(anns)
        print(f"{c}: 年化 中位 {np.median(anns):.1f}%  5% 分位 {np.percentile(anns,5):.1f}%  "
              f"95% 分位 {np.percentile(anns,95):.1f}%  "
              f"跑输基准 {np.mean(anns < metrics(base)[0])*100:.0f}% 概率")


if __name__ == "__main__":
    main()
