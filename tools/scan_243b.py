"""第243轮 B: T+1 卖是单吊的解 —— 再查阈值是否在高原上

243A 的核心发现: 同一批 192 次触发, 只把"T+2 收盘卖"改成"T+1 收盘卖",
  单笔 1.768 -> 1.652 (-0.116pp), 回撤 -45.17 -> -29.54 (+15.6pp), 卡玛 3.38 -> 5.11
多持一天几乎不赚钱, 却把尾部风险翻倍。

本脚本查三件事, 都是"S1 口径下"的:
  ① 仓位:  30/50/70/100%, 找卡玛最高点
  ② 混合:  一半 T+1 走 / 一半 T+2 走; T+1 大涨才延长
  ③ 阈值地形: 逐个移动 idx_dist 和 dist_high60 的阈值, 看单笔是否单调
     —— 单调=规律, 跳变=过拟合。240 轮的阈值是用 T+2 口径挖的, 必须重验。
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "output" / "feat234"
OUT = ROOT / "output"
COST = 0.10

COLS = ["ts_code", "trade_date", "y", "y_1d", "y_3d", "n1_ok", "n1_gap", "pct",
        "circ_mv", "idx_dist", "dist_high60", "amt_yi", "is_paused", "is_st", "listed"]


def load() -> pd.DataFrame:
    parts = []
    for p in sorted(FEAT.glob("f*.parquet")):
        d = pd.read_parquet(p, columns=COLS)
        d = d[(d.is_paused == 0) & (d.is_st == 0) & (d.listed >= 250)]
        d = d[d.pct.between(-11.5, 11.5) & d.y.between(-40, 60)]
        d = d[d.ts_code.str[:2].isin(["60", "00"])]
        d = d[d.n1_ok == True]
        if len(d):
            parts.append(d)
    return pd.concat(parts, ignore_index=True)


def M(r: pd.Series):
    nav = (1 + r / 100).cumprod()
    ann = nav.iloc[-1] ** (244 / len(r)) - 1
    dd = (nav / nav.cummax() - 1).min()
    return round(ann * 100, 2), round(dd * 100, 2), round(ann * 100 / abs(dd * 100), 2)


def main():
    d = load()
    cal = np.sort(d.trade_date.unique())
    tr = pd.read_csv(OUT / "w232_trades.csv")
    cal = cal[cal >= tr.signal.min()]
    tr = tr[tr.signal.isin(cal)]
    busy = set()
    for _, r in tr.iterrows():
        busy.add(r.buy)
        busy.add(r.sell)
    sig = set(tr.signal)
    free = set(x for x in cal if x not in busy and x not in sig)
    base = pd.Series(0.0, index=cal)
    for dt, g in tr.groupby("signal"):
        base[dt] = g["ret"].mean()
    b = M(base)
    nxt = {cal[i]: cal[i + 1] for i in range(len(cal) - 1)}
    nxt[cal[-1]] = cal[-1]

    def pick_days(lo_idx: float, lo_high: float):
        s = d[(d.idx_dist <= lo_idx) & (d.dist_high60 <= lo_high)]
        days = []
        for dt in sorted(s.trade_date.unique()):
            if dt not in free:
                continue
            bd = nxt.get(dt)
            sd = nxt.get(bd, bd)
            if bd in busy or sd in busy:
                continue
            days.append(dt)
        pick = {}
        for dt, g in d[d.trade_date.isin(days)].groupby("trade_date"):
            g2 = g.dropna(subset=["dist_high60"]).sort_values(
                ["dist_high60", "amt_yi"], ascending=[True, False])
            if len(g2):
                pick[dt] = g2.iloc[0]
        P = pd.DataFrame(pick.values())
        P["signal"] = list(pick.keys())
        return P

    P = pick_days(-3.51, -24.46)
    assert len(P) == 192, len(P)
    y1 = P.y_1d.values
    y2 = P.y.values
    idx = P.signal.values

    print(f"基准 年化{b[0]}% 回撤{b[1]}% 卡玛{b[2]}   补位 {len(P)} 次\n")

    print("=" * 96)
    print("① 仓位 (S1 = T+1 收盘卖)")
    rows = []
    for w in (0.2, 0.3, 0.5, 0.7, 0.85, 1.0):
        v = pd.Series(y1 - COST, index=idx)
        sim = base.copy()
        sim[v.index] += v.values * w
        a, dd, cm = M(sim)
        rows.append({"仓位": f"{w:.0%}", "单笔": round(v.mean(), 3), "年化": a,
                     "回撤": dd, "卡玛": cm, "Δ卡玛": round(cm - b[2], 2)})
    for w in (1.0,):
        v = pd.Series(y2 - COST, index=idx)
        sim = base.copy()
        sim[v.index] += v.values * w
        a, dd, cm = M(sim)
        rows.append({"仓位": f"{w:.0%}(T+2对照)", "单笔": round(v.mean(), 3), "年化": a,
                     "回撤": dd, "卡玛": cm, "Δ卡玛": round(cm - b[2], 2)})
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n" + "=" * 96)
    print("② 混合卖法 (满仓)")
    rows = []
    cands = {
        "全 T+1": y1,
        "全 T+2": y2,
        "T+1 半仓 + T+2 半仓": 0.5 * y1 + 0.5 * y2,
        "T+1 七成 + T+2 三成": 0.7 * y1 + 0.3 * y2,
        "T+1 涨>5% 才延到 T+2": np.where(y1 > 5, y2, y1),
        "T+1 涨>8% 才延到 T+2": np.where(y1 > 8, y2, y1),
        "T+1 涨>3% 才延到 T+2": np.where(y1 > 3, y2, y1),
    }
    for k, arr in cands.items():
        v = pd.Series(np.asarray(arr) - COST, index=idx)
        sim = base.copy()
        sim[v.index] += v.values
        a, dd, cm = M(sim)
        rows.append({"卖法": k, "单笔": round(v.mean(), 3), "胜率%": round((v > 0).mean() * 100, 1),
                     "最差": round(v.min(), 1), "年化": a, "回撤": dd, "卡玛": cm,
                     "Δ卡玛": round(cm - b[2], 2)})
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n" + "=" * 96)
    print("③ 阈值地形 (满仓, S1 口径) —— 看单笔是否随阈值放宽而单调")
    print("\n  移动 dist_high60 (固定 idx_dist<=-3.51):")
    rows = []
    for hi in (-18, -20, -22, -24.46, -27, -30, -35, -45):
        Q = pick_days(-3.51, hi)
        v = pd.Series(Q.y_1d.values - COST, index=Q.signal.values)
        sim = base.copy()
        sim[v.index] += v.values
        a, dd, cm = M(sim)
        rows.append({"dist_high60<=": hi, "次数": len(v), "单笔": round(v.mean(), 3),
                     "胜率%": round((v > 0).mean() * 100, 1), "年化": a, "回撤": dd,
                     "卡玛": cm, "Δ卡玛": round(cm - b[2], 2)})
    t3 = pd.DataFrame(rows)
    print(t3.to_string(index=False))

    print("\n  移动 idx_dist (固定 dist_high60<=-24.46):")
    rows = []
    for lo in (-2.0, -2.5, -3.0, -3.51, -4.0, -5.0, -6.0, -8.0):
        Q = pick_days(lo, -24.46)
        if len(Q) < 20:
            rows.append({"idx_dist<=": lo, "次数": len(Q), "单笔": np.nan, "胜率%": np.nan,
                         "年化": np.nan, "回撤": np.nan, "卡玛": np.nan, "Δ卡玛": np.nan})
            continue
        v = pd.Series(Q.y_1d.values - COST, index=Q.signal.values)
        sim = base.copy()
        sim[v.index] += v.values
        a, dd, cm = M(sim)
        rows.append({"idx_dist<=": lo, "次数": len(v), "单笔": round(v.mean(), 3),
                     "胜率%": round((v > 0).mean() * 100, 1), "年化": a, "回撤": dd,
                     "卡玛": cm, "Δ卡玛": round(cm - b[2], 2)})
    t4 = pd.DataFrame(rows)
    print(t4.to_string(index=False))
    t3.to_csv(OUT / "w243b_grid_high.csv", index=False, encoding="utf-8-sig")
    t4.to_csv(OUT / "w243b_grid_idx.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 96)
    print("④ 分年 (满仓, S1 vs S2)")
    t = pd.DataFrame({"yr": idx // 10000, "T+1": y1 - COST, "T+2": y2 - COST})
    g = t.groupby("yr").agg(n=("T+1", "size"), s1=("T+1", "mean"), s2=("T+2", "mean"))
    g["T+1合计"] = t.groupby("yr")["T+1"].sum().round(1)
    g["T+2合计"] = t.groupby("yr")["T+2"].sum().round(1)
    print(g.round(2).to_string())
    print(f"\nT+1: 正年 {(g.s1 > 0).sum()}/{len(g)}   T+2: 正年 {(g.s2 > 0).sum()}/{len(g)}")
    print(f"T+1: 合计 {t['T+1'].sum():+.0f}pp   T+2: 合计 {t['T+2'].sum():+.0f}pp")


if __name__ == "__main__":
    main()
