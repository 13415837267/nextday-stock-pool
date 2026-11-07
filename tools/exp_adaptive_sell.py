"""第 168 轮(三): T+2 日内 alpha 逐年衰减(单调 6 年) —— 尾盘卖这条纪律还剩多少?

反事实: "最近 N 笔 seg3 < 阈值 -> 改开盘卖", 逐笔可执行(T+2 开盘时前 N 笔均已了结)。
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "output")


def load():
    tr = pd.DataFrame(pd.read_pickle(os.path.join(OUT, "main_trades_latest.pkl")))
    tr = tr[tr["ret_pct"].notna()].copy()
    tr["d"] = tr["signal_date"].astype(str).str.replace("-", "").str[:8]
    tr = tr.sort_values("d").reset_index(drop=True)

    p = pd.read_pickle(os.path.join(ROOT, "data", "cache", "panel_all.pkl"))
    p = p.loc[p["is_paused"] == 0, ["ts_code", "d", "open", "close", "factor"]].copy()
    p["d"] = p["d"].astype(str).str.replace("-", "").str[:8]
    p["px_o"] = (p["open"].astype("float64") * p["factor"]).astype("float32")
    p["px_c"] = (p["close"].astype("float64") * p["factor"]).astype("float32")
    mo = p.pivot_table(index="d", columns="ts_code", values="px_o", aggfunc="first")
    mc = p.pivot_table(index="d", columns="ts_code", values="px_c", aggfunc="first")
    days = list(mo.index)
    pos = {d: i for i, d in enumerate(days)}
    cols = {c: j for j, c in enumerate(mo.columns)}

    s1 = np.full(len(tr), np.nan)
    s2 = np.full(len(tr), np.nan)
    s3 = np.full(len(tr), np.nan)
    for k, (c, d) in enumerate(zip(tr["code"].astype(str), tr["d"])):
        i, j = pos.get(d), cols.get(c)
        if i is None or j is None or i + 2 >= len(days):
            continue
        o1, c1, o2, c2 = mo.iat[i + 1, j], mc.iat[i + 1, j], mo.iat[i + 2, j], mc.iat[i + 2, j]
        if o1 > 0 and c1 > 0 and o2 > 0 and c2 > 0:
            s1[k], s2[k], s3[k] = c1 / o1 - 1, o2 / c1 - 1, c2 / o2 - 1
    tr["r_close"] = (1 + s1) * (1 + s2) * (1 + s3) - 1
    tr["r_open"] = (1 + s1) * (1 + s2) - 1
    tr["s1"], tr["s2"], tr["s3"] = s1 * 100, s2 * 100, s3 * 100
    tr["y"] = tr["d"].str[:4]
    return tr.dropna(subset=["s3"]).reset_index(drop=True)


def compound(r):
    return float(np.exp(np.log1p(r).sum()) - 1)


def main():
    a = load()
    print(f"样本 {len(a)} 笔  {a.d.min()} ~ {a.d.max()}")
    rc, ro = a["r_close"].to_numpy(), a["r_open"].to_numpy()
    print(f"全期: 收盘 {compound(rc)*100:+.1f}%  开盘 {compound(ro)*100:+.1f}%")
    for lo, hi, tag in [("2020", "2024", "2020-2024"), ("2025", "2026", "2025-2026")]:
        m = (a.y >= lo) & (a.y <= hi)
        print(f"  {tag}: n={m.sum()} 收盘 {compound(rc[m])*100:+.1f}%  开盘 {compound(ro[m])*100:+.1f}%")

    print("\n=== 自适应反事实: 前 N 笔 seg3 均值 < 阈值 -> 本笔开盘卖 ===")
    s3 = a["s3"].to_numpy()
    best = None
    for N in (20, 30, 40, 60, 80):
        for th in (-0.2, 0.0, 0.2, 0.4):
            pick = np.zeros(len(a), dtype=bool)
            for k in range(N, len(a)):
                if np.nanmean(s3[k - N:k]) < th / 100:
                    pick[k] = True
            r = np.where(pick, ro, rc)
            tot = compound(r) * 100
            # 分年
            yr = {}
            for y, g in a.groupby("y"):
                m = (a.y == y).to_numpy()
                yr[y] = round(compound(r[m]) * 100, 1)
            mark = ""
            if best is None or tot > best[0]:
                best = (tot, N, th, yr)
                mark = " <<"
            print(f"  N={N:>2} th={th:+.1f}%  开盘卖占比 {pick.mean()*100:5.1f}%  全期 {tot:>10.0f}%{mark}")
    print(f"\n最优: N={best[1]} th={best[2]:+.1f}% 全期 {best[0]:.0f}%  分年 {best[3]}")
    print(f"基线(全收盘): {compound(rc)*100:.0f}%")

    print("\n=== 对照: 假规则(阈值无效时的随机切换) ===")
    rng = np.random.default_rng(168)
    base = compound(rc) * 100
    for p_ in (0.1, 0.3, 0.5):
        vals = []
        for _ in range(300):
            pick = rng.random(len(a)) < p_
            vals.append(compound(np.where(pick, ro, rc)) * 100)
        print(f"  随机开盘卖 {p_*100:.0f}%: 中位 {np.median(vals):.0f}%  "
              f"5~95% [{np.percentile(vals,5):.0f}, {np.percentile(vals,95):.0f}]  (基线 {base:.0f}%)")

    a.to_pickle(os.path.join(ROOT, "data", "cache", "seg3_trades.pkl"))


if __name__ == "__main__":
    main()
