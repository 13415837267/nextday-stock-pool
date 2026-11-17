"""第247轮 J: 正确口径下, 深低开路径到底能做多少次

I 轮发现 246k 的 sel() 有个未来信息: 它先用 `n1_gap <= 0.05` 把票筛成"明天会低开的",
再拿这批票的第一名去判"破位"。于是:
  · 排名第一那只高开时, 破位判定用的是排名第二那只的 dh60 (T 日不知道)
  · 深低开路径判定用的是"低开票里第一只"的缺口, 而不是真正 T 日排名第一的缺口
修正后: 256 笔 → 247 笔, 深低开 24 笔 → 14 笔, 卡玛 5.16 → 4.03。

那深低开这条路本来就该更大: T+1 9:25 我可以看全池每一只的开盘价,
不必只盯 T 日第一名。只要某只票"排名靠前 + 自己崩了 5~9.5%", 都可以做。
本脚本把 N 放开, 看笔数与单笔的权衡。
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
COST = 0.30
WINS = (60, 90, 120, 150, 180, 200, 250, 300, 360)
DHC = [f"dh{W}" for W in WINS]


def metrics(r: pd.Series) -> tuple[float, float, float]:
    nav = (1 + r / 100).cumprod()
    ann = nav.iloc[-1] ** (244 / len(r)) - 1
    dd = (nav / nav.cummax() - 1).min()
    return round(ann * 100, 2), round(dd * 100, 2), round(ann * 100 / abs(dd * 100), 2)


def main() -> None:
    print("=" * 78)
    print("第247轮 J: 正确口径下深低开路径的容量")
    print("=" * 78)

    tr = pd.read_csv(OUT / "w232_trades.csv")
    tr["signal"] = tr["signal"].astype(int)
    cal = sorted(int(f.split("/")[-1].replace(".csv.gz", ""))
                 for f in glob.glob(str(ROOT / "data" / "panel" / "*.csv.gz")))
    cal = np.array([d for d in cal if d >= tr.signal.min()])
    base = pd.Series(0.0, index=cal)
    busy = set(int(x) for x in tr.signal)
    for _, r in tr.iterrows():
        busy.add(int(r["buy"]))
        busy.add(int(r["sell"]))
    for dt, g in tr.groupby("signal"):
        base[int(dt)] = g["ret"].mean()

    d = pd.read_parquet(OUT / "w246b_all.parquet",
                        columns=["ts_code", "trade_date", "listed", "n1_gap", "y",
                                 "amt_yi"] + DHC)
    d = d[d.listed >= 250]
    d["trade_date"] = d["trade_date"].astype("int64")
    d = d[(d.trade_date >= cal[0]) & (~d.trade_date.isin(busy))]
    mk = pd.read_csv(OUT / "feat234" / "market.csv", usecols=["trade_date", "idx_ma20_dist"])
    d = d.merge(mk, on="trade_date", how="inner")
    d = d.dropna(subset=DHC + ["n1_gap", "y", "idx_ma20_dist"])

    parts = []
    for _yr, g in d.groupby(d.trade_date // 10000, sort=True):
        den = g[DHC].notna().sum()
        mr = (g[DHC].rank(ascending=True) / den).mean(axis=1, skipna=True)
        g = g.assign(mr=mr)
        parts.append(g)
    full = pd.concat(parts, ignore_index=True)
    del parts, d
    full = full.sort_values(["trade_date", "mr", "amt_yi"],
                            ascending=[True, True, False])
    full["rk"] = full.groupby("trade_date").cumcount() + 1
    gb = {int(k): g for k, g in full.groupby("trade_date")}
    tops = {k: v.iloc[0] for k, v in gb.items()}
    print(f"全池 {len(full):,} 行 / {len(gb)} 天")

    # 每天"排名前 50 且 gap 在 -9.5~-5"的票数, 看这条路到底有多少候选
    cand = full[(full.n1_gap > -9.5) & (full.n1_gap <= -5.0) & (full.rk <= 50)]
    per = cand.groupby("trade_date").size()
    print(f"排名前50 且 gap∈(-9.5,-5] 的天数 {len(per)}  每日候选数: "
          f"中位 {per.median():.0f} P90 {per.quantile(0.9):.0f} 最大 {per.max()}")

    def sel(N: int, dh: float = -35.0, deep_hi: float = -5.0) -> pd.DataFrame:
        n: dict[str, int] = {}
        out = []
        for dt in sorted(gb):
            g = gb[dt]
            t1 = tops[dt]
            broke = (t1.idx_ma20_dist <= -3.5) and (t1.dh60 <= dh)
            if broke:
                pick = None
                for j in range(len(g)):
                    rr = g.iloc[j]
                    if float(rr.n1_gap) <= 0.05 and n.get(rr.ts_code, 0) < 8:
                        pick = rr
                        break
                if pick is None:
                    continue
                path = "破位"
            else:
                cc = g[(g.n1_gap > -9.5) & (g.n1_gap <= deep_hi) & (g.rk <= N)]
                pick = None
                for j in range(len(cc)):
                    rr = cc.iloc[j]
                    if n.get(rr.ts_code, 0) < 8:
                        pick = rr
                        break
                if pick is None:
                    continue
                path = "深低开"
            n[pick.ts_code] = n.get(pick.ts_code, 0) + 1
            out.append({"trade_date": dt, "ts_code": pick.ts_code, "y": pick.y,
                        "n1_gap": pick.n1_gap, "brk": broke, "path": path,
                        "idx": t1.idx_ma20_dist, "rk": pick.rk, "yr": dt // 10000})
        return pd.DataFrame(out)

    print("\n【1】深低开排名上限 N (dh=-35, P=40%)")
    print(f"{'N':>5}{'深低开':>7}{'单笔':>8}{'总笔':>6}{'单笔':>8}"
          f"{'年化':>9}{'回撤':>9}{'卡玛':>7}{'  16-20':>9}{'21-23':>9}{'24-26':>9}")
    rows = []
    for N in (1, 2, 3, 5, 10, 20, 50):
        c = sel(N)
        dp = c[c.path == "深低开"]
        sim = base.copy()
        v = pd.Series((c.y.values - COST) * 0.40, index=c.trade_date.values).sort_index()
        sim[v.index] = sim[v.index] + v.values
        ann, dd, cm = metrics(sim)
        segs = []
        for lo_, hi_ in ((2016, 2020), (2021, 2023), (2024, 2026)):
            s = sim[(sim.index // 10000 >= lo_) & (sim.index // 10000 <= hi_)]
            segs.append(metrics(s)[2])
        print(f"{N:>5}{len(dp):>7}{(dp.y - COST).mean():>8.2f}{len(c):>6}"
              f"{(c.y - COST).mean():>8.2f}{ann:>9.2f}{dd:>9.2f}{cm:>7.2f}"
              f"{segs[0]:>9.2f}{segs[1]:>9.2f}{segs[2]:>9.2f}")
        rows.append({"N": N, "deep": len(dp), "deep_bar": round((dp.y - COST).mean(), 2),
                     "n": len(c), "bar": round((c.y - COST).mean(), 2),
                     "年化": ann, "回撤": dd, "卡玛": cm})
        c.to_csv(OUT / f"w247j_N{N}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(rows).to_csv(OUT / "w247j_grid.csv", index=False, encoding="utf-8-sig")

    print("\n【2】深低开下界也放开 (N=10)")
    print(f"{'区间':>12}{'深低开':>7}{'单笔':>8}{'总笔':>6}{'年化':>9}{'回撤':>9}{'卡玛':>7}")
    for hi in (-5.0, -4.0, -3.0, -2.0):
        c = sel(10, deep_hi=hi)
        dp = c[c.path == "深低开"]
        sim = base.copy()
        v = pd.Series((c.y.values - COST) * 0.40, index=c.trade_date.values).sort_index()
        sim[v.index] = sim[v.index] + v.values
        ann, dd, cm = metrics(sim)
        print(f"{'-9.5~%.0f' % hi:>12}{len(dp):>7}{(dp.y - COST).mean():>8.2f}"
              f"{len(c):>6}{ann:>9.2f}{dd:>9.2f}{cm:>7.2f}")


if __name__ == "__main__":
    main()
