"""第247轮 B(修正版): 深低开路径从 top1 铺开到 topN

★ 上一版的错误: 破位判定用"全池排名第一", 而 246 定稿用的是"低开票里排名第一"
  (先 `n1_gap <= 0.05` 过滤再排序 —— 这就是 T+1 开盘重选的实现)。
  口径不对导致破位路径复现成 306 笔 / 1.60%, 基线也从 86.4% 掉到 32%(base 日历被截断)。
  本版直接复用 246k 的 sel 结构。

深低开路径 = 非破位日, 在"低开票"里找 gap ∈ (-9.5%, -5%] 且 排名前 N 的那只。
N=1 即 246 定稿口径(24 笔)。N 放大看单笔会不会塌。
"""

from __future__ import annotations

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
    tr = pd.read_csv(OUT / "w232_trades.csv")
    all_dates = sorted(int(p.stem.replace(".csv", "")) for p in
                       (ROOT / "data" / "panel").glob("*.csv.gz"))
    cal = np.array([x for x in all_dates if x >= int(tr.signal.min())])
    base = pd.Series(0.0, index=cal)
    busy = set(int(x) for x in tr.signal)
    for _, r in tr.iterrows():
        busy.add(int(r["buy"]))
        busy.add(int(r["sell"]))
    for dt, g in tr.groupby("signal"):
        base[int(dt)] = g["ret"].mean()
    print(f"仅主策略 {metrics(base)}  日历 {cal[0]}~{cal[-1]} {len(cal)} 天")

    d = pd.read_parquet(OUT / "w246b_all.parquet",
                        columns=["ts_code", "trade_date", "listed", "n1_gap", "y",
                                 "amt_yi"] + DHC)
    d = d[d.listed >= 250]
    d["trade_date"] = d["trade_date"].astype("int64")
    d = d[(d.trade_date >= cal[0]) & (~d.trade_date.isin(busy))]
    mk = pd.read_csv(OUT / "feat234" / "market.csv",
                     usecols=["trade_date", "idx_ma20_dist"])
    d = d.merge(mk, on="trade_date", how="inner")
    d = d.dropna(subset=DHC + ["n1_gap", "y", "idx_ma20_dist"])

    parts = []
    for _yr, g in d.groupby(d.trade_date // 10000, sort=True):
        den = g[DHC].notna().sum()
        mr = (g[DHC].rank(ascending=True) / den).mean(axis=1, skipna=True)
        g = g.assign(mr=mr)
        sub = g[g.n1_gap <= 0.05].sort_values(
            ["trade_date", "mr", "amt_yi"], ascending=[True, True, False])
        parts.append(sub)
    lo = pd.concat(parts, ignore_index=True)
    del parts, d
    lo = lo.sort_values(["trade_date", "mr", "amt_yi"], ascending=[True, True, False])
    lo["rk"] = lo.groupby("trade_date").cumcount() + 1
    gb = {int(dt): g for dt, g in lo.groupby("trade_date")}
    print(f"低开票池 {len(lo):,} 行 / {len(gb)} 天")

    def sel(N: int, mode: str = "B") -> pd.DataFrame:
        """N = 深低开路径允许的排名上限。mode=A 表示不含深低开路径。"""
        n: dict[str, int] = {}
        out = []
        for dt in sorted(gb):
            g = gb[dt]
            top = g.iloc[0]
            broke = (top.idx_ma20_dist <= -3.5) and (top.dh60 <= -24.46)
            gap = float(top.n1_gap)
            deep = (-9.5 < gap <= -5.0)
            if not (broke or (mode == "B" and deep)):
                continue
            if broke:
                pick = top
                for j in range(len(g)):
                    rr = g.iloc[j]
                    if n.get(rr.ts_code, 0) < 8:
                        pick = rr
                        break
                path = "破位"
            else:
                cand = g[(g.n1_gap > -9.5) & (g.n1_gap <= -5.0) & (g.rk <= N)]
                if not len(cand):
                    continue
                pick = None
                for j in range(len(cand)):
                    rr = cand.iloc[j]
                    if n.get(rr.ts_code, 0) < 8:
                        pick = rr
                        break
                if pick is None:
                    continue
                path = "深低开"
            n[pick.ts_code] = n.get(pick.ts_code, 0) + 1
            out.append({"trade_date": dt, "ts_code": pick.ts_code, "y": pick.y,
                        "n1_gap": pick.n1_gap, "brk": broke, "path": path,
                        "idx": top.idx_ma20_dist, "rk": pick.rk,
                        "yr": dt // 10000, "m": dt // 100 % 100})
        return pd.DataFrame(out)

    A = sel(1, "A")
    print(f"\nA(仅破位) {len(A)} 笔  单笔 {(A.y - COST).mean():.2f}%  "
          f"卡玛 {metrics(_comb(base, A, 0.40))[2]}")
    B1 = sel(1)
    print(f"B'(N=1)   {len(B1)} 笔  单笔 {(B1.y - COST).mean():.2f}%   "
          f"其中深低开 {int((B1.path == '深低开').sum())} 笔  "
          f"卡玛 {metrics(_comb(base, B1, 0.40))[2]}")

    print(f"\n【1】深低开路径扩到前 N 名")
    print(f"{'N':>6}{'深低开笔数':>10}{'单笔':>9}{'胜率':>7}{'t':>7}"
          f"{'2016-20':>10}{'2021-26':>10}")
    rows = []
    for N in (1, 2, 3, 5, 10, 20, 50):
        c = sel(N)
        dp = c[c.path == "深低开"]
        y = dp.y - COST
        t = y.mean() / (y.std(ddof=1) / np.sqrt(len(y))) if len(y) > 2 else np.nan
        a0 = y[dp.trade_date < 20210000]
        a1 = y[dp.trade_date >= 20210000]
        print(f"{N:>6}{len(dp):>10}{y.mean():>9.2f}{(y > 0).mean() * 100:>7.0f}{t:>7.2f}"
              f"{(a0.mean() if len(a0) else np.nan):>10.2f}"
              f"{(a1.mean() if len(a1) else np.nan):>10.2f}")
        rows.append({"N": N, "n": len(dp), "单笔": round(y.mean(), 2),
                     "胜率": round((y > 0).mean() * 100, 1), "t": round(t, 2),
                     "前段": round(a0.mean(), 2) if len(a0) else np.nan,
                     "后段": round(a1.mean(), 2) if len(a1) else np.nan})
        c.to_csv(OUT / f"w247b_N{N}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(rows).to_csv(OUT / "w247b_grid.csv", index=False, encoding="utf-8-sig")

    print(f"\n【2】破位 40% + 深低开(P2) 的组合")
    print(f"{'N':>5}{'P2':>6}{'总笔数':>8}{'年化':>9}{'回撤':>9}{'卡玛':>7}")
    combo = []
    for N in (1, 3, 5, 10, 20):
        c = sel(N)
        for P2 in (0.0, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0):
            a, dd, cm = metrics(_comb(base, c, P2))
            combo.append({"N": N, "P2": P2, "n": len(c), "年化": a, "回撤": dd, "卡玛": cm})
            print(f"{N:>5}{P2:>6.0%}{len(c):>8}{a:>9.2f}{dd:>9.2f}{cm:>7.2f}")
        print()
    pd.DataFrame(combo).to_csv(OUT / "w247b_combo.csv", index=False, encoding="utf-8-sig")


def _comb(base: pd.Series, c: pd.DataFrame, P2: float) -> pd.Series:
    sim = base.copy()
    for path, P in (("破位", 0.40), ("深低开", P2)):
        if P <= 0:
            continue
        s = c[c.path == path]
        if not len(s):
            continue
        v = pd.Series((s.y.values - COST) * P, index=s.trade_date.values).sort_index()
        sim[v.index] = sim[v.index] + v.values
    return sim


if __name__ == "__main__":
    main()
