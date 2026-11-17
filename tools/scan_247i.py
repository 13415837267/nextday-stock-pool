"""第247轮 I: 修掉 246k 里的一个未来信息, 重跑 dh60 扫描

246k 的 sel() 是这么写的:
    lo = g[g.n1_gap <= 0.05].sort_values(...)   # 只留低开的票
    top = lo.iloc[0]                             # "低开票里排名第一"
    broke = (top.idx_ma20_dist <= -3.5) and (top.dh60 <= -24.46)
`n1_gap` 是 T+1 开盘相对 T 日收盘的缺口 —— T 日收盘时不知道。
于是 broke 的判定用的是"明天低开的那只票"的 dh60, 而不是 T 日排名第一那只的。
在 dh60<=-24.46 这个几乎不筛人的门槛下影响很小(232/243 都满足),
但 247 要把门槛收紧到 -35(开始筛人), 这个偏差就会被放大。

正确口径 (与 245 的 T+1 重选一致):
    T 日收盘: 全池按 9 窗口平均排名排序 → top1 = 第 1 名(不管它明天开在哪)
    broke = (top1.idx_dist <= -3.5) and (top1.dh60 <= dh)      ← 全部是 T 日可知
    deep  = -9.5 < top1.n1_gap <= -5                            ← T+1 9:25 才知道, 但那时决策
    T+1 9:25: 若 broke 或 deep, 从排名往下找第一只"低开且未超限次"的买入
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
    print("第247轮 I: 去掉未来信息后重扫 dh60")
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
    gb = {int(k): g for k, g in full.groupby("trade_date")}
    print(f"全池 {len(full):,} 行 / {len(gb)} 天")

    def sel(dh: float, use_future: bool = False) -> pd.DataFrame:
        n: dict[str, int] = {}
        out = []
        for dt in sorted(gb):
            g = gb[dt]
            top1 = g.iloc[0]                      # T 日排名第一(不管明天开在哪)
            if use_future:                        # 246k 的错口径: 低开票里的第一
                g_low = g[g.n1_gap <= 0.05]
                if not len(g_low):
                    continue
                top1 = g_low.iloc[0]
            broke = (top1.idx_ma20_dist <= -3.5) and (top1.dh60 <= dh)
            gap = float(top1.n1_gap)
            deep = (-9.5 < gap <= -5.0)
            if not (broke or deep):
                continue
            pick = None
            src = g if not use_future else g[g.n1_gap <= 0.05]
            for j in range(len(src)):
                rr = src.iloc[j]
                if float(rr.n1_gap) > 0.05:
                    continue
                if n.get(rr.ts_code, 0) < 8:
                    pick = rr
                    break
            if pick is None:
                continue
            n[pick.ts_code] = n.get(pick.ts_code, 0) + 1
            out.append({"trade_date": dt, "ts_code": pick.ts_code, "y": pick.y,
                        "n1_gap": pick.n1_gap, "brk": broke,
                        "path": "破位" if broke else "深低开",
                        "idx": top1.idx_ma20_dist, "yr": dt // 10000})
        return pd.DataFrame(out)

    print("\n【0】两种口径的差 (-24.46 时)")
    a = sel(-24.46)
    b = sel(-24.46, use_future=True)
    print(f"  正确 {len(a)} 笔 单笔 {(a.y - COST).mean():.2f}%    "
          f"246k 口径 {len(b)} 笔 单笔 {(b.y - COST).mean():.2f}%    "
          f"同日同票 {len(a.merge(b, on=['trade_date','ts_code']))} 天")

    print("\n【1】正确口径下 dh60 扫描")
    print(f"{'dh':>8}{'笔数':>6}{'单笔':>8}{'胜率':>6}{'年化':>9}{'回撤':>9}{'卡玛':>7}"
          f"{'  16-20':>9}{'21-23':>9}{'24-26':>9}")
    best = None
    for dh in (-20.0, -24.46, -28.0, -32.0, -35.0, -38.0, -40.0, -45.0, -50.0):
        c = sel(dh)
        sim = base.copy()
        v = pd.Series((c.y.values - COST) * 0.40, index=c.trade_date.values).sort_index()
        sim[v.index] = sim[v.index] + v.values
        ann, dd, cm = metrics(sim)
        segs = []
        for lo_, hi_ in ((2016, 2020), (2021, 2023), (2024, 2026)):
            s = sim[(sim.index // 10000 >= lo_) & (sim.index // 10000 <= hi_)]
            segs.append(metrics(s)[2])
        print(f"{dh:>8.1f}{len(c):>6}{(c.y - COST).mean():>8.2f}"
              f"{((c.y - COST) > 0).mean() * 100:>6.0f}{ann:>9.2f}{dd:>9.2f}{cm:>7.2f}"
              f"{segs[0]:>9.2f}{segs[1]:>9.2f}{segs[2]:>9.2f}")
        c.to_csv(OUT / f"w247i_{abs(dh):.0f}.csv", index=False, encoding="utf-8-sig")
        if best is None or cm > best[1]:
            best = (dh, cm)
    print(f"\n  最优 {best[0]:.1f} → 卡玛 {best[1]}")

    print("\n【2】定稿分年 (取 -35)")
    c = sel(-35.0)
    sim = base.copy()
    v = pd.Series((c.y.values - COST) * 0.40, index=c.trade_date.values).sort_index()
    sim[v.index] = sim[v.index] + v.values
    ann, dd, cm = metrics(sim)
    yr = sim.groupby(sim.index // 10000).apply(lambda x: (1 + x / 100).prod() - 1) * 100
    for k, val in yr.items():
        print(f"  {k}: {val:+7.1f}%   {int((c.trade_date // 10000 == k).sum())} 笔")
    print(f"  → {len(c)} 笔 (破位 {int(c.brk.sum())} / 深低开 {int((~c.brk).sum())})  "
          f"年化 {ann:.2f}%  回撤 {dd:.2f}%  卡玛 {cm:.2f}  正年 {int((yr > 0).sum())}/{len(yr)}")
    print(f"  票数 {c.ts_code.nunique()}  最长连买 {int(c.groupby('ts_code').size().max())}  "
          f"前5票 {c.ts_code.value_counts().head(5).sum() / len(c) * 100:.0f}%")

    print("\n【3】仓位面 (-35)")
    for P in (0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6):
        sim2 = base.copy()
        v2 = pd.Series((c.y.values - COST) * P, index=c.trade_date.values).sort_index()
        sim2[v2.index] = sim2[v2.index] + v2.values
        print(f"  {P:.0%}: 年化 {metrics(sim2)[0]:>7.2f}  回撤 {metrics(sim2)[1]:>7.2f}  "
              f"卡玛 {metrics(sim2)[2]:>5.2f}")

    c.to_csv(OUT / "w247_i_picks.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
