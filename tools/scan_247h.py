"""第247轮 H: 定稿 —— dh60 -24.46 → -35, 并复核仓位/分年"""

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
DH_NEW = -35.0


def metrics(r: pd.Series) -> tuple[float, float, float]:
    nav = (1 + r / 100).cumprod()
    ann = nav.iloc[-1] ** (244 / len(r)) - 1
    dd = (nav / nav.cummax() - 1).min()
    return round(ann * 100, 2), round(dd * 100, 2), round(ann * 100 / abs(dd * 100), 2)


def main() -> None:
    print("=" * 78)
    print("第247轮 H: 定稿")
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
        parts.append(g[g.n1_gap <= 0.05])
    lo = pd.concat(parts, ignore_index=True)
    del parts, d
    lo = lo.sort_values(["trade_date", "mr", "amt_yi"], ascending=[True, True, False])
    gb = {int(k): g for k, g in lo.groupby("trade_date")}

    def sel(dh: float) -> pd.DataFrame:
        n: dict[str, int] = {}
        out = []
        for dt in sorted(gb):
            g = gb[dt]
            top = g.iloc[0]
            broke = (top.idx_ma20_dist <= -3.5) and (top.dh60 <= dh)
            gap = float(top.n1_gap)
            deep = (-9.5 < gap <= -5.0)
            if not (broke or deep):
                continue
            pick = top
            for j in range(len(g)):
                rr = g.iloc[j]
                if n.get(rr.ts_code, 0) < 8:
                    pick = rr
                    break
            n[pick.ts_code] = n.get(pick.ts_code, 0) + 1
            out.append({"trade_date": dt, "ts_code": pick.ts_code, "y": pick.y,
                        "n1_gap": pick.n1_gap, "brk": broke,
                        "path": "破位" if broke else "深低开",
                        "idx": top.idx_ma20_dist, "yr": dt // 10000})
        return pd.DataFrame(out)

    c = sel(DH_NEW)
    c.to_csv(OUT / "w247_h_picks.csv", index=False, encoding="utf-8-sig")
    print(f"定稿方案 {len(c)} 笔  (破位 {int(c.brk.sum())} / 深低开 {int((~c.brk).sum())})")

    print("\n【1】仓位面 (P1 = P2 = P)")
    print(f"{'P':>6}{'年化':>9}{'回撤':>9}{'卡玛':>7}")
    for P in (0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7):
        sim = base.copy()
        v = pd.Series((c.y.values - COST) * P, index=c.trade_date.values).sort_index()
        sim[v.index] = sim[v.index] + v.values
        a, dd, cm = metrics(sim)
        print(f"{P:>6.0%}{a:>9.2f}{dd:>9.2f}{cm:>7.2f}")

    print("\n【2】P1=40% 时 P2 的曲线 (深低开 24 笔到底配多少仓)")
    print(f"{'P2':>6}{'笔数':>6}{'年化':>9}{'回撤':>9}{'卡玛':>7}")
    for P2 in (0.0, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0):
        sim = base.copy()
        for path, P in (("破位", 0.40), ("深低开", P2)):
            s = c[c.path == path]
            if not len(s) or P <= 0:
                continue
            v = pd.Series((s.y.values - COST) * P, index=s.trade_date.values).sort_index()
            sim[v.index] = sim[v.index] + v.values
        a, dd, cm = metrics(sim)
        print(f"{P2:>6.0%}{len(c):>6}{a:>9.2f}{dd:>9.2f}{cm:>7.2f}")

    print("\n【3】定稿 40% 仓 分年")
    sim = base.copy()
    v = pd.Series((c.y.values - COST) * 0.40, index=c.trade_date.values).sort_index()
    sim[v.index] = sim[v.index] + v.values
    a, dd, cm = metrics(sim)
    yr = sim.groupby(sim.index // 10000).apply(lambda x: (1 + x / 100).prod() - 1) * 100
    for k, val in yr.items():
        n = int((c.trade_date // 10000 == k).sum())
        print(f"  {k}: {val:+7.1f}%   {n} 笔")
    print(f"  → 年化 {a:.2f}%  回撤 {dd:.2f}%  卡玛 {cm:.2f}  正年 {int((yr > 0).sum())}/{len(yr)}")

    print("\n【4】与前三轮对照")
    b, bd, bc = metrics(base)
    print(f"  仅主策略                      {b:>7.2f} {bd:>7.2f} {bc:>5.2f}")
    print(f"  246 定稿 (dh-24.46, P40/40)   {131.69:>7.2f} {-25.52:>7.2f} {5.16:>5.2f}")
    print(f"  247 定稿 (dh-35.0,  P40/40)   {a:>7.2f} {dd:>7.2f} {cm:>5.2f}")

    print("\n【5】集中度")
    print(f"  票数 {c.ts_code.nunique()}  最长连买 {int(c.groupby('ts_code').size().max())} 次"
          f"  前5票占比 {c.ts_code.value_counts().head(5).sum() / len(c) * 100:.0f}%")
    print(f"  单笔 {(c.y - COST).mean():.2f}%  中位 {(c.y - COST).median():.2f}%  "
          f"胜率 {((c.y - COST) > 0).mean() * 100:.0f}%")

    sim.to_frame("r").to_csv(OUT / "w247_h_nav.csv", encoding="utf-8-sig")


if __name__ == "__main__":
    main()
