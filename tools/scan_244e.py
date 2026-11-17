"""第244轮 E: 集成是真本事还是"9"这个数字碰巧? —— 单窗口对照 + 随机子集

244D: 集成(9 窗口排名平均) + 冷却 10 + 50% 仓 = 年化 136.5 / 回撤 -24.55 / 卡玛 5.56
     而 243 定稿(250 窗口 + 冷却 20 + 70% 仓) 同口径只有 117.2 / -27.09 / 4.33

要证明增益来自"多窗口平均"本身, 而不是碰巧凑了 9 个窗口, 需要:
  ① 同样口径下 9 个单窗口各自的卡玛 —— 集成必须高于其中每一个
  ② 随机抽 3/5/7 个窗口做集成, 各 200 次 —— 集成增益应该是"抽得越多越稳",
     而不是"必须正好这 9 个"
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
POOL = OUT / "w244_pool"

WINS = (60, 90, 120, 150, 180, 200, 250, 300, 360)
COST = 0.10
C = 10
P = 0.50


def metrics(r: pd.Series) -> tuple[float, float, float]:
    nav = (1 + r / 100).cumprod()
    ann = nav.iloc[-1] ** (244 / len(r)) - 1
    dd = (nav / nav.cummax() - 1).min()
    return round(ann * 100, 2), round(dd * 100, 2), round(ann * 100 / abs(dd * 100), 2)


def main() -> None:
    sig = pd.read_parquet(POOL / "signals.parquet")
    sig["trade_date"] = sig["trade_date"].astype("int64")
    tr = pd.read_csv(OUT / "w232_trades.csv")
    all_dates = sorted(int(p.stem.replace(".csv", "")) for p in
                       (ROOT / "data" / "panel").glob("*.csv.gz"))
    cal = np.array([d for d in all_dates if d >= tr.signal.min()])
    base = pd.Series(0.0, index=cal)
    for dt, g in tr.groupby("signal"):
        base[int(dt)] = g["ret"].mean()
    dayrank = {int(d): i for i, d in enumerate(cal)}

    wide = sig[["ts_code", "trade_date", "y", "amt_yi"]].copy()
    for W in WINS:
        px = pd.read_parquet(POOL / f"W{W}.parquet").rename(columns={"dh": f"dh{W}"})
        wide = wide.merge(px, on=["ts_code", "trade_date"], how="left")
        del px
    dhcols = [f"dh{W}" for W in WINS]
    wide = wide.dropna(subset=dhcols, how="all")
    byday = {int(dt): g for dt, g in wide.groupby("trade_date")}
    order = sorted(byday)

    # 每日原始值(不排名), 供任意子集现场排名
    raw: dict[int, pd.DataFrame] = {}
    for dt in order:
        g = byday[dt].copy()
        raw[dt] = g

    def run(cols: list[str]) -> tuple[pd.Series, pd.Series]:
        last: dict[str, int] = {}
        v, pk = {}, {}
        for dt in order:
            g = raw[dt].copy()
            denom = {c: max(int(g[c].notna().sum()), 1) for c in cols}
            rk = pd.DataFrame({c: g[c].rank(ascending=True) / denom[c] for c in cols})
            g["mr"] = rk.mean(axis=1, skipna=True)
            g = g.sort_values(["mr", "amt_yi"], ascending=[True, False])
            ok = [(c not in last) or (dayrank[dt] - last[c] >= C) for c in g["ts_code"]]
            cand = g[np.array(ok)]
            if len(cand):
                g = cand
            r = g.iloc[0]
            if pd.isna(r["mr"]):
                continue
            v[dt] = float(r["y"]) - COST
            pk[dt] = r["ts_code"]
            last[r["ts_code"]] = dayrank[dt]
        return pd.Series(v).sort_index(), pd.Series(pk).sort_index()

    def score(v: pd.Series) -> tuple[float, float, float]:
        sim = base.copy()
        sim[v.index] += v.values * P
        return metrics(sim)

    print(f"冷却 {C} 日, 仓位 {P:.0%}, 单笔成本 {COST}%")
    print(f"\n{'单窗口':>8}{'次数':>6}{'单笔':>8}{'年化':>9}{'回撤':>9}{'卡玛':>7}{'票数':>6}{'连买':>5}")
    single = {}
    for c in dhcols:
        v, pk = run([c])
        a, dd, cm = score(v)
        single[c] = cm
        run_len = mx = 0
        prev = None
        for x in pk.values:
            run_len = run_len + 1 if x == prev else 1
            prev = x
            mx = max(mx, run_len)
        print(f"{c:>8}{len(v):>6}{v.mean():>8.3f}{a:>9.2f}{dd:>9.2f}{cm:>7.2f}"
              f"{pk.nunique():>6}{mx:>5}")
    print(f"  单窗口中位 {np.median(list(single.values())):.2f}  "
          f"最好 {max(single.values()):.2f} ({max(single, key=single.get)})")

    v9, pk9 = run(dhcols)
    a, dd, cm9 = score(v9)
    print(f"\n全 9 窗口集成: 年化{a:.2f} 回撤{dd:.2f} 卡玛 {cm9:.2f}  票数 {pk9.nunique()}")

    print(f"\n{'子集大小':>8}{'中位':>7}{'5%':>7}{'95%':>7}{'最优':>7}{'>9窗口?':>9}")
    rng = np.random.default_rng(20261117)
    for k in (2, 3, 5, 7, 9):
        cs = []
        for _ in range(120):
            sub = list(rng.choice(dhcols, size=k, replace=False))
            v, _pk = run(sub)
            cs.append(score(v)[2])
        cs = np.array(cs)
        print(f"{k:>8}{np.median(cs):>7.2f}{np.percentile(cs, 5):>7.2f}"
              f"{np.percentile(cs, 95):>7.2f}{cs.max():>7.2f}"
              f"{(cs >= cm9).mean() * 100:>8.0f}%")


if __name__ == "__main__":
    main()
