"""第244轮 D: 集成方案细化 —— 冷却档 x 仓位档, 集中度, 分年, 与单窗口的配对比较

244C 结果:
  IC(历史卡玛排名 -> 次年卡玛排名) = -0.158, 9 年只有 3 年为正
  => "挑参数"没有信息量, 243 轮定稿的 (250, 20) 在 2017+ 只有 4.41 卡玛,
     比随机选参数的中位数(4.88)还低。

  9 窗口排名平均 + 冷却 10 日: 卡玛 5.09 (2017+ 5.27), 单笔 2.584%,
  用 141 只票、前 5 票占 11% —— 比单窗口 (250, 20) 的 4.33 高 0.76, 分散度还更好。
  代价: 回撤 -31.1% vs -27.1%。

本轮把集成做成可交付的定稿: 冷却 x 仓位全扫 + 集中度 + 分年 + 配对检验。
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


def metrics(r: pd.Series) -> tuple[float, float, float]:
    if not len(r):
        return 0.0, 0.0, 0.0
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
    span = base.index[base.index // 10000 >= 2017]

    wide = sig[["ts_code", "trade_date", "y", "amt_yi"]].copy()
    for W in WINS:
        px = pd.read_parquet(POOL / f"W{W}.parquet").rename(columns={"dh": f"dh{W}"})
        wide = wide.merge(px, on=["ts_code", "trade_date"], how="left")
        del px
    dhcols = [f"dh{W}" for W in WINS]
    wide = wide.dropna(subset=dhcols, how="all")
    byday = {int(dt): g for dt, g in wide.groupby("trade_date")}
    order = sorted(byday)

    # 预计算每日排名(与冷却无关) —— 只算一次
    prep: dict[int, pd.DataFrame] = {}
    for dt in order:
        g = byday[dt].copy()
        denom = {c: max(int(g[c].notna().sum()), 1) for c in dhcols}
        rk = pd.DataFrame({c: g[c].rank(ascending=True) / denom[c] for c in dhcols})
        g["mr"] = rk.mean(axis=1, skipna=True)
        prep[dt] = g.sort_values(["mr", "amt_yi"], ascending=[True, False])
    print(f"{len(order)} 天候选已排好")

    # ---------- 1. 冷却 x 仓位 ----------
    seqs: dict[int, pd.Series] = {}
    picks: dict[int, pd.Series] = {}
    for C in (0, 5, 10, 15, 20, 25, 30):
        last: dict[str, int] = {}
        v, pk = {}, {}
        for dt in order:
            g = prep[dt]
            if C > 0:
                ok = [(c not in last) or (dayrank[dt] - last[c] >= C) for c in g["ts_code"]]
                cand = g[np.array(ok)]
                if len(cand):
                    g = cand
            r = g.iloc[0]
            v[dt] = float(r["y"]) - COST
            pk[dt] = r["ts_code"]
            last[r["ts_code"]] = dayrank[dt]
        seqs[C] = pd.Series(v).sort_index()
        picks[C] = pd.Series(pk).sort_index()

    b = metrics(base)
    print(f"\n基准 年化{b[0]:.2f} 回撤{b[1]:.2f} 卡玛{b[2]:.2f}")
    print(f"\n{'冷却':>5}{'仓位':>6}{'次数':>5}{'单笔':>8}{'年化':>9}{'回撤':>9}{'卡玛':>7}"
          f"{'票数':>5}{'连买':>5}{'前5%':>6}{'2017+':>7}")
    rows = []
    for C, v in seqs.items():
        pk = picks[C]
        vc = pk.value_counts()
        run = mx = 0
        prev = None
        for c in pk.values:
            run = run + 1 if c == prev else 1
            prev = c
            mx = max(mx, run)
        top5 = vc.head(5).sum() / len(pk) * 100
        for P in (0.30, 0.40, 0.50, 0.60, 0.70, 0.85, 1.00):
            sim = base.copy()
            sim[v.index] += v.values * P
            a, dd, cm = metrics(sim)
            s2 = base.loc[span].copy()
            v2 = v[(v.index // 10000) >= 2017]
            s2[v2.index] += v2.values * P
            _, _, c2 = metrics(s2)
            rows.append({"C": C, "P": P, "n": len(v), "单笔": round(v.mean(), 3),
                         "年化": a, "回撤": dd, "卡玛": cm, "票数": len(vc),
                         "最长连买": mx, "前5%": round(top5, 0), "卡玛2017": c2})
            if P in (0.50, 0.60, 0.70):
                print(f"{C:>5}{P:>6.0%}{len(v):>5}{v.mean():>8.3f}{a:>9.2f}{dd:>9.2f}"
                      f"{cm:>7.2f}{len(vc):>5}{mx:>5}{top5:>5.0f}%{c2:>7.2f}")
    g = pd.DataFrame(rows)
    g.to_csv(OUT / "w244_d_grid.csv", index=False, encoding="utf-8-sig")

    # 约束: 回撤不差于基准(-24.24) 的最优
    ok = g[g["回撤"] >= b[1] - 1.0]
    print(f"\n回撤不差于基准-1pp 的最优 5 格:")
    print(ok.sort_values("卡玛", ascending=False).head(5).to_string(index=False))

    # ---------- 2. 定稿候选分年 ----------
    for C, P in ((10, 0.60), (10, 0.70), (20, 0.60)):
        v = seqs[C] * P
        sim = base.copy()
        sim[v.index] += v.values
        p = pd.DataFrame({"yr": v.index // 10000, "y": v.values * 100 / P})
        gg = p.groupby("yr").y.agg(["size", "sum", "mean"])
        gg["win%"] = p.groupby("yr").y.apply(lambda x: (x > 0).mean() * 100).round(0)
        a, dd, cm = metrics(sim)
        print(f"\n集成 C={C} P={P:.0%}: 年化{a:.2f} 回撤{dd:.2f} 卡玛{cm:.2f}")
        print(gg.round(2).to_string())

    # ---------- 3. 与单窗口 250 的配对 ----------
    v_int = seqs[10]
    p250 = pd.read_parquet(POOL / "W250.parquet").rename(columns={"dh": "dh250"})
    one = sig.merge(p250, on=["ts_code", "trade_date"], how="inner")
    one = one.sort_values(["trade_date", "dh250", "amt_yi"], ascending=[True, True, False])
    f = one.groupby("trade_date").first()
    s250 = (f["y"] - COST)
    s250.index = s250.index.astype("int64")
    code250 = f["ts_code"]
    code250.index = code250.index.astype("int64")
    common = v_int.index.intersection(s250.index)
    d = v_int.loc[common] - s250.loc[common]
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    hit = (picks[10].loc[common].values == code250.loc[common].values).mean()
    print(f"\n集成(C=10) vs 单窗口250 同日配对: n={len(d)}  "
          f"集成多赚 {d.mean():+.3f}pp/笔  t={t:.2f}  集成胜率 {(d > 0).mean() * 100:.0f}%  "
          f"选中同一只 {hit * 100:.0f}%")


if __name__ == "__main__":
    main()
