"""第244轮 I: 实盘口径(高开就放弃)下的参数面 + 回撤回因

244H: 198 次信号里 82 次(41%)T+1 高开, 实盘只能放弃。实盘口径:
  集成 C=10 P=50% -> 112.5 / -23.59 / 4.77
  集成 C=10 P=70% -> 123.0 / -23.59 / 5.21   (70% 仓回撤没变!)
     而 243 定稿 P=70% -> 100.9 / -24.17 / 4.17
仓位加上去回撤不动, 说明补位日全部避开了主策略的最大回撤段。那仓位到底能加到多少?
本轮扫实盘口径下的 冷却 x 仓位, 并把最大回撤定位到具体日期。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
POOL = OUT / "w244_pool2"

WINS = (60, 90, 120, 150, 180, 200, 250, 300, 360)
COST = 0.10


def metrics(r: pd.Series) -> tuple[float, float, float]:
    nav = (1 + r / 100).cumprod()
    ann = nav.iloc[-1] ** (244 / len(r)) - 1
    dd = (nav / nav.cummax() - 1).min()
    return round(ann * 100, 2), round(dd * 100, 2), round(ann * 100 / abs(dd * 100), 2)


def main() -> None:
    tr = pd.read_csv(OUT / "w232_trades.csv")
    all_dates = sorted(int(p.stem.replace(".csv", "")) for p in
                       (ROOT / "data" / "panel").glob("*.csv.gz"))
    cal = np.array([d for d in all_dates if d >= tr.signal.min()])
    base = pd.Series(0.0, index=cal)
    for dt, g in tr.groupby("signal"):
        base[int(dt)] = g["ret"].mean()
    dayrank = {int(d): i for i, d in enumerate(cal)}

    sig = pd.read_parquet(POOL / "signals.parquet")
    sig["trade_date"] = sig["trade_date"].astype("int64")
    wide = sig.copy()
    for W in WINS:
        px = pd.read_parquet(POOL / f"W{W}.parquet").rename(columns={"dh": f"dh{W}"})
        wide = wide.merge(px, on=["ts_code", "trade_date"], how="left")
        del px
    dhcols = [f"dh{W}" for W in WINS]
    wide = wide.dropna(subset=dhcols, how="all")
    byday = {int(dt): g for dt, g in wide.groupby("trade_date")}

    prep = {}
    for dt, g in byday.items():
        g = g.dropna(subset=dhcols).copy()
        denom = {c: max(int(g[c].notna().sum()), 1) for c in dhcols}
        rk = pd.DataFrame({c: g[c].rank(ascending=True) / denom[c] for c in dhcols})
        g["mr"] = rk.mean(axis=1, skipna=True)
        prep[dt] = g.sort_values(["mr", "amt_yi"], ascending=[True, False])

    seqs = {}
    skips = {}
    for C in (0, 5, 10, 15, 20, 30):
        last, v, sk = {}, {}, 0
        for dt in sorted(prep):
            g = prep[dt]
            if C > 0:
                ok = [(c not in last) or (dayrank[dt] - last[c] >= C) for c in g["ts_code"]]
                cand = g[np.array(ok)]
                if len(cand):
                    g = cand
            r = g.iloc[0]
            if pd.isna(r["mr"]):
                continue
            if not bool(r["n1_ok"]) or pd.isna(r["y"]):
                v[dt] = 0.0
                sk += 1
                continue
            v[dt] = float(r["y"]) - COST
            last[r["ts_code"]] = dayrank[dt]
        seqs[C] = pd.Series(v).sort_index()
        skips[C] = sk

    b = metrics(base)
    print(f"基准 年化{b[0]:.2f} 回撤{b[1]:.2f} 卡玛{b[2]:.2f}")
    print(f"\n{'冷却':>4}{'仓位':>6}{'年化':>9}{'回撤':>9}{'卡玛':>7}{'放弃':>5}")
    rows = []
    for C, v in seqs.items():
        for P in (0.30, 0.40, 0.50, 0.60, 0.70, 0.85, 1.00):
            sim = base.copy()
            sim[v.index] += v.values * P
            a, dd, cm = metrics(sim)
            rows.append({"C": C, "P": P, "年化": a, "回撤": dd, "卡玛": cm, "放弃": skips[C]})
            if P in (0.50, 0.70, 1.00):
                print(f"{C:>4}{P:>6.0%}{a:>9.2f}{dd:>9.2f}{cm:>7.2f}{skips[C]:>5}")
    g = pd.DataFrame(rows)
    g.to_csv(OUT / "w244_live_grid.csv", index=False, encoding="utf-8-sig")
    print("\n卡玛面(行=冷却, 列=仓位):")
    print(g.pivot_table(index="C", columns="P", values="卡玛").round(2).to_string())

    # ---- 回撤定位 ----
    v = seqs[10]
    sim = base.copy()
    sim[v.index] += v.values * 1.00
    nav = (1 + sim / 100).cumprod()
    peak = nav.cummax()
    ddn = nav / peak - 1
    trough = ddn.idxmin()
    pk = nav.loc[:trough].idxmax()
    print(f"\n满仓最大回撤 {ddn.min() * 100:.2f}%  峰 {pk} -> 谷 {trough}")
    seg = sim.loc[pk:trough]
    print(f"  区间内补位成交 {int((seg != 0).sum())} 天, 合计 {seg.sum():.1f}pp")
    print(f"  区间内主策略 {int(((seg != 0) & (~seg.index.isin(v.index))).sum())} 天, "
          f"合计 {sim.drop(v.index, errors='ignore').loc[pk:trough].sum():.1f}pp")

    # 若把补位日的收益单独看
    yrs = pd.DataFrame({"yr": v.index // 10000, "y": v.values})
    print("\n分年(集成 C=10, 单笔口径):")
    gg = yrs.groupby("yr").agg(信号=("y", "size"), 放弃=("y", lambda x: (x == 0).sum()),
                               单笔=("y", lambda x: x[x != 0].mean()),
                               合计=("y", "sum"))
    print(gg.round(2).to_string())


if __name__ == "__main__":
    main()
