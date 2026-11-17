"""第245轮 D: T+1 重选方案的稳健面 + 与 244 的配对检验 + 流动性过滤

245C: T+1 开盘在低开票里重选, 冷却10, 仓位50% -> 138.1%/-25.15%/5.49
      (244 同回撤档是 70% 仓 123.0%/-23.59%/5.21)
本轮回答三个问题:
  ① 仓位 x 冷却 的面是不是平台(还是又挑出一个孤点)
  ② 与 244 方案逐日配对, 多赚的是不是真钱
  ③ 3 万本金下, 成交额太小的票买不进 —— 加上 amt>=0.5 亿 的流动性约束还剩多少
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
    w = sig.copy()
    for W in WINS:
        px = pd.read_parquet(POOL / f"W{W}.parquet").rename(columns={"dh": f"dh{W}"})
        w = w.merge(px, on=["ts_code", "trade_date"], how="left")
        del px
    dhcols = [f"dh{W}" for W in WINS]
    w = w.dropna(subset=dhcols)
    w["n1_ok"] = w["n1_ok"].fillna(False)
    w["y_net"] = w["y"] - COST

    def prep(g: pd.DataFrame) -> pd.DataFrame:
        den = {c: max(int(g[c].notna().sum()), 1) for c in dhcols}
        rk = pd.DataFrame({c: g[c].rank(ascending=True) / den[c] for c in dhcols})
        g = g.copy()
        g["mr"] = rk.mean(axis=1, skipna=True)
        return g

    w = pd.concat([prep(g) for _, g in w.groupby("trade_date")], ignore_index=True)
    byday = {dt: g for dt, g in w.groupby("trade_date")}

    def run(cd: int, min_amt: float, rule: str = "reallot"):
        last, v, pk = {}, {}, {}
        for dt in sorted(byday):
            g = byday[dt]
            g = g[g.amt_yi >= min_amt]
            if rule == "reallot":
                lo = g[g.n1_ok]
                g = lo if len(lo) else g
            cand = g.sort_values(["mr", "amt_yi"], ascending=[True, False])
            ok = [(c not in last) or (dayrank[dt] - last[c] >= cd) for c in cand["ts_code"]]
            sub = cand[np.array(ok)]
            if len(sub):
                cand = sub
            if not len(cand):
                continue
            r = cand.iloc[0]
            if pd.isna(r["mr"]) or pd.isna(r["y"]):
                continue
            if rule == "pick_then_skip" and not bool(r["n1_ok"]):
                v[dt] = 0.0
                continue
            v[dt] = float(r["y_net"])
            pk[dt] = r["ts_code"]
            last[r["ts_code"]] = dayrank[dt]
        return pd.Series(v).sort_index(), pd.Series(pk).sort_index()

    b = metrics(base)
    print(f"基准 {b}\n")

    print("【1】仓位 x 冷却 (T+1 重选, 无流动性约束)")
    print(f"{'冷却':>4}" + "".join(f"{p:>9.0%}" for p in (0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0)))
    grid = []
    for cd in (0, 5, 10, 15, 20, 30):
        v, _ = run(cd, 0.0)
        line = f"{cd:>4}"
        for P in (0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0):
            sim = base.copy()
            sim[v.index] += v.values * P
            a, dd, cm = metrics(sim)
            line += f"{cm:>9.2f}"
            grid.append({"C": cd, "P": P, "年化": a, "回撤": dd, "卡玛": cm})
        print(line)
    pd.DataFrame(grid).to_csv(OUT / "w245_grid.csv", index=False, encoding="utf-8-sig")

    print("\n【2】流动性约束 (成交额门槛, 亿元)")
    for amt in (0.0, 0.2, 0.5, 1.0):
        v, pk = run(10, amt)
        sim = base.copy()
        sim[v.index] += v.values * 0.5
        a, dd, cm = metrics(sim)
        print(f"  amt>={amt:>4.1f}亿  n={len(v):>4}  单笔{v.mean():>6.2f}%  "
              f"年化{a:>7.2f} 回撤{dd:>7.2f} 卡玛{cm:>5.2f}  票数{pk.nunique()}")

    print("\n【3】与 244 方案逐日配对 (同日各自选一只)")
    v245, _ = run(10, 0.0, "reallot")
    v244, _ = run(10, 0.0, "pick_then_skip")
    both = v244[v244 != 0].index
    d = v245.loc[both] - v244.loc[both]
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    print(f"  244 有成交的 {len(both)} 天里, 245 多赚 {d.mean():+.3f}pp/笔  "
          f"t={t:.2f}  245 胜率 {(d > 0).mean() * 100:.0f}%")
    only = v245.index.difference(both)
    print(f"  244 放弃、245 做成的那 {len(only)} 天: 单笔 {v245.loc[only].mean():.3f}%  "
          f"胜率 {(v245.loc[only] > 0).mean() * 100:.0f}%  合计 {v245.loc[only].sum():.1f}pp")

    print("\n【4】定稿候选的分年 (冷却10, 仓位50%)")
    p = pd.DataFrame({"yr": v245.index // 10000, "y": v245.values})
    g = p.groupby("yr").y.agg(["size", "sum", "mean"])
    g["win%"] = p.groupby("yr").y.apply(lambda x: (x > 0).mean() * 100).round(0)
    print(g.round(2).to_string())
    sim = base.copy()
    sim[v245.index] += v245.values * 0.5
    print(f"  合计: {metrics(sim)}")


if __name__ == "__main__":
    main()
