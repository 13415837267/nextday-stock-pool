"""第246轮 C: 245 定稿复现(带 code) + 「按破位深度分级上仓」

246a 的分档给出一个很干净的东西:
    指数距 20 日线  <-8%   单笔 3.58%
                    -8~-6        2.13%
                    -6~-5        3.47%
                    -5~-4.5      2.78%
                    -4.5~-4      1.66%
                    -4~-3.5      0.98%
最浅那档只有最深那档的 1/4, 而且 44 笔(占 22%)。这是 T 日收盘就完全知道的量。

于是有两件事可试:
  (a) 收紧门槛 -3.51 -> -4.0, 砍掉最浅一档
  (b) 不做二值门槛, 做阶梯仓位: 越深越重

注意 (b) 是「用已知信息调仓位」, 不是「按结果调仓位」, 不存在未来信息。
但要防过拟合: 阶梯的切点和仓位必须用三段分年检验, 每段都得是同向。
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


def build() -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    tr = pd.read_csv(OUT / "w232_trades.csv")
    all_dates = sorted(int(p.stem.replace(".csv", "")) for p in
                       (ROOT / "data" / "panel").glob("*.csv.gz"))
    cal = np.array([d for d in all_dates if d >= tr.signal.min()])
    base = pd.Series(0.0, index=cal)
    for dt, g in tr.groupby("signal"):
        base[int(dt)] = g["ret"].mean()

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
    den = w.groupby("trade_date")[dhcols].transform("count")
    w["mr"] = (w[dhcols].rank(ascending=True) / den).mean(axis=1, skipna=True)
    w = w.sort_values(["trade_date", "mr", "amt_yi"], ascending=[True, True, False])
    return w, base, cal


def pick(w: pd.DataFrame) -> pd.DataFrame:
    """低开组里 mr 最小, 冷却 10; 冷却后无可选则退回不排除。"""
    dayrank = {int(d): i for i, d in enumerate(cal_g)}
    last: dict[str, int] = {}
    rec = []
    for dt, g in w.groupby("trade_date"):
        lo = g[g.n1_ok]
        cand = lo if len(lo) else g
        ok = [(c not in last) or (dayrank[int(dt)] - last[c] >= 10)
              for c in cand["ts_code"]]
        sub = cand[np.array(ok)]
        if len(sub):
            cand = sub
        r = cand.iloc[0]
        if pd.isna(r["mr"]) or pd.isna(r["y"]):
            continue
        last[r["ts_code"]] = dayrank[int(dt)]
        rec.append({"signal": int(dt), "code": r["ts_code"], "y": float(r["y"]) - COST,
                    "mr": float(r["mr"]), "amt_yi": float(r["amt_yi"])})
    return pd.DataFrame(rec)


def main() -> None:
    global cal_g
    w, base, cal = build()
    cal_g = cal
    d = pick(w)
    print(f"复现 245: {len(d)} 笔  单笔 {d.y.mean():.3f}%  合计 {d.y.sum():.0f}pp")

    mk = pd.read_csv(OUT / "feat234" / "market.csv")
    mk = mk[["trade_date", "idx_ma20_dist", "breadth", "limit_dn", "med_pct",
             "amt_shrink"]]
    d = d.merge(mk, left_on="signal", right_on="trade_date", how="left")
    d["yr"] = d.signal // 10000
    d.to_csv(OUT / "w246c_picks.csv", index=False, encoding="utf-8-sig")

    def sim_of(vals: np.ndarray, idx: np.ndarray) -> tuple[float, float, float]:
        s = pd.Series(vals, index=idx).sort_index()
        sim = base.copy()
        sim[s.index] += s.values
        return metrics(sim)

    print("\n【1】245 定稿的基线（含滑点 0.2, 仓位 50% 折算进 vals）")
    a, dd, cm = sim_of(d.y.values * 0.5, d.signal.values)
    print(f"  年化{a:>7.2f}  回撤{dd:>7.2f}  卡玛{cm:.2f}")

    print("\n【2】阶梯仓位: 仓位 = f(破位深度), 滑点 0.2")
    tiers = [
        ("三档 50/70/100", lambda x: 1.00 if x <= -6 else (0.70 if x <= -4.5 else 0.50)),
        ("三档 40/70/100", lambda x: 1.00 if x <= -6 else (0.70 if x <= -4.5 else 0.40)),
        ("两档 70/100", lambda x: 1.00 if x <= -5 else 0.70),
        ("两档 50/100", lambda x: 1.00 if x <= -5 else 0.50),
        ("四档 40/55/75/100", lambda x: 1.00 if x <= -7 else (
            0.75 if x <= -5.5 else (0.55 if x <= -4.5 else 0.40))),
        ("平 70", lambda x: 0.70),
        ("平 50", lambda x: 0.50),
    ]
    rows = []
    for name, f in tiers:
        P = d.idx_ma20_dist.map(f)
        a, dd, cm = sim_of((d.y.values - 0.2) * P.values, d.signal.values)
        print(f"  {name:<18} 年化{a:>7.2f}  回撤{dd:>7.2f}  卡玛{cm:>6.2f}  "
              f"平均仓位{P.mean():.0%}")
        rows.append({"方案": name, "年化": a, "回撤": dd, "卡玛": cm,
                     "平均仓位": round(P.mean(), 3)})

    print("\n【3】分段检验: 上面最好的阶梯, 三段各自是否同向")
    f3 = tiers[0][1]
    d["P3"] = d.idx_ma20_dist.map(f3)
    for lo, hi, tag in ((2016, 2019, "2016-2019"), (2020, 2022, "2020-2022"),
                        (2023, 2026, "2023-2026")):
        s = d[(d.yr >= lo) & (d.yr <= hi)]
        r_same = sim_of((s.y.values - 0.2) * s.P3.values, s.signal.values)
        r_flat = sim_of((s.y.values - 0.2) * 0.70, s.signal.values)
        r_50 = sim_of((s.y.values - 0.2) * 0.50, s.signal.values)
        print(f"  {tag}  n={len(s):>3}  阶梯{r_same[2]:>6.2f}  平70{r_flat[2]:>6.2f}"
              f"  平50{r_50[2]:>6.2f}   单笔{s.y.mean():>6.2f}%")

    print("\n【4】门槛收紧: 把最浅一档剔除")
    for cut in (-3.5, -3.75, -4.0, -4.25, -4.5):
        s = d[d.idx_ma20_dist <= cut]
        a, dd, cm = sim_of((s.y.values - 0.2) * 0.70, s.signal.values)
        print(f"  门槛 {cut:>5.2f}  n={len(s):>3}  单笔{s.y.mean():>6.2f}%  "
              f"年化{a:>7.2f}  回撤{dd:>7.2f}  卡玛{cm:>6.2f}")

    print("\n【5】分年明细 (245 定稿, 平50)")
    g = d.groupby("yr").agg(n=("y", "size"), 单笔=("y", "mean"), 合计=("y", "sum"),
                            最深=("idx_ma20_dist", "min"))
    g["胜率"] = d.groupby("yr").y.apply(lambda x: (x > 0).mean() * 100).round(0)
    print(g.round(2).to_string())
    pd.DataFrame(rows).to_csv(OUT / "w246c_tiers.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
