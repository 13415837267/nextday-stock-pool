"""第 241 轮: 把 W7 从"一次性样本外"升级为 walk-forward

240 轮的问题: 阈值 (-3.51, -24.46) 是用 2016~2020 一次挖定, 2021~2026 只验了一次。
这仍然是"一次切分", 只要切点选得巧就能好看。真正的检验是逐年滚动:
  测 t 年 => 只用 t 年以前的数据挑阈值 (候选网格 8x7=56 组), 原样套到 t 年。

两种挑法对比 (看目标函数本身是否可被事先知道):
  A 稳健: 让训练窗内"最差的一年"最高 (max-min)
  B 朴素: 让训练窗内年均收益最高 (max-mean)

另外必须回答: W7 的收益是 alpha 还是 beta? 破位日买跌得最透的票, 极可能只是
"大盘反弹时小票弹性大"。所以每天同时报 y 与 y_ex (扣当日全市场均值)。

输出: output/w241_wf.csv / w241_detail.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "output" / "feat234"
OUT = ROOT / "output"
COST = 0.10                 # 双边佣金+印花税, 227 轮口径
TRAIN_FROM = 2016
TEST_FROM = 2019
TEST_TO = 2026

IDX_GRID = [-2.0, -2.5, -3.0, -3.51, -4.0, -4.5, -5.0]
H60_GRID = [-15.0, -18.0, -21.0, -24.46, -28.0, -32.0, -36.0]

COLS = ["ts_code", "trade_date", "y", "n1_ok", "pct", "dist_high60",
        "circ_mv", "idx_dist", "is_paused", "is_st", "listed"]


def load() -> pd.DataFrame:
    parts = []
    for p in sorted(FEAT.glob("f*.parquet")):
        d = pd.read_parquet(p, columns=COLS)
        d = d[(d["is_paused"] == 0) & (d["is_st"] == 0) & (d["n1_ok"] == True)]      # noqa: E712
        d = d[d["pct"].between(-11.5, 11.5) & d["y"].between(-40, 60) & (d["listed"] >= 250)]
        d = d[d["ts_code"].str[:2].isin(["60", "00"])]
        d = d[d["dist_high60"] <= -10]          # 粗筛, 比最松网格(-15)还松
        if len(d):
            parts.append(d)
    return pd.concat(parts, ignore_index=True)


def year_ret(rets: pd.Series) -> float:
    return (np.prod(1 + rets / 100) - 1) * 100


def pick_one(sub: pd.DataFrame) -> float:
    """按流通市值最小取 1 只, 返回扣成本后的收益%"""
    return float(sub.nsmallest(1, "circ_mv")["y"].iloc[0]) - COST


def evaluate(d: pd.DataFrame, i_cut: float, h_cut: float, years: list[int]) -> pd.DataFrame:
    c = d[(d["idx_dist"] <= i_cut) & (d["dist_high60"] <= h_cut)]
    c = c[c["yr"].isin(years)]
    if not len(c):
        return pd.DataFrame(columns=["trade_date", "y", "y_ex", "yr"])
    rows = []
    for dt, g in c.groupby("trade_date"):
        r = g.nsmallest(1, "circ_mv").iloc[0]
        rows.append({"trade_date": dt, "y": r["y"] - COST, "y_ex": r["y_ex"] - COST,
                     "yr": r["yr"]})
    return pd.DataFrame(rows)


def main() -> None:
    d = load()
    d["yr"] = d["trade_date"] // 10000
    d = d[d["yr"] >= TRAIN_FROM]

    # 当日全市场均值 (y_ex 口径): 需要全样本, 粗筛前的均值会偏, 故单独读 y 一列
    parts = []
    for p in sorted(FEAT.glob("f*.parquet")):
        x = pd.read_parquet(p, columns=["trade_date", "y", "n1_ok", "pct", "is_paused", "is_st"])
        x = x[(x["is_paused"] == 0) & (x["is_st"] == 0) & (x["n1_ok"] == True)]      # noqa: E712
        x = x[x["pct"].between(-11.5, 11.5) & x["y"].between(-40, 60)]
        if len(x):
            parts.append(x[["trade_date", "y"]])
    full = pd.concat(parts, ignore_index=True)
    mu = full.groupby("trade_date")["y"].mean()
    d["y_ex"] = d["y"] - d["trade_date"].map(mu)

    print(f"样本 {len(d):,} 行  触发候选(dist<=-10) 覆盖 {d.trade_date.nunique()} 日")
    print(f"idx_dist 全期 min={d.idx_dist.min():.2f}  "
          f"<=-3.51 的天数={d[d.idx_dist <= -3.51].trade_date.nunique()}")

    res, detail = [], []
    for ty in range(TEST_FROM, TEST_TO + 1):
        train_years = list(range(TRAIN_FROM, ty))
        best = {"A": None, "B": None}
        for ic in IDX_GRID:
            for hc in H60_GRID:
                e = evaluate(d, ic, hc, train_years)
                if len(e) < 20:
                    continue
                ys = {y: year_ret(e[e.yr == y]["y"]) for y in train_years}
                vals = list(ys.values())
                worst = min(vals)
                mean = float(np.mean(vals))
                npos = int(sum(v > 0 for v in vals))
                rec = {"ic": ic, "hc": hc, "worst": worst, "mean": mean,
                       "npos": npos, "n": len(e)}
                # A: 最差年最高, 同年份平手看正年数->均值
                ka = (worst, npos, mean)
                if best["A"] is None or ka > best["A"][0]:
                    best["A"] = (ka, rec)
                # B: 年均最高
                kb = (mean, worst)
                if best["B"] is None or kb > best["B"][0]:
                    best["B"] = (kb, rec)

        for tag in ("A", "B"):
            rec = best[tag][1]
            e = evaluate(d, rec["ic"], rec["hc"], [ty])
            if not len(e):
                res.append({"year": ty, "target": tag, "ic": rec["ic"], "hc": rec["hc"],
                            "train_worst": round(rec["worst"], 1),
                            "train_npos": rec["npos"], "n": 0, "mean": np.nan,
                            "year_ret": 0.0, "alpha": np.nan, "win": np.nan})
                continue
            res.append({"year": ty, "target": tag, "ic": rec["ic"], "hc": rec["hc"],
                        "train_worst": round(rec["worst"], 1), "train_npos": rec["npos"],
                        "n": len(e), "mean": round(e["y"].mean(), 3),
                        "year_ret": round(year_ret(e["y"]), 2),
                        "alpha": round(e["y_ex"].sum(), 1),
                        "win": round((e["y"] > 0).mean() * 100, 1)})
            detail.append(e.assign(target=tag))

    r = pd.DataFrame(res)
    print("\n" + "=" * 96)
    print("逐年 walk-forward（每年只用此前数据挑阈值）")
    for tag in ("A", "B"):
        s = r[r.target == tag]
        print(f"\n--- {tag} 组 ({'max-min 稳健' if tag == 'A' else 'max-mean 朴素'})")
        print(s[["year", "ic", "hc", "train_worst", "train_npos", "n", "mean",
                 "year_ret", "alpha", "win"]].to_string(index=False))
        print(f"合计 {int(s['n'].sum())} 笔  单笔 {s['mean'].mean():+.3f}%  "
              f"年化 {round((np.prod(1 + s['year_ret'] / 100) ** (1 / len(s)) - 1) * 100, 2)}%  "
              f"正年 {int((s['year_ret'] > 0).sum())}/{len(s)}  "
              f"最差年 {s['year_ret'].min():.1f}%  alpha合计 {s['alpha'].sum():+.1f}pp")

    r.to_csv(OUT / "w241_wf.csv", index=False, encoding="utf-8-sig")
    if detail:
        pd.concat(detail, ignore_index=True).to_csv(
            OUT / "w241_detail.csv", index=False, encoding="utf-8-sig")

    # 固定阈值对照 (240 轮的产物, 从未重挑)
    print("\n" + "=" * 96)
    print("对照: 240 轮固定阈值 (-3.51, -24.46) 同一测试段逐年")
    e = evaluate(d, -3.51, -24.46, list(range(TEST_FROM, TEST_TO + 1)))
    for y in range(TEST_FROM, TEST_TO + 1):
        s = e[e.yr == y]
        if not len(s):
            print(f"  {y}: 0 笔")
            continue
        print(f"  {y}: {len(s):>3} 笔  单笔 {s.y.mean():+.2f}%  年 {year_ret(s.y):+6.1f}%  "
              f"alpha {s.y_ex.sum():+6.1f}pp  胜率 {(s.y > 0).mean() * 100:.0f}%")
    print(f"  合计 {len(e)} 笔 单笔 {e.y.mean():+.3f}%  "
          f"alpha合计 {e.y_ex.sum():+.1f}pp  alpha均值 {e.y_ex.mean():+.3f}%")


if __name__ == "__main__":
    main()
