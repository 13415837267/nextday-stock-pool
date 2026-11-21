"""第263轮: 主策略 edge 的时间衰减与剩余行情维度审计

样本: output/w262_trades.csv (714 笔已平仓交易, 2014-07 ~ 2026-11)
核心问题: 单笔 edge(均值/胜率) 是否随时间衰减 —— 直接决定"继续跑还是改定位"。
副验证: T+1 隔夜缺口 gap_pct 分档是否还有未捕获的预测力(现有 gap_above/gap_floor 已用它做仓位)。
        T+2 开盘卖 vs 收盘卖(用 sell_px_open / sell_px_close 复算, 验证第182轮"收盘卖更优")。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def mann_kendall(x: np.ndarray) -> dict:
    n = len(x)
    s = 0
    for i in range(n):
        for j in range(i + 1, n):
            s += int(np.sign(x[j] - x[i]))
    ties = 0
    vals, cnt = np.unique(x, return_counts=True)
    for c in cnt:
        ties += c * (c - 1) * (2 * c + 5)
    var_s = (n * (n - 1) * (2 * n + 5) - ties) / 18.0
    z = (s - np.sign(s)) / np.sqrt(var_s) if var_s > 0 else 0.0
    tau = s / (n * (n - 1) / 2.0)
    from math import erf, sqrt
    p = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
    return dict(s=int(s), z=round(float(z), 3), tau=round(float(tau), 4),
                p=round(float(p), 4))


def ols(x: np.ndarray, y: np.ndarray) -> dict:
    n = len(x)
    xm, ym = x.mean(), y.mean()
    sxx = ((x - xm) ** 2).sum()
    sxy = ((x - xm) * (y - ym)).sum()
    b = sxy / sxx if sxx > 0 else 0.0
    a = ym - b * xm
    resid = y - (a + b * x)
    dof = n - 2
    se_b = np.sqrt((resid ** 2).sum() / dof / sxx) if dof > 0 else np.nan
    t = b / se_b if se_b and se_b > 0 else 0.0
    from math import erf, sqrt
    p = 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))
    return dict(intercept=round(float(a), 4), slope_per_trade=round(float(b), 6),
                slope_per_1000=round(float(b * 1000), 3), t=round(float(t), 3),
                p=round(float(p), 4))


def quintiles(df: pd.DataFrame, col: str) -> list[dict]:
    d = df.dropna(subset=[col]).copy()
    d["q"] = pd.qcut(d[col], 5, labels=False, duplicates="drop")
    out = []
    for q, g in d.groupby("q"):
        out.append(dict(q=int(q) + 1, n=int(len(g)),
                        lo=round(float(g[col].min()), 2), hi=round(float(g[col].max()), 2),
                        mean_ret=round(float(g["ret_pct"].mean()), 3),
                        win=round(float((g["ret_pct"] > 0).mean() * 100), 1)))
    return out


def main() -> None:
    df = pd.read_csv(ROOT / "output/w262_trades.csv")
    df = df[df["ret_pct"].notna()].copy()
    df["ret_pct"] = df["ret_pct"].astype(float)
    df["buy_dt"] = pd.to_datetime(df["buy_date"].astype(str), format="%Y%m%d")
    df = df.sort_values(["buy_dt", "buy_date"]).reset_index(drop=True)
    df["year"] = df["buy_dt"].dt.year
    n = len(df)
    print(f"交易数 {n}  {df['buy_dt'].min().date()} ~ {df['buy_dt'].max().date()}", flush=True)

    r = df["ret_pct"].to_numpy(dtype=float)
    pf = df["pos_frac"].fillna(1.0).to_numpy(dtype=float)
    log_sum = float(np.sum(np.log1p(pf * r / 100)) * 100)

    res = dict(n=n, date_from=str(df["buy_dt"].min().date()),
               date_to=str(df["buy_dt"].max().date()),
               overall=dict(mean_ret=round(float(r.mean()), 3),
                            win=round(float((r > 0).mean() * 100), 1),
                            median=round(float(np.median(r)), 3),
                            log_sum_pct=round(log_sum, 1)))

    res["yearly"] = {int(y): dict(n=int(len(g)),
                                  mean_ret=round(float(g["ret_pct"].mean()), 3),
                                  win=round(float((g["ret_pct"] > 0).mean() * 100), 1))
                     for y, g in df.groupby("year")}

    res["mk"] = mann_kendall(r)
    res["ols"] = ols(np.arange(n, dtype=float), r)

    half = n // 2
    res["split_half"] = dict(
        first=dict(n=half, mean_ret=round(float(r[:half].mean()), 3),
                   win=round(float((r[:half] > 0).mean() * 100), 1)),
        second=dict(n=n - half, mean_ret=round(float(r[half:].mean()), 3),
                    win=round(float((r[half:] > 0).mean() * 100), 1)))

    early = df[df["year"] <= 2019]
    late = df[df["year"] >= 2020]
    res["split_period"] = dict(
        early=dict(n=len(early), mean_ret=round(float(early["ret_pct"].mean()), 3),
                   win=round(float((early["ret_pct"] > 0).mean() * 100), 1)),
        late=dict(n=len(late), mean_ret=round(float(late["ret_pct"].mean()), 3),
                  win=round(float((late["ret_pct"] > 0).mean() * 100), 1)))

    W = 100
    roll = df["ret_pct"].rolling(W, min_periods=W).mean()
    res["rolling"] = [dict(d=str(d), mean=round(float(v), 3))
                      for d, v in zip(df["buy_dt"], roll) if pd.notna(v)]

    gap = df[df["gap_pct"].between(-30, 30)].copy()
    res["gap_quintiles"] = quintiles(gap, "gap_pct")
    res["gap_corr"] = dict(
        spearman=round(float(gap["gap_pct"].corr(gap["ret_pct"], method="spearman")), 4),
        pearson=round(float(gap["gap_pct"].corr(gap["ret_pct"])), 4))
    res["gap_n"] = int(len(gap))

    s = df[df["sell_px_open"].notna() & df["sell_px_close"].notna()].copy()
    s["open_ret"] = (s["sell_px_open"] / s["buy_px"] - 1) * 100
    s["close_ret"] = (s["sell_px_close"] / s["buy_px"] - 1) * 100
    drift = (s["close_ret"] - s["open_ret"]).to_numpy()
    res["sell"] = dict(n=int(len(s)),
                       open_ret=round(float(s["open_ret"].mean()), 3),
                       close_ret=round(float(s["close_ret"].mean()), 3),
                       drift_mean=round(float(drift.mean()), 3),
                       drift_median=round(float(np.median(drift)), 3),
                       n_close_better=int((drift > 0).sum()),
                       n_open_better=int((drift < 0).sum()))

    (ROOT / "output/exp_edge_263.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps({k: res[k] for k in
                      ("overall", "mk", "ols", "split_half", "split_period",
                       "gap_corr", "sell")},
                     ensure_ascii=False, indent=1), flush=True)


if __name__ == "__main__":
    main()
