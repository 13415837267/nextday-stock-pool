"""第 204 轮: 在**宽池**(22.9 万个股票日)上扫描门槛 —— 只有跳出旧硬条件,
才可能发现"该改的选股条件"。

旧硬条件(pct -6~-3 / close_pos<=0.32 / mom20 5~25 / dist_low20>=6 / vr<=4)是
第一百多轮手调出来的, 从没被全史样本检验过"切点是否落在最优处"。上一版在硬条件
池内扫描时, 一半参数改了笔数都不变(因为门槛不 binding), 本轮改在宽池上做。

指标用**日度等权**(每天算一个池均值再平均), 与引擎"每天最多买 1 只"一致;
样本加权会被"信号极多的普跌日"主导, 得出相反结论。

输出 output/gate_wide_204.csv / .json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
COST = 0.30
SPLIT = 2018


def load() -> pd.DataFrame:
    d = pd.read_parquet(OUT / "panel_train_204.parquet")
    bad = (d["ret"].abs() > 25) | (d["gap"].abs() > 15)
    d = d[~bad].reset_index(drop=True)
    d["year"] = d["d"].str[:4].astype("int16")
    return d


def stat(m: pd.DataFrame, label: str) -> dict:
    if len(m) < 200:
        return {}
    day = m.groupby("d", sort=False)["ret"].mean() - COST
    early = m[m.year < SPLIT].groupby("d", sort=False)["ret"].mean() - COST
    late = m[m.year >= SPLIT].groupby("d", sort=False)["ret"].mean() - COST
    return {"label": label, "n": int(len(m)), "days": int(day.shape[0]),
            "day_mean": round(float(day.mean()), 3),
            "ann_est": round(float(day.mean() * 250), 1),
            "win_days": round(float((day > 0).mean() * 100), 1),
            "early": round(float(early.mean()), 3) if len(early) > 100 else None,
            "late": round(float(late.mean()), 3) if len(late) > 100 else None,
            "per_year": round(len(m) / max(m.d.nunique(), 1), 1)}


def main() -> None:
    d = load()
    print(f"宽池 {len(d):,} 行 / {d.d.nunique()} 日", flush=True)
    rows = []
    # 单因子分档: 看每个维度的整条曲线, 而不是只看 3 个点
    grids = {
        "pct": np.arange(-9.0, -0.9, 0.5),
        "close_pos": np.arange(0.0, 0.56, 0.05),
        "mom20": np.arange(0, 33, 2.5),
        "dist_low20": [1, 2, 3, 4, 6, 8, 10, 13, 16, 20, 25, 30, 40, 70],
        "vr": [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0],
        "excess": np.arange(-10.0, 1.1, 1.0),
        "mkt_ret": [-9, -4, -3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 2, 9],
        "amp_pct": [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 20],
        "turnover_rate": [2, 2.5, 3, 3.5, 4, 5, 6, 8, 10, 12],
        "amt_yi": [0.8, 1.2, 2, 3, 5, 8, 12, 20, 1e9],
        "price": [3, 5, 8, 12, 18, 25, 40, 60, 150],
        "low_tail": [-3, -1, -0.5, 0, 0.2, 0.4, 0.6, 1.0, 2.0],
    }
    for col, edges in grids.items():
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            m = d[(d[col] >= lo) & (d[col] < hi)]
            st = stat(m, f"{col}∈[{lo:g},{hi:g})")
            if st:
                st["factor"], st["lo"], st["hi"] = col, float(lo), float(hi)
                rows.append(st)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "gate_wide_204.csv", index=False)

    for col in grids:
        x = df[df.factor == col]
        if x.empty:
            continue
        print(f"\n== {col} ==")
        print(x[["lo", "hi", "n", "per_year", "day_mean", "ann_est", "win_days", "early", "late"]]
              .to_string(index=False, float_format=lambda v: f"{v:g}"))

    # 现行硬条件 vs 宽池
    base = d[d.hard == 1]
    js = {"宽池": stat(d, "wide"), "现行硬条件": stat(base, "hard")}
    print("\n宽池:", json.dumps(js["宽池"], ensure_ascii=False))
    print("现行:", json.dumps(js["现行硬条件"], ensure_ascii=False))
    (OUT / "gate_wide_204.json").write_text(json.dumps(js, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
