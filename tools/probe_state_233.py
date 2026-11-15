"""第233轮 · 把 559 笔按下单时的"市场状态"分箱：什么时候这套逻辑有效

单一变量分箱用五分位（559 笔 / 5 ≈ 112 笔一档），同时给出 pearson 与 t 值，
避免第 232 轮那种"单档尖峰当规律"的误判。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "w233_probe.csv"

VARS = ["up_ratio", "limit_up", "limit_dn", "med_pct", "med_amp", "amt_yi",
        "amt_ratio", "breadth", "vol20", "ret20", "dd60", "dn_streak"]


def q5(d: pd.DataFrame, col: str) -> tuple[float, float, list[float], list[int]]:
    v = d[col].astype("float64")
    r = d["ret"].astype("float64")
    ok = v.notna() & r.notna()
    if ok.sum() < 50:
        return np.nan, np.nan, [], []
    x, y = v[ok], r[ok]
    pr = float(np.corrcoef(x, y)[0, 1])
    t = pr * np.sqrt((len(x) - 2) / max(1e-12, 1 - pr ** 2))
    bins = pd.qcut(x, 5, labels=False, duplicates="drop")
    means = [float(y[bins == i].mean()) for i in range(bins.max() + 1)]
    cnts = [int((bins == i).sum()) for i in range(bins.max() + 1)]
    return pr, t, means, cnts


def main() -> None:
    tr = pd.read_csv(ROOT / "output" / "w232_trades.csv", dtype={"signal": str})
    st = pd.read_csv(ROOT / "output" / "w233_state.csv", dtype={"date": str})
    d = tr.merge(st.rename(columns={"date": "signal"}), on="signal", how="left")
    print(f"559 笔中匹配到状态 {d['n'].notna().sum()} 笔；单笔均值 {d['ret'].mean():.3f}%\n")

    rows = []
    for c in VARS:
        pr, t, means, cnts = q5(d, c)
        rows.append({"变量": c, "pearson": round(pr, 3), "t": round(t, 2),
                     **{f"Q{i+1}": round(m, 2) for i, m in enumerate(means)},
                     "极差": round(max(means) - min(means), 2) if means else np.nan,
                     "笔数": sum(cnts)})
    r = pd.DataFrame(rows)
    print("== 市场状态五分位（单笔 %）==")
    print(r.to_string(index=False))

    # 月历
    print("\n== 星期 ==")
    print(d.groupby("dow")["ret"].agg(["count", "mean", "median"]).round(3).to_string())
    print("\n== 月内旬 ==")
    d["xun"] = pd.cut(d["dom"], [0, 10, 20, 31], labels=["上旬", "中旬", "下旬"])
    print(d.groupby("xun", observed=True)["ret"].agg(["count", "mean"]).round(3).to_string())

    # 序列依赖：上一笔 / 上两笔 / 上三笔的结果
    d = d.sort_values("signal").reset_index(drop=True)
    for k in (1, 2, 3):
        d[f"prev{k}_mean"] = d["ret"].shift(1).rolling(k).mean()
        d[f"prev{k}_neg"] = (d["ret"].shift(1) < 0)
    print("\n== 上一笔盈亏 -> 本笔 ==")
    d["prev1_win"] = np.where(d["prev1_mean"] > 0, "上笔盈利", "上笔亏损")
    print(d.groupby("prev1_win")["ret"].agg(["count", "mean"]).round(3).to_string())

    print("\n== 上两笔平均 -> 本笔（三分位） ==")
    d["p2"] = pd.qcut(d["prev2_mean"], 3, labels=["上两笔差", "中", "上两笔好"], duplicates="drop")
    print(d.groupby("p2", observed=True)["ret"].agg(["count", "mean", "median"]).round(3).to_string())

    print("\n== 上三笔平均 -> 本笔（三分位） ==")
    d["p3"] = pd.qcut(d["prev3_mean"], 3, labels=["上三笔差", "中", "上三笔好"], duplicates="drop")
    print(d.groupby("p3", observed=True)["ret"].agg(["count", "mean", "median"]).round(3).to_string())

    # 累计盈亏连亏长度
    streak, arr = 0, []
    for v in d["ret"].values:
        streak = streak + 1 if v < 0 else 0
        arr.append(streak)
    d["loss_streak"] = arr
    print("\n== 入场前已连亏 N 笔（N 为本笔之前） ==")
    print(d.groupby(d["loss_streak"].shift(1).fillna(0).clip(0, 4))["ret"]
          .agg(["count", "mean"]).round(3).to_string())

    r.to_csv(OUT, index=False, encoding="utf-8-sig")
    d.to_csv(ROOT / "output" / "w233_joined.csv", index=False, encoding="utf-8-sig")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
