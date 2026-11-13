#!/usr/bin/env python3
"""第222轮: T+0 ETF 的日内低开反转 —— 能不能替代可转债。

沿用第221轮的口径, 但品种换成无权限门槛的 T+0 ETF:
  开盘买入 -> 收盘卖出, 这一笔就是 ETF 当天真实的日内交易。

必须回答四个问题:
  Q1 各类别日内振幅有多大? 振幅小于成本就没得做。
  Q2 gap 分档有没有倒U形(第221轮在转债上是 -1%~-2% 的甜点)?
  Q3 隔夜对照(收->次开)是什么符号? 若隔夜同样为正, 说明日内那点钱只是噪声。
  Q4 成本吃掉多少? ETF 免印花税, 但低价 ETF 一个 tick 就是 0.1%。

幸存者偏差警告: 新浪只给在市 927 只, 已清盘 ETF 不在样本内。清盘 ETF 多是规模
萎缩、流动性枯竭的品种, 它们若被计入大概率只会拉低收益 —— 所以本轮的结论是
上界, 不是下限。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DAYS = 244.0


def load() -> pd.DataFrame:
    d = pd.read_parquet(ROOT / "data" / "etf" / "daily.parquet")
    d = d.sort_values(["code", "date"]).reset_index(drop=True)
    g = d.groupby("code", observed=True, sort=False)
    d["pc"] = g["close"].shift(1)
    d["gap"] = d["open"] / d["pc"] - 1          # T日开盘缺口
    d["intr"] = d["close"] / d["open"] - 1      # 日内: 开盘买收盘卖
    d["on"] = d["open"] / d["pc"] - 1           # 隔夜
    d["r1"] = d["close"] / d["pc"] - 1
    d["amp"] = (d["high"] - d["low"]) / d["pc"]  # 全日振幅
    d["vol20"] = g["r1"].transform(lambda x: x.rolling(20).std())
    # 只保留可交易样本: 有成交额、缺口在合理区间
    d = d.dropna(subset=["pc", "gap"])
    d = d[(d["gap"].between(-0.15, 0.15)) & (d["amount"] > 0)]
    return d


def stats(x: pd.Series) -> tuple[float, float, int]:
    x = x.dropna()
    if len(x) == 0:
        return np.nan, np.nan, 0
    return x.mean() * 100, (x > 0).mean() * 100, len(x)


def q1_amplitude(d: pd.DataFrame) -> None:
    print("=" * 96)
    print("Q1  各类别日内振幅 (close/open-1) 与日波动")
    print("=" * 96)
    rows = []
    for cls, g in d.groupby("cls"):
        m, w, n = stats(g["intr"])
        rows.append({
            "类别": cls, "行数": n, "只数": g["code"].nunique(),
            "日内均值bp": round(m * 100, 1) if pd.notna(m) else np.nan,
            "日内标准差bp": round(g["intr"].std() * 10000, 1),
            "日内胜率%": round(w, 1),
            "振幅中位%": round(g["amp"].median() * 100, 2),
            "日收益std%": round(g["r1"].std() * 100, 2),
            "中位价": round(g["close"].median(), 2),
        })
    print(pd.DataFrame(rows).sort_values("行数", ascending=False).to_string(index=False))


def q2_gap(d: pd.DataFrame, cuts: list[float]) -> None:
    print()
    print("=" * 96)
    print("Q2  gap 分档 -> 日内收益 (开盘买/收盘卖, 未扣成本)")
    print("=" * 96)
    for cls in ["qdii", "bond", "gold", "money", "stock"]:
        g = d[d["cls"] == cls]
        if len(g) < 5000:
            continue
        g = g.copy()
        g["bin"] = pd.cut(g["gap"], cuts)
        rows = []
        for b, gg in g.groupby("bin", observed=True):
            m, w, n = stats(gg["intr"])
            if n < 100:
                continue
            rows.append({"档": f"{b.left*100:.2f}~{b.right*100:.2f}%", "日内bp": round(m * 100, 1),
                         "胜率%": round(w, 1), "样本": n})
        print(f"\n--- {cls}  (n={len(g):,}) ---")
        print(pd.DataFrame(rows).to_string(index=False))


def q3_overnight(d: pd.DataFrame, cuts: list[float]) -> None:
    print()
    print("=" * 96)
    print("Q3  隔夜对照: 同一批低开样本, 日内 vs 隔夜(收->次开) 的符号")
    print("=" * 96)
    rows = []
    for cls in ["qdii", "bond", "gold", "money"]:
        g = d[d["cls"] == cls]
        if len(g) < 5000:
            continue
        sel = g[g["gap"].between(-0.02, -0.004)]
        mi, wi, ni = stats(sel["intr"])
        mo, wo, no = stats(sel["on"])
        rows.append({"类别": cls, "样本": ni,
                     "日内bp": round(mi * 100, 1), "日内胜率%": round(wi, 1),
                     "隔夜bp": round(mo * 100, 1), "隔夜胜率%": round(wo, 1)})
    print(pd.DataFrame(rows).to_string(index=False))


def q4_cost(d: pd.DataFrame) -> None:
    print()
    print("=" * 96)
    print("Q4  成本敏感性: 低价 ETF 一个 tick(0.001) 就是多少 bp")
    print("=" * 96)
    rows = []
    for p in [0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 100.0]:
        tick = 0.001 / p * 10000  # bp
        rows.append({"价格": p, "1tick(bp)": round(tick, 1),
                     "往返0.02%佣金+1tick": round(2 + tick, 1),
                     "往返0.02%佣金+2tick": round(2 + 2 * tick, 1)})
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n各价位 ETF 占比:")
    d2 = d.drop_duplicates("code")
    bins = pd.cut(d2.groupby("code")["close"].last(), [0, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 1e9])
    print(bins.value_counts().sort_index().to_string())


def main() -> None:
    d = load()
    print(f"面板 {len(d):,} 行 / {d['code'].nunique()} 只 / {d['date'].min().date()} ~ {d['date'].max().date()}")
    q1_amplitude(d)
    cuts = [-0.20, -0.03, -0.02, -0.015, -0.010, -0.007, -0.004, -0.002,
            0.0, 0.002, 0.004, 0.007, 0.010, 0.015, 0.020, 0.03, 0.20]
    q2_gap(d, cuts)
    q3_overnight(d, cuts)
    q4_cost(d)


if __name__ == "__main__":
    main()
