"""第257轮: 在事件表上挖条件（全历史 2009-2026）

读 output/mine/events.parquet（一行 = 一个交易日的一层候选），做五件事：
  1. 破位腿: dist20 阈值 x 第1名距60日高 阈值 的二维扫描
  2. 深低开腿: 开盘缺口带的收益曲线（现在的 -8%~-5% 是不是最优）
  3. 层级: 第 1..8 层各自的收益（漏斗到底该在哪一层成交）
  4. 标的条件: 当日跌幅 / 成交额 / 换手 / 距20日线 / 上市天数 / 股价
  5. 分年稳定性

收益口径: T+1 开盘买 -> T+2 收盘卖（与回测同口径, 未扣滑点）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EV = pd.read_parquet(ROOT / "output/mine/events.parquet")
EV = EV[EV["ret_d2"].notna() & EV["gap"].notna() & (EV["buyable"] == 1)].copy()
EV["year"] = (EV["date"] // 10000).astype(int)


def stat(s: pd.Series) -> str:
    if not len(s):
        return "      -"
    return f"{len(s):>4}笔 {s.mean():>6.2f}% 胜{(s > 0).mean() * 100:>4.0f}% 中位{s.median():>6.2f}%"


def show(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main() -> None:
    print(f"事件表 {len(EV):,} 行 / {EV['date'].nunique()} 天 / "
          f"{EV['year'].min()}-{EV['year'].max()}")

    # ---------- 1. 破位腿二维扫描 ----------
    show("1. 破位腿: 指数距20日线 x 第1名距60日高（两层都满足才买, 次日开盘就买）")
    t1 = EV[EV["layer"] == 1].copy()
    bins_d = [-100, -9, -8, -7, -6, -5, -4, -3]
    bins_h = [-100, -60, -50, -42, -35, -30, -20, 100]
    t1["d_b"] = pd.cut(t1["dist20"], bins_d)
    t1["h_b"] = pd.cut(t1["dh60"], bins_h)
    piv_n = t1.pivot_table(index="d_b", columns="h_b", values="ret_d2", aggfunc="size", observed=True)
    piv_r = t1.pivot_table(index="d_b", columns="h_b", values="ret_d2", aggfunc="mean", observed=True)
    print("\n笔数:")
    print(piv_n.fillna(0).astype(int).to_string())
    print("\n单笔均值 %:")
    print(piv_r.round(2).to_string())

    print("\n阈值组合（dist20<=X 且 dh60<=Y）:")
    print(f"{'X':>6}{'Y':>7}{'笔数':>7}{'单笔':>9}{'胜率':>7}{'年化天数/年':>11}")
    for x in (-4, -5, -6, -7, -8):
        for y in (-30, -35, -42, -50, -60):
            s = t1[(t1.dist20 <= x) & (t1.dh60 <= y)]["ret_d2"]
            if len(s) < 8:
                print(f"{x:>6}{y:>7}{len(s):>7}   样本不足")
                continue
            yrs = max(t1["year"].nunique(), 1)
            print(f"{x:>6}{y:>7}{len(s):>7}{s.mean():>8.2f}%{(s > 0).mean() * 100:>6.0f}%"
                  f"{len(s) / yrs:>10.1f}")

    # ---------- 2. 深低开缺口带 ----------
    show("2. 深低开腿: 开盘缺口带的单笔收益（深低开不要求指数条件）")
    print(f"{'缺口区间':<16}{'统计'}")
    for lo, hi in [(-100, -10), (-10, -8), (-8, -7), (-7, -6), (-6, -5), (-5, -4),
                   (-4, -2), (-2, 0), (0, 3), (3, 100)]:
        s = EV[(EV.gap > lo) & (EV.gap <= hi)]["ret_d2"]
        print(f"{f'{lo}% ~ {hi}%':<16}{stat(s)}")
    print("\n仅第 1-3 名（破位名单）:")
    for lo, hi in [(-100, -8), (-8, -5), (-5, 0), (0, 100)]:
        s = EV[(EV.layer <= 3) & (EV.gap > lo) & (EV.gap <= hi)]["ret_d2"]
        print(f"{f'{lo}% ~ {hi}%':<16}{stat(s)}")
    print("\n仅第 4-8 名（深低开名单）:")
    for lo, hi in [(-100, -8), (-8, -5), (-5, 0), (0, 100)]:
        s = EV[(EV.layer >= 4) & (EV.gap > lo) & (EV.gap <= hi)]["ret_d2"]
        print(f"{f'{lo}% ~ {hi}%':<16}{stat(s)}")

    # ---------- 3. 层级 ----------
    show("3. 层级: 各层可成交时（缺口 <= 0）的单笔收益")
    for k in range(1, 9):
        s = EV[(EV.layer == k) & (EV.gap <= 0)]["ret_d2"]
        print(f"第 {k} 层{'':<10}{stat(s)}")

    # ---------- 4. 标的条件 ----------
    show("4. 标的条件（缺口 <= 0 的全部层）")
    d = EV[EV.gap <= 0].copy()

    def by(col, bins, labels=None):
        b = pd.cut(d[col], bins, labels=labels)
        g = d.groupby(b, observed=True)["ret_d2"].agg(["size", "mean", lambda x: (x > 0).mean() * 100])
        g.columns = ["n", "avg", "win"]
        print(f"\n{d[col].name if hasattr(d[col], 'name') else col}:")
        for i, r in g.iterrows():
            print(f"  {str(i):<18}{int(r.n):>5}笔 {r.avg:>6.2f}% 胜{r.win:>4.0f}%")

    by("pct", [-11.5, -9, -7, -5, -3, 0, 100])
    by("amount", [0, 2e8, 5e8, 1e9, 3e9, 1e12])
    by("turnover", [0, 2, 5, 10, 20, 100])
    by("bias20", [-100, -25, -18, -12, -6, 0, 100])
    by("bars", [250, 400, 700, 1200, 2500, 99999])
    by("close", [0, 5, 8, 12, 20, 40, 10000])
    by("dh60", [-100, -60, -50, -42, -35, -25, 0])

    # ---------- 5. 分年 ----------
    show("5. 分年稳定性（缺口 <= 0 的全部层）")
    g = d.groupby("year")["ret_d2"].agg(["size", "mean", lambda x: (x > 0).mean() * 100])
    g.columns = ["n", "avg", "win"]
    for i, r in g.iterrows():
        print(f"  {i}  {int(r.n):>5}笔 {r.avg:>7.2f}% 胜{r.win:>4.0f}%")


if __name__ == "__main__":
    main()
