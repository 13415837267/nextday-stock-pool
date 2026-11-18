"""第257轮 分析 v2: 控制变量后再挖条件

v1 发现的所有"截面条件"（pct / bias20 / dh60 / close / turnover）高度相关 ——
跌得多的票 pct 负、bias20 负、dh60 深、股价低, 四个面孔其实是同一件事。
这一版固定"补位实际会买的范围"（缺口 <= -5%）, 再逐条拆。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EV = pd.read_parquet(ROOT / "output/mine/events.parquet")
EV = EV[EV["ret_d2"].notna() & EV["gap"].notna() & (EV["buyable"] == 1)].copy()
EV["year"] = (EV["date"] // 10000).astype(int)


def stat(s: pd.Series) -> str:
    if not len(s):
        return "     -"
    return f"{len(s):>4}笔 {s.mean():>6.2f}% 胜{(s > 0).mean() * 100:>4.0f}%"


print(f"事件表 {len(EV):,} 行 / {EV['date'].nunique()} 天")

# ============ A. 深低开带内: 名次到底该不该跳过前 3 ============
print("\n" + "=" * 84)
print("A. 深低开带（-10% < 缺口 <= -4%）内, 按名次 x 指数状态")
print("=" * 84)
d = EV[(EV.gap > -10) & (EV.gap <= -4)].copy()
d["grp"] = np.where(d.layer <= 3, "1-3名", np.where(d.layer <= 6, "4-6名", "7-8名"))
d["mk"] = np.where(d.dist20 <= -5, "破位日(dist<=-5)", "普通日")
print(f"{'':<16}{'1-3名':<26}{'4-6名':<26}{'7-8名':<26}")
for mk in ("破位日(dist<=-5)", "普通日"):
    cells = []
    for g in ("1-3名", "4-6名", "7-8名"):
        s = d[(d.mk == mk) & (d.grp == g)]["ret_d2"]
        cells.append(f"{stat(s):<26}")
    print(f"{mk:<16}" + "".join(cells))

print("\n分缺口档 x 名次:")
for lo, hi in [(-10, -8), (-8, -7), (-7, -6), (-6, -5), (-5, -4)]:
    cells = []
    for g in ("1-3名", "4-6名"):
        s = d[(d.gap > lo) & (d.gap <= hi) & (d.grp == g)]["ret_d2"]
        cells.append(f"{stat(s):<26}")
    print(f"  {lo}~{hi}{'':<6}" + "".join(cells))

# ============ B. 深低开带内, 加当日跌幅条件 ============
print("\n" + "=" * 84)
print("B. 深低开带内（4-6 名, 即现行 deep_skip=3 的口径）加当日跌幅条件")
print("=" * 84)
b = EV[(EV.gap > -10) & (EV.gap <= -4) & (EV.layer >= 4) & (EV.layer <= 6)]
for lo, hi in [(-11.5, -9), (-9, -7), (-7, -5), (-5, -3), (-3, 11.5)]:
    s = b[(b.pct > lo) & (b.pct <= hi)]["ret_d2"]
    print(f"  当日 {lo:>5}% ~ {hi:>5}%   {stat(s)}")
print("\n  加换手率:")
for lo, hi in [(0, 3), (3, 6), (6, 100)]:
    s = b[(b.turnover > lo) & (b.turnover <= hi)]["ret_d2"]
    print(f"  换手 {lo:>3}% ~ {hi:>3}%      {stat(s)}")

# ============ C. 破位腿: 现状 -5/-42, 看能否加条件 ============
print("\n" + "=" * 84)
print("C. 破位腿（dist20<=-5 且 第1名 dh60<=-42, 缺口<=0.05）拆条件")
print("=" * 84)
t1 = EV[EV.layer == 1].copy()
br = t1[(t1.dist20 <= -5) & (t1.dh60 <= -42) & (t1.gap <= 0.05)]
print(f"  合计 {stat(br['ret_d2'])}")
print("\n  按 dist20 深度:")
for lo, hi in [(-100, -8), (-8, -7), (-7, -6), (-6, -5)]:
    s = br[(br.dist20 > lo) & (br.dist20 <= hi)]["ret_d2"]
    print(f"    {lo}~{hi:<8}{stat(s)}")
print("\n  按第1名当日跌幅:")
for lo, hi in [(-11.5, -9), (-9, -7), (-7, -5), (-5, -3), (-3, 11.5)]:
    s = br[(br.pct > lo) & (br.pct <= hi)]["ret_d2"]
    print(f"    当日 {lo:>5}% ~ {hi:>5}%   {stat(s)}")
print("\n  按开盘缺口:")
for lo, hi in [(-100, -6), (-6, -3), (-3, 0.05)]:
    s = br[(br.gap > lo) & (br.gap <= hi)]["ret_d2"]
    print(f"    缺口 {lo}% ~ {hi}%   {stat(s)}")

# ============ D. 深低开带的"名次"是不是被跌幅带偏 ============
print("\n" + "=" * 84)
print("D. 深低开带内 1-3 名 vs 4-6 名: 当日跌幅分布对比（看是不是同一批票）")
print("=" * 84)
for g, m in ((d.layer <= 3, "1-3名"), (d.layer.between(4, 6), "4-6名")):
    s = d[g]
    print(f"  {m}: n={len(s)}, pct 中位 {s.pct.median():.2f}%, "
          f"dh60 中位 {s.dh60.median():.1f}%, amount 中位 {s.amount.median() / 1e8:.2f}亿, "
          f"换手中位 {s.turnover.median():.2f}%")

# ============ E. 空仓日: 极端日次日整体反弹基准 ============
print("\n" + "=" * 84)
print("E. 各类极端日, 候选池第 1 名（缺口<=0 才买）的收益 —— 对照基准")
print("=" * 84)
print(f"{'dist20 档':<14}{'第1名':<26}{'第1名且 dh60<=-42':<26}")
for lo, hi in [(-100, -8), (-8, -7), (-7, -6), (-6, -5), (-5, -4), (-4, -3), (-3, 0)]:
    s1 = t1[(t1.dist20 > lo) & (t1.dist20 <= hi) & (t1.gap <= 0.05)]["ret_d2"]
    s2 = t1[(t1.dist20 > lo) & (t1.dist20 <= hi) & (t1.gap <= 0.05) & (t1.dh60 <= -42)]["ret_d2"]
    print(f"{lo}~{hi:<10}{stat(s1):<26}{stat(s2):<26}")
