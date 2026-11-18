"""第257轮 分析 v3: 按"引擎实际会走到哪条腿"切样本

破位腿: dist20 <= X 且 当日第1名 dh60 <= Y, 次日开盘 gap <= 0.05 即买第1名。
深低开腿: 破位**不成立**的日子, 从名单第 k 层起, gap 落在带内才买。
           破位不成立 = dist20 > -5 或 第1名 dh60 > -42（与 plan() 一致）

这样切出来的样本才对应引擎的真实决策, 而不是"所有日子混在一起"。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EV = pd.read_parquet(ROOT / "output/mine/events.parquet")
EV = EV[EV["ret_d2"].notna() & EV["gap"].notna() & (EV["buyable"] == 1)].copy()
EV["year"] = (EV["date"] // 10000).astype(int)

# 日级: 第1名的 dh60 与是否破位
t1 = EV[EV.layer == 1][["date", "dist20", "dh60"]].rename(columns={"dh60": "top1_dh60"})
EV = EV.merge(t1, on="date", how="left", suffixes=("", "_d"))
EV["broke"] = (EV.dist20 <= -5) & (EV.top1_dh60 <= -42)


def stat(s: pd.Series) -> str:
    if not len(s):
        return "     -"
    return f"{len(s):>4}笔 {s.mean():>6.2f}% 胜{(s > 0).mean() * 100:>4.0f}%"


def half(s: pd.DataFrame) -> str:
    """前后两半各自的均值, 看稳不稳。"""
    a = s[s.year <= 2017]["ret_d2"]
    b = s[s.year >= 2018]["ret_d2"]
    return f"{a.mean():>5.2f}%({len(a)}) / {b.mean():>5.2f}%({len(b)})"


print(f"事件表 {len(EV):,} 行 / {EV['date'].nunique()} 天 / "
      f"{int(EV.year.min())}-{int(EV.year.max())}")
print(f"破位日 {EV[EV.broke].date.nunique()} 天, 非破位日 "
      f"{EV[~EV.broke].date.nunique()} 天")

# ================= 破位腿 =================
print("\n" + "=" * 92)
print("一、破位腿（dist20<=-5 且 第1名 dh60<=-42, 次日开盘不高开就买第1名）")
print("=" * 92)
br = EV[(EV.layer == 1) & EV.broke & (EV.gap <= 0.05)].copy()
print(f"合计 {stat(br.ret_d2)}    前后半: {half(br)}")
print(f"\n{'条件':<26}{'统计':<28}{'前后半（笔数）'}")
for lo, hi, nm in [(-11.5, -9, "当日跌停级 (<=-9%)"), (-9, -7, "当日 -9~-7%"),
                   (-7, -5, "当日 -7~-5%"), (-5, -3, "当日 -5~-3%"),
                   (-3, 11.5, "当日 > -3%")]:
    s = br[(br.pct > lo) & (br.pct <= hi)]
    print(f"{nm:<26}{stat(s['ret_d2']):<28}{half(s)}")

print(f"\n{'当日<=-5% 再拆缺口':<26}{'统计':<28}{'前后半'}")
core = br[br.pct <= -5]
for lo, hi in [(-100, -6), (-6, -3), (-3, 0.05)]:
    s = core[(core.gap > lo) & (core.gap <= hi)]
    print(f"  缺口 {lo}~{hi}{'':<10}{stat(s['ret_d2']):<28}{half(s)}")

print(f"\n{'当日<=-5% 再拆 dist20':<26}{'统计':<28}{'前后半'}")
for lo, hi in [(-100, -8), (-8, -7), (-7, -6), (-6, -5)]:
    s = core[(core.dist20 > lo) & (core.dist20 <= hi)]
    print(f"  dist {lo}~{hi}{'':<12}{stat(s['ret_d2']):<28}{half(s)}")

# ================= 深低开腿 =================
print("\n" + "=" * 92)
print("二、深低开腿（破位不成立的日子）: 缺口带 x 名次")
print("=" * 92)
dp = EV[(~EV.broke)].copy()
for k_lo, k_hi, kn in [(1, 3, "1-3名"), (4, 6, "4-6名（现行）"), (7, 8, "7-8名")]:
    print(f"\n{kn}:")
    for lo, hi in [(-100, -10), (-10, -8), (-8, -7), (-7, -6), (-6, -5), (-5, -4), (-4, -3)]:
        s = dp[(dp.layer >= k_lo) & (dp.layer <= k_hi) & (dp.gap > lo) & (dp.gap <= hi)]
        print(f"  缺口 {lo:>5}~{hi:>4}   {stat(s['ret_d2']):<28}{half(s)}")

print("\n合并带（4-6 名）:")
for lo, hi in [(-10, -5), (-10, -6), (-8, -5), (-8, -6), (-7, -5), (-6, -4), (-10, -4)]:
    s = dp[(dp.layer.between(4, 6)) & (dp.gap > lo) & (dp.gap <= hi)]
    print(f"  缺口 {lo:>5}~{hi:>4}   {stat(s['ret_d2']):<28}{half(s)}")

print("\n合并带（1-3 名）:")
for lo, hi in [(-10, -5), (-10, -6), (-8, -5), (-8, -6), (-7, -5), (-6, -4), (-10, -4)]:
    s = dp[(dp.layer.between(1, 3)) & (dp.gap > lo) & (dp.gap <= hi)]
    print(f"  缺口 {lo:>5}~{hi:>4}   {stat(s['ret_d2']):<28}{half(s)}")

# ================= 深低开带内加当日跌幅 =================
print("\n" + "=" * 92)
print("三、深低开腿（4-6 名, 缺口 -10~-4）加当日跌幅")
print("=" * 92)
b = dp[dp.layer.between(4, 6) & (dp.gap > -10) & (dp.gap <= -4)]
for lo, hi in [(-11.5, -9), (-9, -7), (-7, -5), (-5, -3), (-3, 11.5)]:
    s = b[(b.pct > lo) & (b.pct <= hi)]
    print(f"  当日 {lo:>5}~{hi:>5}%   {stat(s['ret_d2']):<28}{half(s)}")

# ================= 分年明细 =================
print("\n" + "=" * 92)
print("四、分年（破位腿 vs 深低开腿 4-6 名 -8~-5）")
print("=" * 92)
print(f"{'年':<6}{'破位腿':<26}{'深低开4-6名':<26}")
d2 = dp[dp.layer.between(4, 6) & (dp.gap > -8) & (dp.gap <= -5)]
for y in sorted(EV.year.unique()):
    a = br[br.year == y]["ret_d2"]
    c = d2[d2.year == y]["ret_d2"]
    fa = f"{len(a):>2}笔 {a.mean():>6.2f}%" if len(a) else "     -"
    fc = f"{len(c):>2}笔 {c.mean():>6.2f}%" if len(c) else "     -"
    print(f"{y:<6}{fa:<26}{fc:<26}")
