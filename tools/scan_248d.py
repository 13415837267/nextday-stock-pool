"""第248轮 D: S2 稳健性 —— 分年/仓位面/分段/去极端年"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scan_248c import build_cal, metrics, simulate  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
COST = 0.30


def yearly(sim: pd.Series) -> pd.Series:
    return sim.groupby(sim.index // 10000).apply(lambda x: (1 + x / 100).prod() - 1) * 100


def main() -> None:
    print("=" * 78)
    print("第248轮 D: S2 稳健性")
    print("=" * 78)
    cal = build_cal()
    tr = pd.read_csv(OUT / "w232_trades.csv")
    cal = [d for d in cal if d >= tr.signal.min()]
    c = pd.read_csv(OUT / "w248_cands.csv")

    s0, _, f0 = simulate(cal, tr, c, busy_mode="coarse", priority="main")
    s2, sub2, f2 = simulate(cal, tr, c, busy_mode="exact", priority="fill")

    print("\n【1】逐年 (左 S0 / 右 S2)")
    y0, y2 = yearly(s0), yearly(s2)
    print(f"{'年':>6}{'S0':>9}{'S2':>9}{'差':>8}")
    for k in y0.index:
        print(f"{k:>6}{y0[k]:>8.1f}%{y2[k]:>8.1f}%{y2[k]-y0[k]:>+8.1f}pp")

    # 补位笔数逐年
    f2["yr"] = f2.trade_date // 10000
    print(f"\n{'年':>6}{'S0笔':>6}{'S2笔':>6}{'S2破位':>7}{'S2深低':>7}{'S2单笔':>8}")
    for k in y0.index:
        a = f0[f0.trade_date // 10000 == k]
        b = f2[f2.yr == k]
        print(f"{k:>6}{len(a):>6}{len(b):>6}{int(b.brk.sum()):>7}{int((~b.brk).sum()):>7}"
              f"{(b.y - COST).mean():>7.2f}%")

    print("\n【2】S2 分路径")
    for pa, s in f2.groupby("path"):
        r = s.y - COST
        t = r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))
        print(f"  {pa:<6}{len(s):>4} 笔  单笔 {r.mean():>5.2f}%  胜率 {(r>0).mean()*100:>3.0f}%  t={t:>5.2f}")

    print("\n【3】仓位面 (S2 调度固定, 扫 破位仓/深低开仓)")
    print(f"{'':>8}" + "".join(f"{p:>8.0%}" for p in (0.3, 0.4, 0.5, 0.6, 0.7)))
    for pb in (0.2, 0.3, 0.4, 0.5):
        line = f"{pb:>8.0%}"
        for pd_ in (0.3, 0.4, 0.5, 0.6, 0.7):
            sim, _, _ = simulate(cal, tr, c, busy_mode="exact", priority="fill",
                                 p_brk=pb, p_deep=pd_)
            line += f"{metrics(sim)[2]:>8.2f}"
        print(line)

    print("\n【4】去掉极端年 (每年独立剔除, 看卡玛还剩多少)")
    for k in yearly(s2).index:
        sim = s2.copy()
        sim[sim.index // 10000 == k] = 0.0
        print(f"  剔除 {k}: 年化 {metrics(sim)[0]:>6.2f}%  回撤 {metrics(sim)[1]:>6.2f}%  "
              f"卡玛 {metrics(sim)[2]:.2f}")

    print("\n【5】集中度")
    print(f"  S2 {len(f2)} 笔  票数 {f2.ts_code.nunique()}  最长连买 "
          f"{int(f2.groupby('ts_code').size().max())} 次  前5票 "
          f"{f2.ts_code.value_counts().head(5).sum()/len(f2)*100:.0f}%")
    # 净值贡献: 前 10 笔盈利占总盈利
    r = (f2.y - COST) * f2.p
    pos = r[r > 0].sum()
    print(f"  总盈利 {pos:.0f}pp, 前 10 笔盈利占 {r.nlargest(10).sum()/pos*100:.0f}%")
    print(f"  年化 {metrics(s2)[0]:.2f}%  回撤 {metrics(s2)[1]:.2f}%  卡玛 {metrics(s2)[2]:.2f}")

    f2.to_csv(OUT / "w248_final_S2.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"S0": s0, "S2": s2}).to_csv(OUT / "w248_nav_cmp.csv", encoding="utf-8-sig")

    # 深低开 119 笔前后段
    dp = f2[f2.path == "深低开"]
    print(f"\n【6】深低开 {len(dp)} 笔分段")
    for lo, hi, nm in ((2016, 2019, "16-19"), (2020, 2022, "20-22"), (2023, 2026, "23-26")):
        s = dp[(dp.yr >= lo) & (dp.yr <= hi)]
        if len(s):
            print(f"  {nm}: {len(s):>3} 笔 单笔 {(s.y - COST).mean():>5.2f}%  胜率 "
                  f"{((s.y-COST)>0).mean()*100:>3.0f}%")


if __name__ == "__main__":
    main()
