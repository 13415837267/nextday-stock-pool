"""第248轮 E: 压力测试 —— 每条腿单独失效会怎样"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scan_248c import build_cal, metrics, simulate  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
COST = 0.30


def main() -> None:
    print("=" * 78)
    print("第248轮 E: 压力测试")
    print("=" * 78)
    cal = build_cal()
    tr = pd.read_csv(OUT / "w232_trades.csv")
    cal = [d for d in cal if d >= tr.signal.min()]
    c = pd.read_csv(OUT / "w248_cands.csv")

    s0, sub0, f0 = simulate(cal, tr, c, busy_mode="coarse", priority="main")
    s1, sub1, f1 = simulate(cal, tr, c, busy_mode="exact", priority="main")
    s2, sub2, f2 = simulate(cal, tr, c, busy_mode="exact", priority="fill")

    print(f"\n方案          笔数    年化     回撤    卡玛")
    for nm, sim, f in (("S0", s0, f0), ("S1", s1, f1), ("S2", s2, f2)):
        a, d_, m = metrics(sim)
        print(f"{nm}          {len(f):>4}  {a:>7.2f}% {d_:>7.2f}% {m:>6.2f}")

    print(f"\n【1】S2 下主策略单独 (被砍 {len(tr)-len(sub2)} 笔)")
    b = pd.Series(0.0, index=cal)
    for dt, g in sub2.groupby("signal"):
        b[int(dt)] = g.ret.mean()
    print(f"  主策略 {len(sub2)} 笔: 年化 {metrics(b)[0]:.2f}%  回撤 {metrics(b)[1]:.2f}%  "
          f"卡玛 {metrics(b)[2]:.2f}  (原 559 笔 86.42/3.57)")

    print("\n【2】单腿失效")
    for cond, nm in ((f2.path == "深低开", "深低开归零"),
                     (f2.path == "破位", "破位归零")):
        g = f2.copy()
        g.loc[cond, "y"] = COST           # 收益打平(只亏成本以外的部分没了)
        sim = b.copy()
        v = pd.Series((g.y.values - COST) * g.p.values, index=g.trade_date.values)
        v = v.groupby(level=0).sum().sort_index()
        sim[v.index] = sim[v.index] + v.values
        a, d_, m = metrics(sim)
        print(f"  {nm}: 年化 {a:>7.2f}%  回撤 {d_:>7.2f}%  卡玛 {m:.2f}")
    g = f2.copy()
    g.loc[g.path == "深低开", "y"] = COST + (g.loc[g.path == "深低开", "y"] - COST) / 2
    sim = b.copy()
    v = pd.Series((g.y.values - COST) * g.p.values, index=g.trade_date.values)
    v = v.groupby(level=0).sum().sort_index()
    sim[v.index] = sim[v.index] + v.values
    print(f"  深低开减半: 年化 {metrics(sim)[0]:>7.2f}%  卡玛 {metrics(sim)[2]:.2f}")

    print("\n【3】重叠日 41 笔深低开 vs 其它深低开")
    ov = pd.read_csv(OUT / "w248_overlap_detail.csv")
    ovd = set(ov[ov.路径 == "深低开"].trade_date)
    dp = f2[f2.path == "深低开"]
    a1 = dp[dp.trade_date.isin(ovd)]
    a2 = dp[~dp.trade_date.isin(ovd)]
    print(f"  重叠日深低开  {len(a1):>3} 笔  单笔 {(a1.y-COST).mean():>5.2f}%  胜率 {((a1.y-COST)>0).mean()*100:.0f}%")
    print(f"  其它深低开    {len(a2):>3} 笔  单笔 {(a2.y-COST).mean():>5.2f}%  胜率 {((a2.y-COST)>0).mean()*100:.0f}%")
    # 全部深低开候选(未限次)在重叠日的表现
    ovc = c[(c.trade_date.isin(ovd)) & (~c.brk)]
    print(f"  全部候选(重叠日) {len(ovc)} 天 单笔 {(ovc.y-COST).mean():.2f}%")

    print("\n【4】S1 -> S2 只在 48 天上有差别")
    dif = f2.set_index("trade_date").y - f1.set_index("trade_date").y
    both = dif.dropna()
    print(f"  共同交易日 {len(both)}, 收益差异合计 {both.sum():+.1f}pp (未乘仓位)")
    print(f"  S2 独有 {len(set(f2.trade_date)-set(f1.trade_date))} 天, "
          f"S1 独有 {len(set(f1.trade_date)-set(f2.trade_date))} 天")

    print("\n【5】最差 10 天 (S2)")
    print(f"  {(s2.nsmallest(10))}")
    print(f"\n  单日亏损 >5% 的天数: S0 {int((s0<-5).sum())} / S2 {int((s2<-5).sum())}")


if __name__ == "__main__":
    main()
