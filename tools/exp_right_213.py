"""第 213 轮: 右侧追高"持有多日"重建 —— 逼近 250 日新高的持有期优化与单吊可行性。

用户决定: 不加杠杆; 右侧允许持有多日(T+1买, 持有 h 日后卖, h=3/5/10);
若右侧仍不达标则删除副策略。

本轮回答三个问题:
  ① 逼近新高甜点叠加哪些条件能把 ret5/ret10 净收益抬到多高;
  ② 单吊(每日选 1 只)在各排序键、各持有期下能拿到多少;
  ③ 等权分散(正确考虑 10 日资金周转)的近似年化。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "output" / "panel_all_205.parquet"
COST = 0.30
HOLDS = ("ret3", "ret5", "ret10")
COLS = ["d", "pct", "mom20", "mom60", "dist_ph250", "dist_ph20", "vr", "vr5",
        "vol20", "ivol20", "lu5", "limit_up", "turnover_rate", "amt_ratio",
        "up_streak", "ma20_slope", "bias20", "ret1", "ret3", "ret5", "ret10"]


def load() -> pd.DataFrame:
    df = pd.read_parquet(PANEL, columns=COLS)
    df = df.dropna(subset=["ret10"])
    for c in HOLDS:
        df[c + "_net"] = df[c] - COST
    df["d"] = df["d"].astype(str)
    return df


def refine(df: pd.DataFrame) -> None:
    """① 逼近新高池内, 逐层叠加条件, 看 ret5/ret10 净收益。"""
    base = df[(df["pct"] > 0) & df["dist_ph250"].between(-5, 0)].copy()
    nd = base["d"].nunique()
    print(f"\n== ① 逼近新高池 n={len(base):,} 每日{len(base)/nd:.0f} ==")
    for c in HOLDS:
        s = base[c + "_net"]
        print(f"  基准 {c}_net: 均值{s.mean():+.2f}% 胜率{(s>0).mean()*100:.1f}%")

    # 关键条件扫描: 每个条件维度, 看 ret5/ret10 均值如何随该维度变化
    print("\n  维度细分 (ret10_net 均值 / 胜率 / 每日):")
    dims = [
        ("vr", [0, 1, 1.5, 2, 3, 5, 99]),
        ("vol20", [0, 1.5, 2, 2.5, 3, 4, 99]),
        ("mom60", [-999, 0, 10, 25, 50, 100, 999]),
        ("up_streak", [-1, 0, 1, 2, 3, 5, 99]),
        ("lu5", [-1, 0, 1, 2, 99]),
    ]
    for col, edges in dims:
        print(f"\n  [{col}]")
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            sub = base[(base[col] >= lo) & (base[col] < hi)]
            if len(sub) < 200:
                continue
            s = sub["ret10_net"]
            print(f"    {lo:>5}~{hi:<5} n={len(sub):>6} 每日{len(sub)/nd:>5.1f} "
                  f"ret10 {s.mean():+.2f}% 胜率{(s>0).mean()*100:.1f}%")


def head_pick(df: pd.DataFrame) -> None:
    """② 单吊: 各排序键每日第 1 名, 在 ret3/ret5/ret10 下的表现。"""
    pool = df[(df["pct"] > 0) & df["dist_ph250"].between(-5, 0)].copy()
    print(f"\n== ② 逼近新高池单吊 (全池 ret10_net {pool['ret10_net'].mean():+.2f}%) ==")
    keys = [("dist_ph250", False, "距新高最近"), ("mom60", True, "刚启动(动量最小)"),
            ("mom60", False, "动量最强"), ("vr", True, "量比最小"), ("vol20", True, "波动最低"),
            ("up_streak", False, "连涨最多"), ("lu5", True, "涨停最少"),
            ("pct", True, "当日涨幅最小")]
    for col, asc, nm in keys:
        p = pool.dropna(subset=[col]).copy()
        p["rk"] = p.groupby("d")[col].rank(ascending=asc, method="first")
        n = p.groupby("d")[col].transform("size")
        t1 = p[(p.rk == 1) & (n >= 2)]
        parts = []
        for c in HOLDS:
            parts.append(f"{c[3:]}日{t1[c + '_net'].mean():+.2f}%")
        print(f"  {nm:16s} n={len(t1):>5}  " + "  ".join(parts))


def equal_weight(df: pd.DataFrame) -> None:
    """③ 等权分散近似年化(资金 h 日周转一次, 每日滚动 1/h 仓位)。"""
    pool = df[(df["pct"] > 0) & df["dist_ph250"].between(-5, 0)].copy()
    print(f"\n== ③ 逼近新高池等权分散近似年化 ==")
    pool["yr"] = pool["d"].str[:4]
    for c in HOLDS:
        h = int(c[3:])
        batch = pool.groupby("d")[c + "_net"].mean()
        daily = batch / 100 / h          # 每日每份资金的平均收益(滚动 h 日持仓)
        byyr = daily.groupby(daily.index.str[:4]).mean()
        ann = (1 + byyr) ** 250 - 1
        tot = (1 + ann).prod()
        yrs = sorted(byyr.index)
        print(f"  {c} (持有{h}日): 复利 {tot:,.1f}x  年化 {(tot**(1/(len(yrs)))-1)*100:.1f}%"
              f"  近5年 {ann[yrs[-5:]].mean()*100:+6.1f}%")

    # 核心池(叠加最优条件)的等权年化
    core = df[(df["pct"] > 0) & df["dist_ph250"].between(-5, 0)
              & df["vr"].between(1, 2) & (df["vol20"] < 3) & (df["lu5"] == 0)].copy()
    print(f"\n  核心池(+vr1~2+vol20<3+lu5=0) n={len(core):,}:")
    core["yr"] = core["d"].str[:4]
    for c in HOLDS:
        h = int(c[3:])
        batch = core.groupby("d")[c + "_net"].mean()
        daily = batch / 100 / h
        byyr = daily.groupby(daily.index.str[:4]).mean()
        ann = (1 + byyr) ** 250 - 1
        tot = (1 + ann).prod()
        yrs = sorted(byyr.index)
        print(f"  {c} (持有{h}日): 复利 {tot:,.1f}x  年化 {(tot**(1/len(yrs))-1)*100:.1f}%")


def main() -> None:
    df = load()
    print(f"面板 {len(df):,} 股票日 / {df.d.nunique():,} 天")
    refine(df)
    head_pick(df)
    equal_weight(df)


if __name__ == "__main__":
    main()
