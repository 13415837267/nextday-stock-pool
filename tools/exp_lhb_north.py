# -*- coding: utf-8 -*-
"""第一百六十七轮: 把"龙虎榜 北向净买 & 当日跌>=5%"从统计量做成可交易策略。

背景(第166轮留的缝): 该组在 T+2 收盘卖口径下毛收益 +1.157%、扣 0.13% 双边费后
净超额 +0.854pp, 高于主策略资金的两天租金 0.57%。但那是 1,068 个样本的**平均**,
不是"每天 1 只"的收益 —— 本轮把它做成单票策略实盘化验证。

口径:
  T 日盘后龙虎榜发布 -> T+1 开盘买(一字涨停/停牌剔除) -> T+2 收盘卖
  费用 0.13% 双边 + 印花税(2023-08-28 后万5, 前千1)
  每日最多 1 笔, 排序键在小样本上比较

判据(预注册):
  1) 全样本单笔净超额 > +0.57%(两天租金);
  2) 两个半年段同号;
  3) 排序键必须在**前段定、后段验**, 不允许全样本挑最优。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "data", "cache")


def build_price_matrices():
    p = pd.read_pickle(os.path.join(CACHE, "panel_all.pkl"))
    p = p.loc[p["is_paused"] == 0, ["ts_code", "d", "open", "close", "factor"]]
    p["px_o"] = (p["open"].astype("float64") * p["factor"]).astype("float32")
    p["px_c"] = (p["close"].astype("float64") * p["factor"]).astype("float32")
    mo = p.pivot_table(index="d", columns="ts_code", values="px_o", aggfunc="first")
    mc = p.pivot_table(index="d", columns="ts_code", values="px_c", aggfunc="first")
    return mo, mc


def main() -> None:
    lhb = pd.read_pickle(os.path.join(CACHE, "lhb_probe.pkl"))
    mo, mc = build_price_matrices()
    days = list(mo.index)
    pos = {d: i for i, d in enumerate(days)}
    o_np, c_np = mo.to_numpy(), mc.to_numpy()
    cols = {c: i for i, c in enumerate(mo.columns)}

    def fwd(code: str, d: str):
        """T+1 开盘 -> T+2 收盘 收益(复权)。"""
        i = pos.get(d)
        j = cols.get(code)
        if i is None or j is None or i + 2 >= len(days):
            return np.nan
        o1, c2 = o_np[i + 1, j], c_np[i + 2, j]
        if not (o1 > 0) or not (c2 > 0):
            return np.nan
        r = c2 / o1 - 1.0
        return r if abs(r) <= 0.6 else np.nan

    def o1_ret_pre(code: str, d: str):
        """T+1 开盘 / T 日收盘 - 1, 用于判断 T+1 是否一字涨停(买不到)。"""
        i, j = pos.get(d), cols.get(code)
        if i is None or j is None or i + 1 >= len(days):
            return np.nan
        c0, o1 = c_np[i, j], o_np[i + 1, j]
        if not (c0 > 0) or not (o1 > 0):
            return np.nan
        return o1 / c0 - 1.0

    base = lhb[(lhb["north"] > 0) & (lhb["chg"] <= -5)].copy()
    print(f"子集 {len(base)} 条 / {base['d'].nunique()} 天", flush=True)

    base["ret_gross"] = [fwd(c, d) for c, d in zip(base["code"], base["d"])]
    base["gap1"] = [o1_ret_pre(c, d) for c, d in zip(base["code"], base["d"])]
    # 一字涨停买不到: T+1 开盘相对 T 日收盘 >= +9.5%(主板) 视为买不进
    base["buyable"] = ~(base["gap1"] >= 0.095)
    ok = base.dropna(subset=["ret_gross"]).copy()
    ok = ok[ok["buyable"]]
    print(f"可回测 {len(ok)} 条 ({len(ok)/len(base)*100:.1f}%)  "
          f"{ok['d'].nunique()} 天", flush=True)

    # 费用: 双边 0.13% + 印花税(卖出)
    def net(r, d):
        stamp = 0.001 if d < "20230828" else 0.0005
        return (1 + r) * (1 - stamp) * (1 - 0.0013) - 1

    ok["ret_net"] = [net(r, d) for r, d in zip(ok["ret_gross"], ok["d"])]
    print(f"\n全样本单笔: 毛 {ok['ret_gross'].mean()*100:+.3f}%  "
          f"净 {ok['ret_net'].mean()*100:+.3f}%  胜率 {(ok['ret_net']>0).mean()*100:.1f}%", flush=True)

    # 按日选 1 只: 不同排序键
    keys = {
        "north_max": ("north", False),
        "inst_max": ("inst", False),
        "chg_min": ("chg", True),
        "turn_max": ("turn", False),
        "north_x_chg": ("north_chg", False),
    }
    ok["north_chg"] = ok["north"] * (-ok["chg"])     # 北向净买额 x 跌幅

    print("\n=== 每日选 1 只: 各排序键(全样本, 仅诊断) ===", flush=True)
    rows = []
    for name, (col, asc) in keys.items():
        pick = ok.sort_values(["d", col], ascending=[True, asc]).groupby("d").head(1)
        r = pick["ret_net"]
        rows.append(dict(key=name, n=len(pick), mean=round(r.mean() * 100, 3),
                         med=round(r.median() * 100, 3),
                         win=round((r > 0).mean() * 100, 1),
                         sum_log=round(np.log1p(r).sum(), 3)))
    print(pd.DataFrame(rows).to_string(index=False))

    # 前段定 / 后段验
    mid = sorted(ok["d"].unique())[ok["d"].nunique() // 2]
    print(f"\n分段: 前段 <= {mid}, 后段 > {mid}", flush=True)
    rows = []
    for name, (col, asc) in keys.items():
        pick = ok.sort_values(["d", col], ascending=[True, asc]).groupby("d").head(1)
        a, b = pick[pick["d"] <= mid], pick[pick["d"] > mid]
        rows.append(dict(key=name, n1=len(a), m1=round(a["ret_net"].mean() * 100, 3),
                         n2=len(b), m2=round(b["ret_net"].mean() * 100, 3)))
    print(pd.DataFrame(rows).to_string(index=False))

    # 全部买入(等权)对照
    eq = ok.groupby("d")["ret_net"].mean()
    print(f"\n全买等权: {len(eq)} 天, 日均 {eq.mean()*100:+.3f}%, "
          f"年化(252) {((1+eq.mean())**252-1)*100:.2f}%", flush=True)

    ok.to_pickle(os.path.join(CACHE, "lhb_north_dip.pkl"))


if __name__ == "__main__":
    main()
