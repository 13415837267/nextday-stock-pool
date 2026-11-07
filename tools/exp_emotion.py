# -*- coding: utf-8 -*-
"""第一百六十七轮: 市场微结构情绪指标构建 + 条件化检验。

背景: config 的 regime 层只有三个慢变量(market_ma / breadth_min / vol_max),
      全是"指数均线/站上MA20占比/20日波动"这类 20 日尺度的指标。
      短线技能(SKILL: ashare-short-term-trading / a-share-daily-review)里的
      "五维情绪"用的是**日频快变量**: 涨停家数、炸板率、连板高度、打板溢价、赚钱效应。
      CHANGELOG 全库检索: "炸板" 0 次、"涨停家数" 0 次、"赚钱效应" 1 次(非指标语境)。
      => 这一族从未算过。

本脚本只做第 1 步: 逐日从 data/panel 原始面板计算 T 日收盘即可得的情绪指标。
不保留全量面板(1661 日 x 5000 只), 只输出每日一行, 内存恒定。

口径:
  zt  涨停: close >= high_limit - 0.001  (用数据源给的涨跌停价, 自动适配 ST 5% / 创业 20% / 北交 30%)
  zb  炸板: high >= high_limit - 0.001 且 close < high_limit - 0.001
  dt  跌停: close <= low_limit + 0.001
  up  上涨: close > pre_close
  streak  连板: 跨日累积的连续涨停天数(停牌日不变, 非涨停清零)
  zt_prem 打板溢价: 昨日涨停股在今日的 pct 均值(情绪最核心的日频读数)
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANEL = os.path.join(ROOT, "data", "panel")
OUT = os.path.join(ROOT, "data", "cache", "emotion.pkl")

EPS = 0.001


def main() -> None:
    files = sorted(glob.glob(os.path.join(PANEL, "*.csv.gz")))
    print(f"面板 {len(files)} 日: {os.path.basename(files[0])[:8]} ~ {os.path.basename(files[-1])[:8]}", flush=True)

    streak: dict[str, int] = {}
    prev_zt: set[str] = set()
    rows: list[dict] = []

    for i, f in enumerate(files):
        d = os.path.basename(f)[:8]
        df = pd.read_csv(
            f,
            usecols=["ts_code", "close", "high", "pre_close", "pct",
                     "high_limit", "low_limit", "amount", "is_paused", "is_st"],
            dtype={"ts_code": str},
        )
        live = df[df["is_paused"] == 0]
        if len(live) == 0:
            continue
        cl = live["close"].to_numpy()
        hi = live["high"].to_numpy()
        hl = live["high_limit"].to_numpy()
        ll = live["low_limit"].to_numpy()
        pc = live["pre_close"].to_numpy()
        pct = live["pct"].to_numpy()
        code = live["ts_code"].to_numpy()

        zt = cl >= hl - EPS
        zb = (hi >= hl - EPS) & ~zt
        dt = cl <= ll + EPS
        up = cl > pc

        # 连板: 未涨停即清零; 停牌股不在 live 里, 保持原值
        cur_zt = set(code[zt])
        codeset = set(code)          # 必须在循环外构造: 放在 for 里是 O(n^2), 实测 1661 日要跑 10 分钟
        for c in cur_zt:
            streak[c] = streak.get(c, 0) + 1
        for c in list(streak.keys()):
            if c not in cur_zt and c in codeset:
                streak[c] = 0
        max_board = max((streak.get(c, 0) for c in cur_zt), default=0)
        b2 = sum(1 for c in cur_zt if streak.get(c, 0) >= 2)

        # 打板溢价: 昨日涨停股今日涨幅(需今日在交易)
        if prev_zt:
            mask = np.isin(code, list(prev_zt))
            prem = float(np.nanmean(pct[mask])) if mask.any() else np.nan
            prem_med = float(np.nanmedian(pct[mask])) if mask.any() else np.nan
            n_prem = int(mask.sum())
        else:
            prem = prem_med = np.nan
            n_prem = 0

        rows.append(dict(
            d=d,
            zt_n=int(zt.sum()),
            zb_n=int(zb.sum()),
            dt_n=int(dt.sum()),
            zb_rate=float(zb.sum() / max(zt.sum() + zb.sum(), 1)),
            up_ratio=float(up.mean()),
            max_board=int(max_board),
            b2_n=int(b2),
            zt_prem=prem,
            zt_prem_med=prem_med,
            n_prem=n_prem,
            amt_yi=float(live["amount"].sum() / 1e8),
            n_live=int(len(live)),
            n_st=int((live["is_st"] == 1).sum()),
        ))
        prev_zt = cur_zt

        if i % 300 == 0:
            print(f"  {i}/{len(files)} {d} zt={int(zt.sum())} dt={int(dt.sum())} maxB={max_board}", flush=True)

    em = pd.DataFrame(rows)
    # 派生: 情绪的"相对温度"——与自身 5/20 日均值比, 消除牛熊中枢漂移
    for c in ("zt_n", "zb_rate", "zt_prem", "max_board", "up_ratio"):
        em[f"{c}_ma5"] = em[c].rolling(5, min_periods=3).mean()
        em[f"{c}_ma20"] = em[c].rolling(20, min_periods=10).mean()
        em[f"{c}_z20"] = (em[c] - em[f"{c}_ma20"]) / em[f"{c}_ma20"].replace(0, np.nan)
    em["zt_ratio5"] = em["zt_n"] / em["zt_n_ma5"].replace(0, np.nan)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    em.to_pickle(OUT)
    print(f"\n保存 {OUT}  {em.shape}")
    print(em[["zt_n", "dt_n", "zb_rate", "max_board", "zt_prem", "up_ratio", "amt_yi"]].describe().round(3).to_string())
    print("\n分年:")
    em["y"] = em["d"].str[:4]
    print(em.groupby("y").agg(
        zt=("zt_n", "mean"), dt=("dt_n", "mean"), zb=("zb_rate", "mean"),
        maxB=("max_board", "mean"), prem=("zt_prem", "mean"), up=("up_ratio", "mean"),
        amt=("amt_yi", "mean"), n=("d", "size")).round(2).to_string())


if __name__ == "__main__":
    main()
