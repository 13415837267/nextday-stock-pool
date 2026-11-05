# -*- coding: utf-8 -*-
"""独立副策略(卫星仓): 低特质波动 + 缓冲带, 月度调仓, 等权 20 只。

为什么它与主策略(dip)完全独立
------------------------------
| 维度 | 主策略 dip | 副策略 satellite |
|---|---|---|
| alpha 来源 | 日内恐慌错杀 + 隔夜修复 | 横截面低波动定价偏差 |
| 持有期 | 2 个交易日 | 约 1 个月 |
| 资金形态 | 单票满仓 | 等权 20 只 |
| 交易节奏 | 每日 | 每月 |
| 账本 | output/paper.json | output/satellite.json |
| 代码路径 | src/backtest.py | 本文件(不 import 回测引擎) |

全史 14 年(161 期, 2012-11~2026-11): 年化 **+9.6%**、回撤 **-29.4%**、卡玛 0.32、
月胜率 52.8%、与 dip 月收益相关 **+0.082**。基准(全市场等权)同期 +1.6%/年。
研究过程与候选比较见 `tools/exp_satellite.py`、CHANGELOG 第一百三十七轮。

为什么是"低波 + 缓冲带"
-----------------------
· 低波(ivol20 最低 20 只)是 136 轮 IC 检验里唯一在长短持有期都稳定为正的因子
  (k=60 时 IC +0.141), 也是九个候选里唯一全史为正的;
· 裸低波年化 +7.0%, 但月换手 0.88, 成本吃掉 3.2%/年;
· 加 3 倍缓冲带(旧持仓跌出前 60 才卖)后换手降到 0.73, 年化 +9.6%、回撤反而降到 -29.4%;
· 季度调仓(换手 0.30)只有 +4.7% —— 再平衡太慢会错过低波组合的均值回复。

红线
----
· 只读 T 日及之前数据; 不 import backtest/daily_run, 不改主策略任何参数;
· 账本只增不改、同月幂等、不参与主策略资金调度。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import board_of

MIN_BARS = 250       # 上市满 250 日
# 默认值 = 研究口径(tools/exp_satellite.py), 可在 config.yaml 的 satellite 段覆盖
DEFAULT = {"top_n": 20, "buf_n": 60, "amount_min_yi": 1.0, "cost_pct": 0.3}


def params(cfg: dict) -> dict:
    s = {**DEFAULT, **(cfg.get("satellite") or {})}
    return {"top_n": int(s["top_n"]), "buf_n": int(s["buf_n"]),
            "amt_min": float(s["amount_min_yi"]) * 1e8,
            "cost": float(s["cost_pct"]) / 100.0}


def universe(day: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """与主策略一致的股票池: 非停牌/非ST/上市满250日/成交额达标/板块在内。"""
    boards = set(cfg["universe"]["boards"])
    amt_min = params(cfg)["amt_min"]
    m = ((day["is_paused"] == 0) & (day["is_st"] == 0)
         & (day["bars"] >= MIN_BARS) & (day["amount"] >= amt_min))
    df = day.loc[m]
    return df.loc[df["ts_code"].map(lambda c: board_of(c) in boards)]


def rank_lowvol(day: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """按 20 日特质波动升序排列(低波在前)。"""
    df = universe(day, cfg)
    v = pd.to_numeric(df["ivol20"], errors="coerce")
    ok = v.notna() & np.isfinite(v)
    df = df.loc[ok].assign(_ivol=v[ok])
    return df.sort_values("_ivol", kind="mergesort").reset_index(drop=True)


def pick(day: pd.DataFrame, cfg: dict, prev_codes=None, names=None) -> list[dict]:
    """选出本月持仓。prev_codes 用于缓冲带(降低换手)。"""
    p = params(cfg)
    top_n, buf_n = p["top_n"], p["buf_n"]
    rk = rank_lowvol(day, cfg)
    if len(rk) < top_n:
        return []
    top_codes = rk.head(top_n)["ts_code"].tolist()
    if prev_codes:
        buf = set(rk.head(buf_n)["ts_code"].tolist())
        keep = [c for c in prev_codes if c in buf]
        add = [c for c in top_codes if c not in set(keep)]
        codes = (keep + add)[:top_n]
    else:
        codes = top_codes
    ivol = rk.set_index("ts_code")["_ivol"].to_dict()
    nm = names or {}
    w = round(1.0 / max(1, len(codes)), 4)
    return [{"code": c, "name": nm.get(c, ""),
             "ivol20": round(float(ivol.get(c, 0.0)), 2), "weight": w} for c in codes]


def turnover(new_codes, old_codes) -> float:
    """换手率: 1 - 重叠比例。"""
    if not old_codes:
        return 1.0
    a, b = set(new_codes), set(old_codes)
    return 1.0 - len(a & b) / max(1, len(a))


def apply_cost(nav_before: float, turn: float, cost: float = 0.0030) -> float:
    """按换手扣往返成本, 返回调仓后净值。"""
    return float(nav_before) * (1.0 - turn * cost)
