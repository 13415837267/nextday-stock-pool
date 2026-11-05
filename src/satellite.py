# -*- coding: utf-8 -*-
"""独立副策略(卫星仓) v2: 每日一只低波票, 持有 40 个交易日。

v1(第一百三十七轮)是月度调仓、等权 20 只。v2 改成"每天推一只":
  · 每天收盘后在股票池里按 20 日特质波动(ivol20)升序, 取**第一只未持有**的票;
  · 次日开盘买入 1/40 仓位, 持有 40 个交易日后结算;
  · 稳态持仓 40 只, 每天一进一出, 日换手 1/40。

为什么不是"每天只持 1 只、天天换"
----------------------------------
第一百三十八轮实测(1500 日, 同窗口基准 -5.86%):
| 方式 | 年化 | 回撤 | 年换手 |
|---|---:|---:|---:|
| 只持 1 只, 每天换 | **-50.8%** | -98.3% | 250 倍 |
| 每天新增 1 只, 持 20 日 | +6.0% | -30.1% | 14.6 倍 |
| 每天新增 1 只, 持 40 日 | **+7.7%** | **-27.4%** | 7.7 倍 |
| 每天新增 1 只, 持 60 日 | +5.8% | -27.8% | 4.7 倍 |
| 月度 20 只 + 缓冲(v1) | +7.5% | -28.7% | 15.6 倍 |

"只持 1 只"的 -50.8% 几乎全部是成本: 每天一次往返 0.3%, 一年 250 次,
(1-0.003)^250-1 = **-52.8%** —— 低波因子的毛 alpha 被券商吃干。
持有 40 日把成本压到 2.3%/年, 才留得住收益。

为什么不做趋势过滤
------------------
"低波 + 站上 20 日线"在 1500 日窗口能到 +9.5%, 但全史只有 +9.9%(无过滤 +8.8%),
且回撤 -51.5% 不比无过滤好 —— 是近五年的现象, 不是稳定 alpha, 故不采纳。
H 从 20 到 120 收益都在 6~9% 之间(全史 8~10%), 说明 alpha 来自"持有低波票"
本身而不是换仓时点, 因此 H=40 取的是换手与分散的折中, 不是收益峰值。

与主策略的关系
--------------
| 维度 | 主策略 dip | 副策略 lowvol_daily |
|---|---|---|
| alpha 来源 | 日内恐慌错杀 + 隔夜修复 | 横截面低波动定价偏差 |
| 持有期 | 2 个交易日 | 40 个交易日 |
| 资金形态 | 单票满仓 | 40 只等权 |
| 交易节奏 | 每日 1 买 1 卖 | 每日 1 买 1 卖(不同标的池) |
| 账本 | output/paper.json | output/satellite.json |

全史 14 年(2013-01~2026-11, 3279 日): 年化 +8.8%、回撤 -52.9%(日频口径)、
与 dip 月收益相关 +0.08。日频回撤远大于 v1 月频口径的 -29.4%, 这是口径差异,
不是策略变差 —— 月频净值天然平滑, 页面上必须按日频披露。

红线
----
· 只读 T 日及之前数据; 不 import backtest/daily_run 的交易逻辑;
· 账本只增不改、同日幂等、不参与主策略资金调度。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import board_of

MIN_BARS = 250       # 上市满 250 日

# v1 月度参数(仅用于归档旧账本)
DEFAULT = {"top_n": 20, "buf_n": 60, "amount_min_yi": 1.0, "cost_pct": 0.3}
# v2 日频参数
DEFAULT_DAILY = {"hold_days": 40, "amount_min_yi": 1.0, "cost_pct": 0.3,
                 "ma20_filter": False, "scan": 50}


def params(cfg: dict) -> dict:
    s = {**DEFAULT, **(cfg.get("satellite") or {})}
    return {"top_n": int(s["top_n"]), "buf_n": int(s["buf_n"]),
            "amt_min": float(s["amount_min_yi"]) * 1e8,
            "cost": float(s["cost_pct"]) / 100.0}


def daily_params(cfg: dict) -> dict:
    s = {**DEFAULT_DAILY, **(cfg.get("satellite") or {})}
    return {"hold_days": max(2, int(s["hold_days"])),
            "amt_min": float(s["amount_min_yi"]) * 1e8,
            "cost": float(s["cost_pct"]) / 100.0,
            "ma20_filter": bool(s.get("ma20_filter", False)),
            "scan": max(1, int(s.get("scan", 50)))}


def universe(day: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """与主策略一致的股票池: 非停牌/非ST/上市满250日/成交额达标/板块在内。"""
    boards = set(cfg["universe"]["boards"])
    amt_min = daily_params(cfg)["amt_min"]
    m = ((day["is_paused"] == 0) & (day["is_st"] == 0)
         & (day["bars"] >= MIN_BARS) & (day["amount"] >= amt_min))
    df = day.loc[m]
    return df.loc[df["ts_code"].map(lambda c: board_of(c) in boards)]


def rank_lowvol(day: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """按 20 日特质波动升序排列(低波在前), 可选"站上 20 日线"过滤。"""
    df = universe(day, cfg)
    p = daily_params(cfg)
    if p["ma20_filter"]:
        m = pd.to_numeric(df["close_adj"], errors="coerce") > pd.to_numeric(df["ma20"], errors="coerce")
        df = df.loc[m.fillna(False)]
    v = pd.to_numeric(df["ivol20"], errors="coerce")
    ok = v.notna() & np.isfinite(v) & (v > 0)
    df = df.loc[ok].assign(_ivol=v[ok])
    return df.sort_values("_ivol", kind="mergesort").reset_index(drop=True)


def pick_one(day: pd.DataFrame, cfg: dict, held=()) -> dict | None:
    """选今日 top1: 按 ivol 升序, 取第一只**不在 held** 的票(最多顺延 scan 名)。

    去重是必须的: 低波票持续性很强, 若允许重复买入, 连续 20 天都推同一只,
    账户会退化成单票满仓 —— 第一百三十八轮第一版就是这么跑出 -1.5% 的。
    """
    p = daily_params(cfg)
    rk = rank_lowvol(day, cfg)
    if len(rk) < 40:          # 池子太小(数据不全)时不出票, 避免极端集中
        return None
    held = set(held or ())
    for _, row in rk.head(p["scan"]).iterrows():
        c = str(row["ts_code"])
        if c in held:
            continue
        return {"code": c, "name": "", "ivol20": round(float(row["_ivol"]), 2),
                "close": round(float(row["close_adj"]), 4)}
    return None


# ---------------------------------------------------------------------------
# v1 月度函数(归档用, 新账本不再调用)
# ---------------------------------------------------------------------------
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
