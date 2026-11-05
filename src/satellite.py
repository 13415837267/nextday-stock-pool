# -*- coding: utf-8 -*-
"""独立副策略(卫星仓) v3 —— 单吊, 与主策略同构。

与主策略的关系(第一百三十九轮定稿)
----------------------------------
· **数据完全共用**: 同一个 ZZClient、同一份行情面板、同一套指标(build_indicators)。
  副策略不单独抓数、不单独建库; 每日只取一次最新数据, 两个策略在同一份数据上各算各的。
· **执行架构一致**: 100% 仓位单吊、每日最多买 1 只、T 日收盘出信号 / T+1 开盘买、
  持满 N 个交易日卖出、不到标准就不成交(空仓等待)。
· **策略逻辑独立**: 选股条件、持有期、止损、大盘闸门都是自己的参数, 不读主策略的
  候选池与评分, 也不参与主策略的资金调度。两条腿各记各的账。

选股(mode)
----------
· trend_lowvol: 均线多头(close>MA5>MA10>MA20 且 MA20 向上) + 20 日动量 > 0,
  在其中取 20 日特质波动(ivol) 最低的一只 —— "上涨趋势里最稳的票"。
· lowvol:       不要求趋势, 直接取全池 ivol 最低的一只(对照组)。

为什么"上涨趋势"只当过滤器、不当排序器
--------------------------------------
第一百三十九轮四组实测(1500 日 / 全史 3378 日, 均为单吊满仓):

| 选股方式 | 1500 日 | 全史 |
|---|---|---|
| 阳线突破(近20日新高) H20 | +6.0% / -69.7% | -1.2% / -73.5% |
| 趋势内回调(pullback) H10 | +12.4% / -76.1% | — |
| 纯均线多头 H5 | +3.2% / -88.6% | — |
| **趋势过滤 + ivol 最低 H20** | **-6.4% / -72.9%** | +5.2% / -71.0% |
| 纯 ivol 最低 H20(无趋势) | **+7.8% / -42.7%** | +6.1% / -57.6% |

把"上涨趋势"当排序因子(追最强)是负期望 —— A 股短周期均值回复, 追突破是接盘。
当过滤器(只在多头排列里挑最稳的)也没能跑赢不加过滤: 趋势过滤在全史是负贡献。
所以最终版默认 **mode=lowvol**(与 v2 同一因子), 趋势过滤作为可选开关保留,
默认关闭 —— 数据不支持的事不写进默认路径。

本模块零耦合: 不 import backtest / daily_run / selector。
"""
from __future__ import annotations

MIN_BARS = 250

DEFAULT = {
    "mode": "lowvol",         # lowvol | trend_lowvol
    "hold_days": 20,          # 持有交易日数
    "max_ivol": 0,            # ivol 上限(0=不限)
    "amount_min_yi": 1.0,     # 成交额下限(亿)
    "cost_pct": 0.3,          # 往返成本(%)
    "gap_min": -3.0,          # T+1 竞价缺口下限(%): 深低开破位不接
    "gap_max": 5.0,           # 上限: 高开过多不追
    "stop_pct": 0.0,          # 止损(%): 0=关闭; 收盘相对买入价跌破即卖
    "market_gate": False,     # 大盘闸门: 指数在 MA20 上方才开新仓
    "min_pool": 20,           # 池子最少只数, 不足则不出票
}

STRATEGY_CN = {"trend_lowvol": "趋势低波单吊", "lowvol": "低波单吊"}


def daily_params(cfg: dict) -> dict:
    s = {**DEFAULT, **(cfg.get("satellite") or {})}
    return {"mode": str(s["mode"]),
            "hold_days": int(s["hold_days"]),
            "max_ivol": float(s["max_ivol"] or 0),
            "amt_min": float(s["amount_min_yi"]) * 1e8,
            "cost": float(s["cost_pct"]) / 100.0,
            "gap_min": float(s["gap_min"]),
            "gap_max": float(s["gap_max"]),
            "stop": float(s["stop_pct"]) / 100.0,
            "gate": bool(s["market_gate"]),
            "min_pool": int(s["min_pool"])}


def apply_cost(nav_before: float, turn: float, cost: float = 0.0030) -> float:
    """按换手率扣成本(单边)。turn=1.0 表示全部换手一次。"""
    return float(nav_before) * (1.0 - float(cost) * float(turn))


def rank_pool(day, cfg: dict):
    """当日候选池, 已按 ivol 升序。池子不足返回空表。"""
    from src.config import board_of

    P = daily_params(cfg)
    uni = cfg.get("universe") or {}
    boards = set(uni.get("boards") or ["main", "gem"])
    d = day
    m = ((d["is_paused"] == 0) & (d["is_st"] == 0) & (d["bars"] >= MIN_BARS)
         & (d["amount"] >= P["amt_min"]) & (d["ivol20"] > 0))
    if P["max_ivol"] > 0:
        m &= (d["ivol20"] <= P["max_ivol"])
    m &= d["ts_code"].map(lambda c: board_of(c) in boards).to_numpy()
    if P["mode"] == "trend_lowvol":
        m &= ((d["close_adj"] > d["ma5"]) & (d["ma5"] > d["ma10"])
              & (d["ma10"] > d["ma20"]) & (d["ma20"] > d["ma20_prev"])).to_numpy()
        m &= (d["mom20"] > 0).to_numpy()
    df = d.loc[m]
    if len(df) < P["min_pool"]:
        return df.iloc[0:0]
    return df.assign(_ivol=df["ivol20"].astype(float)).sort_values("_ivol")


def pick_one(day, cfg: dict) -> dict | None:
    """当日唯一候选; 池子不足 min_pool 时返回 None(宁缺勿滥)。"""
    rk = rank_pool(day, cfg)
    if len(rk) < daily_params(cfg)["min_pool"]:
        return None
    r = rk.iloc[0]
    return {"code": str(r["ts_code"]),
            "ivol20": round(float(r["_ivol"]), 3),
            "pct": round(float(r["pct"]), 2),
            "mom20": round(float(r["mom20"]), 1),
            "close": round(float(r["close"]), 2),
            "amount_yi": round(float(r["amount"]) / 1e8, 2),
            # 池深随候选一起带出: 回放要解释"这天为什么没买", 需要知道是全池排序的结果,
            # 不是再算一遍 rank_pool(重复排序 1500 次)
            "pool_n": int(len(rk))}


def entry_gate(gap_pct: float, cfg: dict) -> tuple[bool, str]:
    """T+1 竞价缺口是否允许成交。"""
    P = daily_params(cfg)
    if gap_pct < P["gap_min"]:
        return False, "低开 %.2f%% 破位, 不接" % gap_pct
    if gap_pct > P["gap_max"]:
        return False, "高开 %.2f%%, 不追" % gap_pct
    return True, ""


def should_exit(held_days: int, ret_pct: float, cfg: dict) -> tuple[bool, str]:
    """是否卖出。held_days = 已持有交易日数(买入当日为 0)。"""
    P = daily_params(cfg)
    if P["stop"] < 0 and ret_pct <= P["stop"] * 100:
        return True, "止损 %+.1f%%" % ret_pct
    if held_days >= P["hold_days"]:
        return True, "持满 %d 个交易日" % P["hold_days"]
    return False, ""
