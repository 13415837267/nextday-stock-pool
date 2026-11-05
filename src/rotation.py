# -*- coding: utf-8 -*-
"""副策略 v4 —— 低波组合轮动(第一百四十五轮)。

为什么从"单吊"改成"组合"
------------------------
因子扫描(output/exp_factor_scan.json, 1500 日 69 个调仓点)的结论:

  · 10 个因子里低波(ivol20)最强: 五组完全单调, Q1-Q5 多空 2.26%/20日, IC -0.131;
    动量是负的(Q5 低波组 -1.16% 对 Q1 +1.10%), 换赛道换不到更好的地方。
  · 但那个 +1.10% 是**一组约 100 只等权**的均值 —— 单吊等于从 100 只里抽 1 只,
    把年化 14% 的期望丢进一个方差极大的抽样。
  · 实测(1500 日 / 3378 日): 单吊 +50.88%/-49.13%/卡玛 0.14;
    低波 20 只等权、20 日调仓 +55.50%/-27.78%/0.27(全史 12.79%/-40.13%/0.32)。
    收益不变、回撤减半、卡玛翻倍 —— 差别全部来自"不再抽样"。

规则
----
· 池子: 与主策略共用一份面板与硬性条件(沪深主板/非ST/非停牌/上市满250日/
  成交额≥1亿/ivol20>0) —— 不单独抓数、不单独建库。
· 选股: 池内按 20 日特质波动(ivol20)升序取前 n_hold 只, 等权。
· 调仓: 每 hold_days 个交易日一次, T 日收盘出清单 → T+1 开盘执行;
  仍在新榜里的票不动(只补差额), 出榜的卖、新进榜的买。
· 不做大盘闸门: 回测里加闸门(R6)反而更差 —— 低波组合本身的 beta 已经不高,
  再叠加择时是在用交易成本换一个已经被因子吃掉的信号。
· 无未来函数: 排序只用 T 日收盘列, 成交用 T+1 开盘价。

本模块零耦合: 不 import backtest / daily_run / selector。
"""
from __future__ import annotations

from src.config import board_of

STRATEGY_CN = "低波轮动"

DEFAULT = {
    "n_hold": 20,             # 持仓只数(等权)
    "hold_days": 20,          # 调仓间隔(交易日)
    "amount_min_yi": 1.0,     # 成交额下限(亿)
    "max_ivol": 0,            # ivol 上限(0=不限)
    "min_pool": 100,          # 基础票池样本下限, 不足则不出新清单(截面太窄分位无意义)
}

MIN_BARS = 250


def params(cfg: dict) -> dict:
    s = {**DEFAULT, **(cfg.get("rotation") or {})}
    return {"n_hold": int(s["n_hold"]),
            "hold_days": int(s["hold_days"]),
            "amt_min": float(s["amount_min_yi"]) * 1e8,
            "max_ivol": float(s["max_ivol"] or 0),
            "min_pool": int(s["min_pool"])}


def base_pool(day, cfg: dict):
    """基础票池(与副策略 v3 的 hard filter 完全一致, 保证两条腿用同一套资格线)。"""
    P = params(cfg)
    uni = cfg.get("universe") or {}
    boards = set(uni.get("boards") or ["main", "gem"])
    m = ((day["is_paused"] == 0) & (day["is_st"] == 0) & (day["bars"] >= MIN_BARS)
         & (day["amount"] >= P["amt_min"]) & (day["ivol20"] > 0))
    if P["max_ivol"] > 0:
        m &= (day["ivol20"] <= P["max_ivol"])
    m &= day["ts_code"].map(lambda c: board_of(c) in boards).to_numpy()
    return day.loc[m]


def rank_basket(day, cfg: dict):
    """当日目标组合, 已按 ivol 升序。池子不足 min_pool 返回空表(宁缺勿滥)。"""
    P = params(cfg)
    df = base_pool(day, cfg)
    if len(df) < P["min_pool"]:
        return df.iloc[0:0]
    return df.assign(_ivol=df["ivol20"].astype(float)).sort_values("_ivol")


def pick_basket(day, cfg: dict, names: dict | None = None) -> list[dict]:
    """返回 n_hold 只候选(第 1 名权重最高? 不 —— 等权, 顺序只用于展示)。"""
    P = params(cfg)
    rk = rank_basket(day, cfg)
    if not len(rk):
        return []
    names = names or {}
    out = []
    for i, (_, r) in enumerate(rk.head(P["n_hold"]).iterrows(), 1):
        code = str(r["ts_code"])
        out.append({"rank": i, "code": code, "name": names.get(code, ""),
                    "ivol20": round(float(r["ivol20"]), 3),
                    "pct": round(float(r["pct"]), 2),
                    "close": round(float(r["close"]), 2),
                    "amount_yi": round(float(r["amount"]) / 1e8, 2),
                    "vol_ratio": round(float(r["vol_ratio"]), 2),
                    "turnover_rate": round(float(r["turnover_rate"]), 2),
                    "mom20": round(float(r["mom20"]), 1),
                    # 等权: 每只 100/n_hold % 仓位
                    "weight_pct": round(100.0 / max(P["n_hold"], 1), 2)})
    return out


def diff_basket(prev: list[str], target: list[str]) -> dict:
    """调仓清单: 卖(在prev不在target)、买(target里prev没有的)、留(两边都有)。"""
    p, t = list(prev or []), list(target or [])
    ts = set(t)
    ps = set(p)
    return {"sell": [c for c in p if c not in ts],
            "buy": [c for c in t if c not in ps],
            "keep": [c for c in p if c in ts]}


def funnel(day, cfg: dict) -> list[dict]:
    """页面漏斗: 每一步还剩多少只(与 base_pool 同源, 不另写条件)。"""
    P = params(cfg)
    m_listed = ((day["is_paused"] == 0) & (day["is_st"] == 0) & (day["bars"] >= MIN_BARS)).to_numpy()
    m_liquid = m_listed & ((day["amount"] >= P["amt_min"]) & (day["ivol20"] > 0)).to_numpy()
    uni = cfg.get("universe") or {}
    boards = set(uni.get("boards") or ["main", "gem"])
    m_final = m_liquid & day["ts_code"].map(lambda c: board_of(c) in boards).to_numpy()
    return [{"label": "全市场", "n": int(len(day))},
            {"label": "非ST·非停牌·上市满%d日" % MIN_BARS, "n": int(m_listed.sum())},
            {"label": "成交额≥%.1f亿" % (P["amt_min"] / 1e8), "n": int(m_liquid.sum())},
            {"label": "沪深主板白名单", "n": int(m_final.sum())}]
