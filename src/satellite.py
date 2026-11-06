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

候选池 = 评分达标, 最多 3 只(第一百四十九轮)
------------------------------------------------
用户要求"不能备选二十个, 要有一个评分阈值达标最多 3 个"。改法:

  · 评分(score) = 100 × (1 − ivol20 / IVOL_REF), IVOL_REF=1.5(百分点口径),
    ivol 越低评分越高: 0 波动=100 分, ivol≥1.5=0 分;
  · 候选池 = 池内评分 ≥ min_score 的前 3 名(0~3 只), 一个都不达标则当日空仓;
  · min_score=20 分(ivol≤1.2)由回测训练确定: 更严的线(30/40/50 分)会持续砍掉
    盈利交易(见 CHANGELOG 第一百四十九轮明细), 阈值只作极端波动保险丝;
  · 阈值只判定"能不能进池", 不再无差别列出第 2~20 名备选 —— 备选折叠已删除(pool_alt=0)。

三层口径(页面 legend / 漏斗 / 账本逐字对应):

  base(基础票池) = 通过硬性条件(主板/非ST/非停牌/上市满250日/成交额≥1亿)的全部只数,
                   约 1800 只 —— 只用来画漏斗, 以及检查样本量是否够格;
  pool(候选池)   = base 中评分达标的**前 3 名**, pool_show=3, 页面最多列 3 行;
  score          = 100×(1−ivol/1.5), 阈值 min_score 见 config.yaml 的 satellite 段。

执行仍是单吊: 只买候选池第 1 名(评分最高), 第 2/3 名是候补展示、不参与执行。

min_pool(20) 是**样本下限**不是池深: 基础票池不足 20 只说明当日可选面太窄
(极端行情/大面积停牌), 单吊不下注, 当日空仓 —— 与"候选池最多 3 只"不矛盾。
页面 legend 与漏斗都由 pool_funnel() 给出, 与 rank_pool() 同一套掩码, 不各算各的。

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
POOL_SHOW = 3             # 候选池最多 3 只(评分达标的前 3 名, 可能 0~3 只)
POOL_ALT = 0              # 备选折叠已删除: 达标才进池, 不达标不列
IVOL_REF = 1.5            # 评分基准(百分点): ivol=0 -> 100 分, ivol>=1.5 -> 0 分

DEFAULT = {
    "mode": "lowvol",         # lowvol | trend_lowvol
    "hold_days": 10,          # 持有交易日数(第一百五十轮 40->10, 见 config.yaml 注释)
    "min_score": 20.0,        # 评分阈值(0=不设): 评分 < 阈值不进候选池, 一个都不达标则空仓
    "max_ivol": 0,            # 兼容旧口径: 直接给 ivol 上限(0=不用); min_score>0 时以评分为准
    # 第一百五十轮新增: 深度回调门槛(距20日最高点的跌幅, 负值%)
    #   训练结论: 低波池里再要求"距20日最高 <= -8%", H20 胜率 47.0% -> 52.8%,
    #   中位 -0.30% -> +0.56%, 且 -5/-8/-10/-12/-15 是单调平台而非尖峰。
    #   None = 不启用(保持旧行为)。
    "dist_high20_max": None,
    # 第一百五十轮新增: 基础池内 ivol 分位上限(0=不启用)。
    #   0.10 = 只在当日 ivol 最低的 10% 里选, 比绝对阈值 ivol<=1.2 更适应波动率环境变化。
    "ivol_pct_max": 0.0,
    # 第一百五十一轮: 距 MA5 的距离上限(%), None=不启用。
    #   训练(178 万样本, 训练/验证分段): "跌破 MA5 越深, 后续 10 日收益越高" 是**连续单调**
    #   的, 不是某个切点的偶然 —— <=-2% +1.36%、<=-3% +2.60%、<=-4% +4.19%、<=-6% +8.54%、
    #   <=-7% +10.46%, 训练段与验证段同向且胜率从 49% 单调升到 85%。
    #   -3% 是"频率 x 质量"的平衡点(日均 0.18 个信号), 更深则样本太少、年化反而被噪声主导。
    "dist_ma5_max": None,
    # 第一百五十一轮: 并发持仓上限。单吊 100% 仓位的日波动就是一只票的波动, 回撤压不住;
    #   资金分 max_pos 份、每日最多买 1 只, 卖出后当日收盘再出信号。
    #   训练: max_pos=2 年化 18.7% / 回撤 -19.3% / 卡玛 0.97; 3 只降到 11.0%/0.67
    #   (第 3 只票平均贡献 -0.37%, 是负的 —— 同一天出现 3 个深度回调信号说明是普跌)。
    "max_pos": 1,
    # 第一百五十轮新增: 排序键(从低波池里挑哪一只), 默认 ivol=现行行为。
    "sort_by": "ivol",
    "amount_min_yi": 1.0,     # 成交额下限(亿)
    "cost_pct": 0.3,          # 往返成本(%)
    "gap_min": -3.0,          # T+1 竞价缺口下限(%): 深低开破位不接
    "gap_max": 5.0,           # 上限: 高开过多不追
    "stop_pct": 0.0,          # 止损(%): 0=关闭; 收盘相对买入价跌破即卖
    "market_gate": False,     # 大盘闸门: 指数在 MA20 上方才开新仓
    "min_pool": 20,           # 基础票池样本下限, 不足则不出票(与池深无关)
    "pool_show": POOL_SHOW,   # 候选池只数上限(评分达标的前 N 名)
    "pool_alt": POOL_ALT,     # 备选排序深度(已废弃, 固定 0)
}

STRATEGY_CN = {"trend_lowvol": "趋势低波单吊", "lowvol": "低波单吊"}


def score_of(ivol) -> float:
    """ivol20 -> 0~100 评分(越高越稳)。"""
    try:
        x = float(ivol)
    except Exception:  # noqa: BLE001
        return 0.0
    if x != x:  # NaN
        return 0.0
    return max(0.0, min(100.0, 100.0 * (1.0 - x / IVOL_REF)))


def daily_params(cfg: dict) -> dict:
    s = {**DEFAULT, **(cfg.get("satellite") or {})}
    ms = float(s.get("min_score") or 0)
    cap = float(s.get("max_ivol") or 0)
    if ms > 0:                                  # 评分阈值优先: 换算成 ivol 上限
        cap = IVOL_REF * (1.0 - min(ms, 100.0) / 100.0)
    return {"mode": str(s["mode"]),
            "hold_days": int(s["hold_days"]),
            "min_score": ms,
            "ivol_cap": cap,
            "max_ivol": cap,
            "dist_high20_max": (None if s.get("dist_high20_max") is None
                                else float(s["dist_high20_max"])),
            "dist_ma5_max": (None if s.get("dist_ma5_max") is None
                             else float(s["dist_ma5_max"])),
            "max_pos": max(1, int(s.get("max_pos") or 1)),
            "ivol_pct": float(s.get("ivol_pct_max") or 0.0),
            "sort_by": str(s.get("sort_by") or "ivol"),
            "amt_min": float(s["amount_min_yi"]) * 1e8,
            "cost": float(s["cost_pct"]) / 100.0,
            "gap_min": float(s["gap_min"]),
            "gap_max": float(s["gap_max"]),
            "stop": float(s["stop_pct"]) / 100.0,
            "gate": bool(s["market_gate"]),
            "min_pool": int(s["min_pool"]),
            "pool_show": int(s["pool_show"] or POOL_SHOW),
            "pool_alt": max(0, int(s["pool_alt"] or POOL_ALT))}   # 备选已废弃, 恒 0


def apply_cost(nav_before: float, turn: float, cost: float = 0.0030) -> float:
    """按换手率扣成本(单边)。turn=1.0 表示全部换手一次。"""
    return float(nav_before) * (1.0 - float(cost) * float(turn))


def _masks(day, cfg: dict):
    """逐道筛选掩码, 供 rank_pool 与 pool_funnel 共用(同源, 不各算各的)。

    顺序: 上市状态 → 成交额/波动 → 板块白名单 → (可选)趋势过滤。
    注意: 评分阈值**不在这里过滤** —— 阈值只决定"能不能进候选池", 基础票池保持完整,
    否则阈值一严, 漏斗最后一步会缩水, 读者分不清是"票不够稳"还是"可选面太窄"。
    返回 (m_listed, m_liquid, m_final), 均为与 day 等长的布尔数组。
    """
    from src.config import board_of

    P = daily_params(cfg)
    uni = cfg.get("universe") or {}
    boards = set(uni.get("boards") or ["main", "gem"])

    m_listed = ((day["is_paused"] == 0) & (day["is_st"] == 0)
                & (day["bars"] >= MIN_BARS)).to_numpy()
    m_liquid = m_listed & ((day["amount"] >= P["amt_min"]) & (day["ivol20"] > 0)).to_numpy()
    m_final = m_liquid & day["ts_code"].map(lambda c: board_of(c) in boards).to_numpy()
    if P["mode"] == "trend_lowvol":
        m_final = m_final & ((day["close_adj"] > day["ma5"]) & (day["ma5"] > day["ma10"])
                             & (day["ma10"] > day["ma20"]) & (day["ma20"] > day["ma20_prev"])
                             & (day["mom20"] > 0)).to_numpy()
    return m_listed, m_liquid, m_final


def rank_pool(day, cfg: dict):
    """当日基础票池, 已按 sort_by 指定的键升序。池子不足 min_pool 返回空表。

    sort_by 决定"从低波池里挑哪一只"(第一百五十轮参数化, 默认 ivol 与旧行为一致):
      ivol       —— 20日特质波动最低(现行)
      amount     —— 成交额最小(小市值/冷门)
      dist_high20—— 距20日最高点跌幅最深(超跌)
      mom20_asc  —— 20日动量最弱(超跌反转)
      mom20_desc —— 20日动量最强(追动量, 对照组)
    """
    P = daily_params(cfg)
    df = day.loc[_masks(day, cfg)[2]]
    if len(df) < P["min_pool"]:
        return df.iloc[0:0]
    key = P["sort_by"]
    if key == "amount":
        s = df["amount"].astype(float)
    elif key == "dist_high20":
        s = df["dist_high20"].astype(float)
    elif key == "dist_ma5":
        s = df["dist_ma5"].astype(float)
    elif key == "mom20_asc":
        s = df["mom20"].astype(float)
    elif key == "mom20_desc":
        s = -df["mom20"].astype(float)
    else:
        s = df["ivol20"].astype(float)
    return df.assign(_ivol=df["ivol20"].astype(float), _key=s).sort_values("_key")


def rank_qualified(day, cfg: dict, rk=None):
    """评分达标的票池: rank_pool 之后按「评分阈值 + ivol 分位 + 深回调」过滤, 仍按 ivol 升序。

    三道门槛都在这里(而不是 _masks), 因为它们是"候选资格"而非"票池边界":
      · ivol_cap      —— 绝对阈值, 对应页面上的评分线;
      · ivol_pct      —— 当日横截面分位, 适应波动率环境变化(绝对阈值在低波动年份会失效);
      · dist_high20_max —— 距 20 日最高点跌幅, 剔除"还在高位"的票(训练见第一百五十轮)。
    rk 已按 ivol 升序, 所以取前 n 名即当日最低分位。
    """
    P = daily_params(cfg)
    rk = rank_pool(day, cfg) if rk is None else rk
    q = rk
    # 顺序: 先限定"池边界"(ivol 分位), 再判"资格"(评分线), 最后叠加形态条件。
    # 分位必须按 ivol 的**数值**取, 不能用 head(n) —— rk 已按 sort_by 排序(可能是 dist_ma5),
    # head(n) 会取成"跌得最深的前 n 只", 池边界就变成了排序结果, 与分位定义相反。
    if len(q) and P["ivol_pct"] > 0:
        cut = float(rk["_ivol"].quantile(P["ivol_pct"]))
        q = q[q["_ivol"] <= cut]
    if len(q) and P["ivol_cap"] > 0:
        q = q[q["_ivol"] <= P["ivol_cap"]]
    if len(q) and P["dist_ma5_max"] is not None:
        q = q[q["dist_ma5"] <= P["dist_ma5_max"]]
    if len(q) and P["dist_high20_max"] is not None:
        q = q[q["dist_high20"] <= P["dist_high20_max"]]
    return q


def pool_funnel(day, cfg: dict) -> list:
    """页面漏斗: 每一步"还剩多少只"。数字与 rank_pool 同源, 不是另写一套条件。"""
    m_listed, m_liquid, m_final = _masks(day, cfg)
    P = daily_params(cfg)
    tail = "沪深主板白名单"
    if P["mode"] == "trend_lowvol":
        tail += " + 均线多头"
    return [{"label": "全市场", "n": int(len(day))},
            {"label": "非ST·非停牌·上市满%d日" % MIN_BARS, "n": int(m_listed.sum())},
            {"label": "成交额≥%.1f亿" % (P["amt_min"] / 1e8), "n": int(m_liquid.sum())},
            {"label": tail, "n": int(m_final.sum())}]


def pick_one(day, cfg: dict) -> dict | None:
    """当日唯一执行候选; 基础票池不足 min_pool 或评分不达标时返回 None(宁缺勿滥)。"""
    rk = rank_pool(day, cfg)
    P = daily_params(cfg)
    if len(rk) < P["min_pool"]:
        return None
    q = rank_qualified(day, cfg, rk)
    if not len(q):
        return None
    r = q.iloc[0]
    base = int(len(rk))
    return {"code": str(r["ts_code"]),
            "ivol20": round(float(r["_ivol"]), 3),
            "score": round(score_of(r["_ivol"]), 1),
            "pct": round(float(r["pct"]), 2),
            "mom20": round(float(r["mom20"]), 1),
            "close": round(float(r["close"]), 2),
            "amount_yi": round(float(r["amount"]) / 1e8, 2),
            # 池深随候选一起带出: 回放要解释"这天为什么没买", 需要知道全池排序的结果,
            # 不是再算一遍 rank_pool(重复排序 1500 次)。
            # pool_n = 候选池只数(评分达标的前 pool_show 名, 0~3);
            # pool_base = 基础票池只数(硬性条件之后的全部), 回放用它判断样本量是否够格。
            "pool_n": min(len(q), P["pool_show"]),
            "pool_base": base}


def pick_list(day, cfg: dict, held=(), k: int = 1) -> list[dict]:
    """当日可执行候选列表(按排序键升序、跳过已持有的), 最多 k 只。

    与 pick_one 共用同一套掩码/排序/阈值 —— 组合化之后"候选池"与"执行"必须是
    同一份列表, 不能各算一套(否则页面列 3 只、实际只买 1 只, 又是两套口径)。
    """
    rk = rank_pool(day, cfg)
    P = daily_params(cfg)
    if len(rk) < P["min_pool"]:
        return []
    q = rank_qualified(day, cfg, rk)
    if not len(q):
        return []
    skip = set(held or ())
    out = []
    for _, r in q.iterrows():
        code = str(r["ts_code"])
        if code in skip:
            continue
        out.append({"code": code,
                    "ivol20": round(float(r["_ivol"]), 3),
                    "score": round(score_of(r["_ivol"]), 1),
                    "dist_ma5": round(float(r["dist_ma5"]), 2),
                    "pct": round(float(r["pct"]), 2),
                    "mom20": round(float(r["mom20"]), 1),
                    "close": round(float(r["close"]), 2),
                    "amount_yi": round(float(r["amount"]) / 1e8, 2)})
        if len(out) >= k:
            break
    return out


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
