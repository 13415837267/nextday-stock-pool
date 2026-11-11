# -*- coding: utf-8 -*-
"""执行口径文案 —— 回测 / 实盘计划 / 历史回放的**唯一来源**。

为什么单独成模块(第一百二十五轮):
  在此之前, "出场规则"这句话在四处各写一份:
    - daily_run._sell_rule_text()      → 按 execution 动态生成(正确)
    - daily_run.latest["strategy"]     → 硬编码"再次日开盘卖出"(错)
    - backtest.meta["rules"]           → 硬编码"T+2开盘卖出"(错)
    - replay.meta["strategy"]          → 硬编码"开盘浮盈≥阈值则开盘兑现"(错)
  而现行 config 是 execution.sell_open_frac = 0.00(第一百一十五轮取消开盘腿),
  真实口径为"T+2 当日收盘（15:00 收盘集合竞价）全额卖出"（第 200 轮修正：此前文案写
  14:55，而回测一直按日线收盘价 15:00 成交 —— 见 phase-200 的 0.028~0.044%/笔差值）。
  于是同一个页面上,
  顶部横幅写"再次日开盘卖出"、判定卡写"T+2 当日收盘卖出", 互相打脸。

  口径一变就要改四处、且漏改不会有任何报错 —— 这类缺陷只能靠"只留一处"
  根治。本模块只做字符串拼装, 不参与任何交易决策, 因此不存在前视/口径风险。
"""
from __future__ import annotations


def _open_frac(ex: dict) -> float:
    """开盘腿比例, 越界钳到 [0, 1]; 缺省 1.0(全额开盘卖, 与历史行为一致)。"""
    try:
        return min(1.0, max(0.0, float(ex.get("sell_open_frac", 1.0) or 0.0)))
    except (TypeError, ValueError):
        return 1.0


def _deep_tail(ex: dict) -> str:
    """开盘腿的"深跌改收盘"触发条件描述(仅当确实配置了该保护时)。"""
    spb = ex.get("sell_close_if_profit_below")
    scd = float(ex.get("sell_close_if_open_drop") or 0)
    if spb is not None:
        return f"相对买入价浮亏超 {abs(float(spb)):g}%（或开盘跌停）"
    if scd > 0:
        return f"开盘跌幅≥{scd:g}%（或跌停）"
    return "开盘跌停"


def sell_rule_text(ex: dict, short: bool = False) -> str:
    """出场规则一句话。

    long 版(默认): 判定卡/计划卡/规则区, 需讲清"两条腿 + 深跌保护"。
    short 版: 页面横幅/回测 meta 的一行摘要, 只讲腿的比例, 不带条件分支。

    口径(T+2 出场):
      sell_open_frac=1  → 开盘集合竞价全额卖; 深跌/跌停改收盘
      sell_open_frac=0  → 当日收盘(15:00 收盘集合竞价)全额卖, 无开盘腿
      0<f<1             → 开盘卖 f、收盘卖 1-f; 深跌时剩余部分也留到收盘
    """
    f = _open_frac(ex)
    if short:
        if f >= 1.0:
            return "T+2开盘卖出"
        if f <= 0.0:
            return "T+2收盘卖出（15:00）"
        return f"T+2开盘卖{f:.0%}+收盘卖{1 - f:.0%}"
    tail = _deep_tail(ex)
    if 0 < f < 1:
        return (f"T+2 开盘卖 {f:.0%}（9:25 集合竞价）、当日收盘卖 {1 - f:.0%}（15:00）；"
                f"若{tail}，剩余部分也留到收盘")
    if f <= 0:
        return "T+2 当日收盘卖出（15:00 收盘集合竞价，持有整个 T+2 交易日）"
    return f"T+2 开盘集合竞价卖出；若{tail}，改为当日收盘卖出"


def rules_line(ex: dict, tail: str = "") -> str:
    """回测 meta 的一行执行口径: "T+1开盘买入/T+2收盘卖出（15:00）/复利/成本…"。

    tail 用于追加"复利/佣金/印花税/滑点"等成本说明, 由调用方给(回测与回放的成本
    口径可能不同, 不在本模块硬编码)。
    """
    s = f"T+1开盘买入/{sell_rule_text(ex, short=True)}"
    return f"{s}/{tail}" if tail else s
