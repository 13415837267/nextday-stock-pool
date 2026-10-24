# -*- coding: utf-8 -*-
"""滚动失效监测器(2026-10-24 第三十九轮, 纯告警层)。

背景(第三十二/三十八轮结论): 本策略的edge依赖2024-09后形成的市场微观结构,
十年维度常态期望≈零。因此最有价值的不是继续调参数, 而是回答"这个市场结构
还在吗" —— 用实盘滚动战绩作为失效探针。

设计原则(事前注册, 防挖掘):
  - 窗口: 最近20笔已结算交易(净费用后), 整数锚点不拟合;
    实盘节奏约每月4~6笔, 20笔≈3~5个月, 对"结构退潮"响应足够快
  - 阈值: 全部取整数/半整数锚点(0 / +1 / -1), 不做网格搜索 ——
    策略声称正期望, 滚动均值跌破0即为客观数据与主张背离
  - 行为: 仅告警, 不改仓位/不改承接线 —— 硬治理(强制降仓/停机)属于
    执行逻辑变更, 需按走前纪律单独验证后另行定稿, 本层只负责"发出信号"
"""
from __future__ import annotations


def compute_monitor(settled_rets: list[float], mon_cfg: dict) -> dict:
    """按事前注册阈值评估滚动失效状态。

    settled_rets: 已结算交易的单笔净收益(%)时间序列(旧→新);
    mon_cfg: config.yaml 的 monitor 块(window/watch_mean_pct/warn_mean_pct/halt_mean_pct)。
    """
    window = int(mon_cfg.get("window", 20))
    watch = float(mon_cfg.get("watch_mean_pct", 1.0))
    warn = float(mon_cfg.get("warn_mean_pct", 0.0))
    halt = float(mon_cfg.get("halt_mean_pct", -1.0))

    recent = [float(r) for r in settled_rets[-window:]]
    n = len(recent)
    mean = round(sum(recent) / n, 2) if n else None
    win = round(sum(1 for r in recent if r > 0) / n * 100, 1) if n else None

    if n < window:
        level = "样本不足"
        note = f"已结算{n}/{window}笔, 随实盘积累自动生效(期间维持现有风控纪律)"
    elif mean < halt:
        level = "深红-建议停机复检"
        note = (f"近{window}笔均值{mean}%跌破{halt}%停机线: 建议暂停新开仓, "
                f"复检市场结构与参数前提(尤其2024-09后微观结构是否退潮)")
    elif mean < warn:
        level = "红-失效警报"
        note = (f"近{window}笔均值{mean}%跌破{warn}%: 数据与正期望主张背离, "
                f"建议新开仓降至最小试验仓并暂缓放大")
    elif mean < watch:
        level = "黄-观察带"
        note = (f"近{window}笔均值{mean}%低于{watch}%观察线(回测基准+3.24%的一半), "
                f"缩短复盘周期, 关注承接线成交质量")
    else:
        level = "绿-正常"
        note = f"近{window}笔均值{mean}%/胜率{win}%, 与回测口径相符"

    return {
        "level": level,
        "n": n,
        "window": window,
        "mean_pct": mean,
        "win_pct": win,
        "note": note,
        "thresholds": {"watch_mean_pct": watch, "warn_mean_pct": warn,
                       "halt_mean_pct": halt, "window": window},
    }
