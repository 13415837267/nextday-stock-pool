# -*- coding: utf-8 -*-
"""月度盈利熔断(month_profit_cap)全链路审计(tools/profit_cap_audit.py, 第五十三轮)。

发现: 回测引擎实现了 risk.month_profit_cap(当月净值增益超cap_pct%后买入仓位
×cut_frac, 第八轮为降盈利集中度引入/第十一轮定稿{30, 0.6}), 但实盘链路
(daily_run.py)完全没有该机制 —— t1_decision.position_pct 只算 rf_ramp 的
over_frac, 从不乘 cut_frac。某月净值增益超30%后, 回测后续买入按60%仓执行,
网页指引仍显示100%仓 → 回测与实盘指引行为分叉(第五十二轮rf_ramp展示分叉的
同类问题, 但影响更实质: 直接改变仓位执行, 且回测收益数字包含降仓行为)。

两个诊断(事前注册, 零参数变更):
  A. 触发画像: BASE成交按买入日"当月净值增益>cap_pct"分桶(capped/normal),
     触发月的买入质量如何; 近似口径: 用每日收盘nav近似买入时点equity
     (回测买入时点为开盘估值, 边界日可能误判1~2笔, 不影响方向);
  B. 反证: NO_CAP(month_profit_cap=None) vs BASE, 复利口径对比 ——
     熔断机制的净贡献(第八轮动机是降盈利集中度, 代价是压制复利右尾;
     README三档口径: 稳健1798%/平衡2518%/极限3467%, 熔断是"平滑"设计)。

用法: python3 tools/profit_cap_audit.py   (依赖本地panel缓存, 约4~6分钟)
"""
from __future__ import annotations

import copy
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("BT_FULL_TRADES", "1")   # 需要BASE的trades_all做触发画像

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config                     # noqa: E402
from src.backtest import prepare_context, run_backtest  # noqa: E402

ERA = "20240901"   # era切分锚点, 与第42~52轮一致(按buy_date/equity日期)

VERDICT = ("事前注册: 本审计量化月度盈利熔断的十年画像与净贡献(第八轮为降盈利集中度引入), "
           "无论结果方向均不修改risk参数; 核心产出是实盘链路(daily_run)缺失该机制的量化证据。")


def _stats(trades: list[dict]) -> dict:
    """成交桶统计: 单笔均值/胜率/复利和极值。"""
    if not trades:
        return {"n": 0}
    rets = [float(t["ret_pct"]) for t in trades]
    comp = 1.0
    for r in rets:
        comp *= (1 + r / 100)
    return {"n": len(rets), "avg_pct": round(sum(rets) / len(rets), 3),
            "win_pct": round(100 * sum(r > 0 for r in rets) / len(rets), 1),
            "comp_pct": round((comp - 1) * 100, 2),
            "worst_pct": round(min(rets), 2), "best_pct": round(max(rets), 2)}


def _seg(eq: list[dict], start: str, end: str) -> float | None:
    """equity([{d,nav}])在[start,end]段(YYYYMMDD)的复利收益%。"""
    pts = [x["nav"] for x in eq if start <= x["d"].replace("-", "") <= end]
    if len(pts) < 2:
        return None
    return round((pts[-1] / pts[0] - 1) * 100, 2)


def _yearly(eq: list[dict]) -> dict[str, float]:
    """按自然年收益(跨年衔接用上年末nav; 首年基=窗口首日nav)。"""
    years: dict[str, list[float]] = {}
    for x in eq:
        years.setdefault(x["d"][:4], []).append(x["nav"])
    out, prev_last = {}, None
    for y in sorted(years):
        arr = years[y]
        base = prev_last if prev_last else arr[0]
        out[y] = round((arr[-1] / base - 1) * 100, 2)
        prev_last = arr[-1]
    return out


def _metrics(bt: dict) -> dict:
    m = bt["metrics"]
    keep = ("总收益率%", "年化收益率%", "最大回撤%", "卡玛比率", "夏普比率",
            "交易次数", "胜率%", "平均单笔收益%", "空仓天数")
    return {k: m.get(k) for k in keep}


def month_gain_series(eq: list[dict], initial_capital: float) -> dict[str, float]:
    """逐日"当月净值增益%"(精确复刻回测月初基准=前一交易日收盘nav;
    窗口首日基准=initial_capital)。键=YYYYMMDD, 值=(nav/月基准-1)*100。"""
    out: dict[str, float] = {}
    prev_nav: float | None = None
    month_start: float | None = None
    month_key = None
    for x in eq:
        d = x["d"].replace("-", "")
        ym = d[:6]
        if ym != month_key:
            month_key = ym
            month_start = prev_nav if prev_nav is not None else initial_capital
        if month_start:
            out[d] = (x["nav"] / month_start - 1) * 100
        prev_nav = x["nav"]
    return out


def main():
    cfg = load_config()
    days = int(os.environ.get("PROFIT_CAP_AUDIT_DAYS", "2500"))
    mc = (cfg.get("risk") or {}).get("month_profit_cap") or {}
    cap_pct, cut_frac = float(mc.get("cap_pct", 0) or 0), float(mc.get("cut_frac", 1.0) or 1.0)
    print(f"[1/3] prepare_context({days}) ... (月度熔断: cap={cap_pct}% cut={cut_frac})")
    t0 = time.time()
    ctx = prepare_context(cfg, days)

    print("[2/3] BASE 回测(含月度熔断) ...")
    base = run_backtest(cfg, ctx=ctx)
    cfg_no = copy.deepcopy(cfg)
    cfg_no["risk"]["month_profit_cap"] = None   # mc_cap=0 -> 熔断永不触发
    print("[3/3] NO_CAP 回测(删除月度熔断) ...")
    nocap = run_backtest(cfg_no, ctx=ctx)

    eq_b = base["equity"]
    gains = month_gain_series(eq_b, float(base["meta"].get("initial_capital", 1e6)))

    # ---- 诊断A: BASE成交按买入日当月增益分桶 ----
    trades = [t for t in base.get("trades_all", []) if t.get("ret_pct") is not None]
    capped, normal = [], []
    for t in trades:
        g = gains.get(t["buy_date"])
        t["_month_gain"] = round(g, 1) if g is not None else None
        (capped if (g is not None and g > cap_pct) else normal).append(t)
    cap_months = sorted({t["buy_date"][:6] for t in capped})
    by_month = {ym: _stats([t for t in capped if t["buy_date"][:6] == ym]) for ym in cap_months}
    era_c = {"before_20240901": _stats([t for t in capped if t["buy_date"] < ERA]),
             "after_20240901": _stats([t for t in capped if t["buy_date"] >= ERA])}
    years_c = sorted({t["buy_date"][:4] for t in capped})
    by_year_c = {y: _stats([t for t in capped if t["buy_date"].startswith(y)]) for y in years_c}

    # ---- 诊断B: BASE vs NO_CAP ----
    eq_n = nocap["equity"]
    first, last = eq_b[0]["d"].replace("-", ""), eq_b[-1]["d"].replace("-", "")
    compare = {
        "base": _metrics(base), "no_cap": _metrics(nocap),
        "era": {"base": {"before_20240901": _seg(eq_b, first, "20240831"),
                         "after_20240901": _seg(eq_b, ERA, last)},
                "no_cap": {"before_20240901": _seg(eq_n, first, "20240831"),
                           "after_20240901": _seg(eq_n, ERA, last)}},
        "by_year": {"base": _yearly(eq_b), "no_cap": _yearly(eq_n)},
    }

    out = {
        "meta": {"run_time": time.strftime("%Y-%m-%d %H:%M:%S"), "window_days": days,
                 "days": len(eq_b), "cap_pct": cap_pct, "cut_frac": cut_frac,
                 "capped_trades_n": len(capped), "cap_months": cap_months,
                 "判据": VERDICT,
                 "口径": ("诊断A: BASE十年成交按买入日当月净值增益分桶(收盘nav近似买入时点"
                          "equity, 边界日可能误判); 诊断B: 同ctx复用prepare_context仅删"
                          "risk.month_profit_cap重跑, 复利口径精确。")},
        "profile": {"normal_trades": _stats(normal), "capped_trades": _stats(capped),
                    "capped_era": era_c, "capped_by_year": by_year_c,
                    "capped_by_month": by_month, "capped_trades": capped},
        "compare": compare,
    }
    out_path = ROOT / "output" / "profit_cap_audit.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float),
                        encoding="utf-8")
    print(f"写出 {out_path} (耗时{time.time()-t0:.0f}s)")
    print(f"\n=== 快报 ===")
    print(f"熔断触发月: {cap_months or '无'} | 触发期成交{len(capped)}笔 vs 正常{len(normal)}笔")
    print(f"诊断A capped: {_stats(capped)}")
    print(f"诊断A normal: {_stats(normal)}")
    print(f"诊断B BASE  : {compare['base']}")
    print(f"诊断B NO_CAP: {compare['no_cap']}")
    print(f"诊断B era   : {compare['era']}")


if __name__ == "__main__":
    main()
