# -*- coding: utf-8 -*-
"""第183轮·C: 校准 base(补 entry_ok) + 两条右侧支线收口

第二轮(exp_sat_right2)发现一个口径漏洞: 脚本没把入场缓冲传进 run(),
所以 base 复现的是第 180 轮的 28.1%/-24.2%/120 笔, 不是现役的 42.2%。
本脚本补上 entry_ok_series —— 所有变体都在"完整现役口径"下比较, 否则不可比。

已排除(不重复测)
----------------
| 右侧做法 | 1500 日 | 结论 |
|---|---|---|
| 追突破(涨>=3%/5%、贴 20 日新高、放量、收最高) H1/H2/H3 | 年化 -36~-54%、单笔 -0.3~-1.9% | 接盘 |
| 全市场 20 日动量最强 H1/H2 | -54% / -54% | 出低波池即负 |
| 右侧趋势回调(全市场) H1/H2/H3 | -15% / -45% / -37% | 出低波池即负 |
| 右侧趋势回调 + 低波池 H2/H3 | +20.1% / +13.2%(无缓冲) | 唯一正, 待校准 |
| 低波池动量最强 H1/H2/H3 | -4.9% / +2.8% / -6.7%(无缓冲) | 短持有摊薄单笔 |

本轮三件事
----------
A. base 校准到 42.2%(补 entry_ok), 让所有对比有统一基准
B. 涨停次日(limit_up): 当日收盘封板 + 开盘未封(盘中打开过) -> T+1 开盘买, 持 H 日
   —— A 股最经典的右侧短线, 与"追突破"测的是不同的资金行为, 此前从未测过
C. bp_lv(右侧回调 + 低波池)校准后 H2/H3/H4, 看它是不是"次数翻倍"的可行解

产物: output/exp_sat_right3_<days>.json
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context  # noqa: E402
from src.config import load_config  # noqa: E402
from src.satellite import entry_ok_series  # noqa: E402
import tools.satellite_backtest as sb  # noqa: E402


def yearly(res: dict) -> dict:
    out = {}
    for pt in res["equity"]:
        y = pt["d"][:4]
        out.setdefault(y, [pt["nav"], pt["nav"]])
        out[y][1] = pt["nav"]
    return {y: round((v[1] / v[0] - 1) * 100, 1) for y, v in out.items()}


def exit_mix(trades: list) -> dict:
    out = {}
    for t in trades:
        k = str(t["note"]).split(" ")[0]
        d = out.setdefault(k, {"n": 0, "sum": 0.0})
        d["n"] += 1
        d["sum"] += float(t["ret_pct"] or 0)
    return {k: {"n": v["n"], "avg": round(v["sum"] / v["n"], 2)} for k, v in out.items()}


def run_one(ctx, cfg, sat: dict):
    c = copy.deepcopy(cfg)
    c.setdefault("satellite", {}).update(sat)
    # 关键: 入场缓冲(第 181 轮)由调用方算好传入, run() 内部不读 config
    e = entry_ok_series(ctx.get("reg"), c)
    res = sb.run(dict(ctx), c, entry_ok=e)
    m = res["metrics"]
    return {"总收益%": m["总收益率%"], "年化%": m["年化收益率%"], "回撤%": m["最大回撤%"],
            "卡玛": m["卡玛比率"], "笔数": m["交易次数"], "胜率%": m["胜率%"],
            "单笔%": m["平均单笔收益%"], "盈亏比": m.get("盈亏比"),
            "空仓天": m["空仓天数"], "年度": yearly(res), "退出": exit_mix(res["trades"])}


FULL = {"mode": "lowvol", "max_pos": 1, "buys_per_day": 1, "hold_days": 10,
        "ivol_pct_max": 0.1, "sort_by": "mom20_ivol", "min_score": 0,
        "pullback_pct": 1.0, "pullback_days": 3, "pullback_mode": "low",
        "market_gate": True, "gate_entry_buffer": 0.3,
        "gate_exit": True, "gate_exit_mode": "all",
        "trail_pct": 8.0, "stop_pct": -12.0, "min_pool": 20,
        "exit_limit_pct": 0.0}

LU = {"mode": "limit_up", "ivol_pct_max": 0.0, "sort_by": "amount",
      "hold_days": 1, "min_pool": 3, "trail_pct": 0.0, "stop_pct": -10.0}

PLAN = [
    # A. 校准(必须 = 42.2% / -21.63% / 1.95 / 106)
    ("base",              {}),
    # C. 右侧回调 + 低波池, 校准后
    ("bp_lv_h2",          {"mode": "bo_pullback", "sort_by": "mom20_desc",
                           "hold_days": 2, "min_pool": 5}),
    ("bp_lv_h3",          {"mode": "bo_pullback", "sort_by": "mom20_desc",
                           "hold_days": 3, "min_pool": 5}),
    ("bp_lv_h4",          {"mode": "bo_pullback", "sort_by": "mom20_desc",
                           "hold_days": 4, "min_pool": 5}),
    # B. 涨停次日
    ("lu_h1",             {**LU}),
    ("lu_h2",             {**LU, "hold_days": 2}),
    ("lu_h1_gap10",       {**LU, "gap_max": 10.0}),
    ("lu_h1_gap12",       {**LU, "gap_max": 12.0}),
    ("lu_h2_gap10",       {**LU, "hold_days": 2, "gap_max": 10.0}),
    ("lu_h1_nogate",      {**LU, "market_gate": False, "gate_exit": False}),
    ("lu_h1_g10_ng",      {**LU, "gap_max": 10.0, "market_gate": False, "gate_exit": False}),
    ("lu_h1_turn5",       {**LU, "lu_turn_min": 5.0}),
]


def main():
    cfg = load_config()
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    print(f"\n===== 窗口 {days} 日 =====", flush=True)
    ctx = prepare_context(cfg, days)
    print(f"实际 {len(ctx['main_dates'])} 日 {ctx['main_dates'][0]} ~ {ctx['main_dates'][-1]}", flush=True)
    w = {"days": len(ctx["main_dates"]), "start": ctx["main_dates"][0],
         "end": ctx["main_dates"][-1], "variants": {}}
    for nm, ov in PLAN:
        sat = {**FULL, **ov}
        try:
            r = run_one(ctx, cfg, sat)
        except Exception as ex:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            w["variants"][nm] = {"error": str(ex)}
            print(f"{nm:16s} ERROR {ex}", flush=True)
            continue
        w["variants"][nm] = r
        print(f"{nm:16s} 总{r['总收益%']:>11.1f}% 年化{r['年化%']:7.1f}% "
              f"回撤{r['回撤%']:7.1f}% 卡玛{r['卡玛']:6.2f} 笔{r['笔数']:4d} "
              f"胜{r['胜率%']:5.1f}% 单笔{r['单笔%']:+.2f}% 空仓{r['空仓天']:4d}", flush=True)
    out = {"windows": {str(days): w}}
    p = ROOT / f"output/exp_sat_right3_{days}.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"\n写出 {p}", flush=True)


if __name__ == "__main__":
    main()
