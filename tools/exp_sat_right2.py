# -*- coding: utf-8 -*-
"""第183轮·B: 右侧交易的第二条路 —— "趋势里的回调" 与 "低波池内短持有"

第一轮(exp_sat_right.py, 1500 日)的结论
----------------------------------------
追突破在 1~3 日周期是**接盘**, 且随持有期变短单调恶化:

| 变体 | 年化 | 单笔 | 笔数 |
|---|---|---|---|
| 突破(涨>=3%, 贴 20 日新高, 放量, 收最高) H1 | -47.9% | -0.66% | 296 |
| 同上 H2 / H3 | -53.7% / -42.2% | -1.49% / -0.74% | 190 / 197 |
| 全市场 20 日动量最强 H1 / H2 | -54.0% / -53.7% | -1.50% / -1.85% | 209 / 169 |
| **低波池内 20 日动量最强 H2** | **+12.3%** | +0.50% | **291** |
| 低波池内 20 日动量最强 H1 | -4.9% | +0.16% | 420 |

两件事: ① 出低波池就是负 alpha —— 说明"A 股短周期追强"必须叠加"低波"这个抗跌前提;
② lvmom_h2 只有 12.3% 年化, 但 **291 笔 vs 现役 106 笔**, 单笔仍为正 —— 它买到的是
"次数", 缺的是"赔率"。而现役 42.2% 的关键是 -1% 回踩限价(第一轮 base 去掉限价后
只有 11.8%), 所以这两者需要叠起来看, 不能分开比较。

本轮变量
--------
A. lvmom(低波池 + 动量最强) × H1/H2/H3, **补齐限价 + 闸门 + 入场缓冲** -> 与现役同口径比较
B. bo_pullback(右侧趋势回调: 站上 MA20 + 动量>0 + 从 20 日高点回撤 1~12% + 缩量)
   —— 这是"右侧"的另一种定义: 买强势股的回调, 不是买突破那根阳线
C. base 复现: 确认脚本口径与线上 42.2%/-21.63%/106 笔一致(否则 A/B 都不能比)

产物: output/exp_sat_right2_<days>.json
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
    res = sb.run(dict(ctx), c)
    m = res["metrics"]
    return {"总收益%": m["总收益率%"], "年化%": m["年化收益率%"], "回撤%": m["最大回撤%"],
            "卡玛": m["卡玛比率"], "笔数": m["交易次数"], "胜率%": m["胜率%"],
            "单笔%": m["平均单笔收益%"], "盈亏比": m.get("盈亏比"),
            "空仓天": m["空仓天数"], "年度": yearly(res), "退出": exit_mix(res["trades"])}


# 现行定稿口径(与 config.yaml 一致), 作为所有变体的公共底座
FULL = {"mode": "lowvol", "max_pos": 1, "buys_per_day": 1, "hold_days": 10,
        "ivol_pct_max": 0.1, "sort_by": "mom20_ivol", "min_score": 0,
        "pullback_pct": 1.0, "pullback_days": 3, "pullback_mode": "low",
        "market_gate": True, "gate_entry_buffer": 0.3,
        "gate_exit": True, "gate_exit_mode": "all",
        "trail_pct": 8.0, "stop_pct": -12.0, "min_pool": 20,
        "exit_limit_pct": 0.0}

PLAN = [
    # 口径复现(必须逐位等于 42.2%/-21.63%/1.95/106)
    ("base",               {}),
    # A. 低波池内动量最强, 短持有, 补齐限价+闸门
    ("lvmom_h1",           {"sort_by": "mom20_desc", "hold_days": 1}),
    ("lvmom_h2",           {"sort_by": "mom20_desc", "hold_days": 2}),
    ("lvmom_h3",           {"sort_by": "mom20_desc", "hold_days": 3}),
    ("lvmom_h2_nogate",    {"sort_by": "mom20_desc", "hold_days": 2,
                            "market_gate": False, "gate_exit": False}),
    # B. 右侧趋势回调(全市场, 不限定低波池)
    ("bp_h1",              {"mode": "bo_pullback", "ivol_pct_max": 0.0,
                            "sort_by": "mom20_desc", "hold_days": 1, "min_pool": 5}),
    ("bp_h2",              {"mode": "bo_pullback", "ivol_pct_max": 0.0,
                            "sort_by": "mom20_desc", "hold_days": 2, "min_pool": 5}),
    ("bp_h3",              {"mode": "bo_pullback", "ivol_pct_max": 0.0,
                            "sort_by": "mom20_desc", "hold_days": 3, "min_pool": 5}),
    # B2. 右侧趋势回调 + 低波池(抗跌前提)
    ("bp_lv_h2",           {"mode": "bo_pullback", "ivol_pct_max": 0.1,
                            "sort_by": "mom20_desc", "hold_days": 2, "min_pool": 5}),
    ("bp_lv_h3",           {"mode": "bo_pullback", "ivol_pct_max": 0.1,
                            "sort_by": "mom20_desc", "hold_days": 3, "min_pool": 5}),
    # B3. 右侧趋势回调 + 更宽松的回撤带(2~15%)
    ("bp_wide_h2",         {"mode": "bo_pullback", "ivol_pct_max": 0.0,
                            "bp_high_min": -15.0, "bp_high_max": -2.0,
                            "sort_by": "mom20_desc", "hold_days": 2, "min_pool": 5}),
]


def main():
    cfg = load_config()
    out = {"windows": {}}
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
    out["windows"][str(days)] = w
    p = ROOT / f"output/exp_sat_right2_{days}.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"\n写出 {p}", flush=True)


if __name__ == "__main__":
    main()
