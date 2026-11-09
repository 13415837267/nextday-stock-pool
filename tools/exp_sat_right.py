# -*- coding: utf-8 -*-
"""第183轮: 右侧交易的真正空白点 —— "右侧选股 + 主策略的短持有节奏"

为什么这轮必须重测
------------------
用户: "不可能没有方法实现右侧交易 ... 左侧右侧交易都要实际百倍收益以上, 回撤要小, 交易次数要多"

第 139 轮判过"追突破是接盘", 但那批实验的持有期是 **H5 / H10 / H20**:

| 选股 | 持有期 | 1500 日 |
|---|---|---|
| 阳线突破 H20 | 20 日 | +6.0% / -69.7% |
| 纯均线多头 H5 | 5 日 | +3.2% / -88.6% |
| 趋势过滤+ivol H20 | 20 日 | -6.4% / -72.9% |

**H1 / H2 从来没测过。** 而主策略年化 184%、498 倍, 靠的恰恰是 H1
(T+1 开盘买 / T+2 收盘卖)。也就是说: 主策略证明了"A 股 1~2 日的短周期有 alpha",
副策略却一直在 10 日周期上做右侧, 然后得出"右侧不行"的结论 —— 两者不是同一件事。

本轮把两个维度交叉:  {右侧选股} × {H1/H2/H3 短持有}
  右侧选股三种口径:
    bo      breakout: 涨>=X%、贴 20 日新高、放量、收在日内高位 (全市场, 不受低波池限制)
    mom     全市场 20 日动量最强 (不限低波池)
    lvmom   低波前 10% 内 20 日动量最强 (现行池边界 + 右侧排序)
  另外测两个执行变量: gap_max(高开容忍度) / 大盘闸门开关。

口径: 同 prepare_context、同 run(); 单吊(max_pos=1, buys_per_day=1);
      pullback_pct=0 -> T+1 开盘价直接买(右侧不需要等回踩); gate_entry_buffer=0 隔离变量。
产物: output/exp_sat_right.json
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
    # 隔离变量: 入场缓冲默认关掉(本实验要单独看选股与持有期)
    c["satellite"].setdefault("gate_entry_buffer", 0.0)
    res = sb.run(dict(ctx), c)
    m = res["metrics"]
    return {"总收益%": m["总收益率%"], "年化%": m["年化收益率%"], "回撤%": m["最大回撤%"],
            "卡玛": m["卡玛比率"], "笔数": m["交易次数"], "胜率%": m["胜率%"],
            "单笔%": m["平均单笔收益%"], "盈亏比": m.get("盈亏比"),
            "空仓天": m["空仓天数"], "年度": yearly(res), "退出": exit_mix(res["trades"])}


BASE = {"max_pos": 1, "buys_per_day": 1, "pullback_pct": 0.0,
        "gate_entry_buffer": 0.0, "min_score": 0, "ivol_pct_max": 0.1,
        "sort_by": "mom20_ivol", "hold_days": 10, "trail_pct": 8.0, "stop_pct": -12.0,
        "market_gate": True, "gate_exit": True, "gate_exit_mode": "all"}

BO = {"mode": "breakout", "bo_pct_min": 3.0, "bo_near_high": 1.0,
      "bo_vol_ratio": 1.5, "bo_close_pos": 0.8, "ivol_pct_max": 0.0}

PLAN = [
    # ---- 对照 ----
    ("base_h10",        {}),
    # ---- 右侧: 突破 + 短持有 ----
    ("bo3_h1",          {**BO, "hold_days": 1}),
    ("bo3_h2",          {**BO, "hold_days": 2}),
    ("bo3_h3",          {**BO, "hold_days": 3}),
    ("bo5_h1",          {**BO, "bo_pct_min": 5.0, "hold_days": 1}),
    ("bo5_h2",          {**BO, "bo_pct_min": 5.0, "hold_days": 2}),
    # ---- 右侧: 放高高开容忍(强势票往往高开) ----
    ("bo3_h1_gap10",    {**BO, "hold_days": 1, "gap_max": 10.0}),
    ("bo5_h2_gap10",    {**BO, "bo_pct_min": 5.0, "hold_days": 2, "gap_max": 10.0}),
    # ---- 右侧: 关闸门(右侧策略自带强弱判断) ----
    ("bo3_h1_nogate",   {**BO, "hold_days": 1, "market_gate": False, "gate_exit": False}),
    ("bo3_h1_g10_ng",   {**BO, "hold_days": 1, "gap_max": 10.0,
                         "market_gate": False, "gate_exit": False}),
    # ---- 右侧: 全市场动量最强 ----
    ("mom_h1",          {"mode": "lowvol", "ivol_pct_max": 0.0,
                         "sort_by": "mom20_desc", "hold_days": 1}),
    ("mom_h2",          {"mode": "lowvol", "ivol_pct_max": 0.0,
                         "sort_by": "mom20_desc", "hold_days": 2}),
    # ---- 右侧: 低波池内动量最强(现行池边界) ----
    ("lvmom_h1",        {"mode": "lowvol", "ivol_pct_max": 0.1,
                         "sort_by": "mom20_desc", "hold_days": 1}),
    ("lvmom_h2",        {"mode": "lowvol", "ivol_pct_max": 0.1,
                         "sort_by": "mom20_desc", "hold_days": 2}),
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
        sat = {**BASE, **ov}
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
    p = ROOT / f"output/exp_sat_right_{days}.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"\n写出 {p}", flush=True)


if __name__ == "__main__":
    main()
