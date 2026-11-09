# -*- coding: utf-8 -*-
"""第183轮·D: "百倍"的算术 —— 杠杆扫描 + 全史验证

为什么测杠杆
------------
前三轮把"右侧"三种做法全部证伪(追突破 / 涨停次日 / 趋势回调 + 短持有), 且是单调负的。
剩下的唯一变量是**仓位杠杆** —— 它不改 alpha, 只放大路径。用户要"百倍",
就必须把"百倍需要多少杠杆、代价是什么"算成一张表, 而不是含糊地说"不可行"。

复利恒等式(这套策略的实际约束)
------------------------------
    总对数收益 ≈ 日 alpha × 资金在场天数 × 杠杆
现行副策略: 单笔 +2.55% / 持 10 日 = 0.255%/日, 在场 610/1500 日, 杠杆 1.0
主策略:     单笔 +1.44% / 持 1 日 = 1.44%/日, 在场 704/1500 日, 杠杆 1.25
日 alpha 差 4~5 倍, 这才是 498 倍与 8 倍的来源 —— 不是频率。

百倍需要什么(1500 日 = 6 年, 100 倍 = 年化 115%)
  ① 日 alpha × 3.5(0.255 → 0.90%/日): 右侧做不到(前三轮已证), 左侧已由主策略占满
  ② 在场天数 × 2.5(610 → 1500): 等于取消闸门, 第 181 轮已证会亏
  ③ 杠杆 × 2.8(1.0 → 2.8): 唯一可算的路径, 代价是回撤同倍放大

本脚本给 ③ 的真实数字(含融资利息), 并补现行与 bp_lv_h2 的全史表现。
产物: output/exp_sat_lev2.json
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


def run_one(ctx, cfg, sat: dict):
    c = copy.deepcopy(cfg)
    c.setdefault("satellite", {}).update(sat)
    e = entry_ok_series(ctx.get("reg"), c)
    res = sb.run(dict(ctx), c, entry_ok=e)
    m = res["metrics"]
    return {"总收益%": m["总收益率%"], "倍数": round(1.0 + m["总收益率%"] / 100.0, 2),
            "年化%": m["年化收益率%"], "回撤%": m["最大回撤%"], "卡玛": m["卡玛比率"],
            "笔数": m["交易次数"], "胜率%": m["胜率%"], "单笔%": m["平均单笔收益%"],
            "空仓天": m["空仓天数"], "年度": yearly(res)}


BASE = {"mode": "lowvol", "max_pos": 1, "buys_per_day": 1, "hold_days": 10,
        "ivol_pct_max": 0.1, "sort_by": "mom20_ivol", "min_score": 0,
        "pullback_pct": 1.0, "pullback_days": 3, "pullback_mode": "low",
        "market_gate": True, "gate_entry_buffer": 0.3,
        "gate_exit": True, "gate_exit_mode": "all",
        "trail_pct": 8.0, "stop_pct": -12.0, "min_pool": 20,
        "exit_limit_pct": 0.0, "leverage": 1.0, "margin_rate": 0.055}

BP = {**BASE, "mode": "bo_pullback", "sort_by": "mom20_desc",
      "hold_days": 2, "min_pool": 5}

PLAN_1500 = [
    ("lev1.0",     BASE, {}),
    ("lev1.5",     BASE, {"leverage": 1.5}),
    ("lev2.0",     BASE, {"leverage": 2.0}),
    ("lev2.5",     BASE, {"leverage": 2.5}),
    ("lev3.0",     BASE, {"leverage": 3.0}),
    ("bp_lev1.5",  BP, {"leverage": 1.5}),
    ("bp_lev2.0",  BP, {"leverage": 2.0}),
]
PLAN_3053 = [
    ("lev1.0",     BASE, {}),
    ("lev2.0",     BASE, {"leverage": 2.0}),
    ("bp_lev1.0",  BP, {}),
]


def main():
    cfg = load_config()
    out = {"windows": {}}
    for days, plan in ((1500, PLAN_1500), (3053, PLAN_3053)):
        print(f"\n===== 窗口 {days} 日 =====", flush=True)
        ctx = prepare_context(cfg, days)
        print(f"实际 {len(ctx['main_dates'])} 日 "
              f"{ctx['main_dates'][0]} ~ {ctx['main_dates'][-1]}", flush=True)
        w = {"days": len(ctx["main_dates"]), "start": ctx["main_dates"][0],
             "end": ctx["main_dates"][-1], "variants": {}}
        for nm, base, ov in plan:
            sat = {**base, **ov}
            try:
                r = run_one(ctx, cfg, sat)
            except Exception as ex:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                w["variants"][nm] = {"error": str(ex)}
                print(f"{nm:12s} ERROR {ex}", flush=True)
                continue
            w["variants"][nm] = r
            print(f"{nm:12s} 总{r['总收益%']:>12.0f}% {r['倍数']:>10.1f}x "
                  f"年化{r['年化%']:7.1f}% 回撤{r['回撤%']:7.1f}% 卡玛{r['卡玛']:6.2f} "
                  f"笔{r['笔数']:4d} 胜{r['胜率%']:5.1f}% 单笔{r['单笔%']:+.2f}%", flush=True)
        out["windows"][str(days)] = w
    p = ROOT / "output/exp_sat_lev2.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"\n写出 {p}", flush=True)


if __name__ == "__main__":
    main()
