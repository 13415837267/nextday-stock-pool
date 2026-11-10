# -*- coding: utf-8 -*-
"""第184轮·B: 文献因子直接当副策略排序键 / 池边界, 双窗口验证

背景
----
exp_web_ic 已确认五个文献因子(MAX/IMAX/隔夜跳空/CTC/低换手)在对未来 10 日收益的
截面 RankIC 上全部为负且显著 —— 即"值越小越好"。但 IC 显著 ≠ 策略能赚:
副策略是**单吊 1 只 + 持 10 日**, 只吃排序的第 1 名, 对尾部噪声极敏感。
本脚本把这五个键逐个替换现役的 mom20_ivol, 在完整现役口径(闸门缓冲/回踩限价/止损)
下比较, 并顺带测"扩池"(ivol 分位 10%→20%/30%)—— 用户要的"交易次数多"。

口径铁律: 所有变体都传 entry_ok(第181轮入场缓冲), 否则会和 42.2% 的基座不可比。

用法: python tools/exp_web_sort.py 1500
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
    return {"总收益%": m["总收益率%"], "年化%": m["年化收益率%"], "回撤%": m["最大回撤%"],
            "卡玛": m["卡玛比率"], "笔数": m["交易次数"], "胜率%": m["胜率%"],
            "单笔%": m["平均单笔收益%"], "空仓天": m["空仓天数"], "年度": yearly(res)}


FULL = {"mode": "lowvol", "max_pos": 1, "buys_per_day": 1, "hold_days": 10,
        "ivol_pct_max": 0.1, "sort_by": "mom20_ivol", "min_score": 0,
        "pullback_pct": 1.0, "pullback_days": 3, "pullback_mode": "low",
        "market_gate": True, "gate_entry_buffer": 0.3,
        "gate_exit": True, "gate_exit_mode": "all",
        "trail_pct": 8.0, "stop_pct": -12.0, "min_pool": 20,
        "exit_limit_pct": 0.0}

PLAN = [
    ("base",        {}),
    ("ctc_asc",     {"sort_by": "ctc_asc"}),
    ("imax_asc",    {"sort_by": "imax_asc"}),
    ("max20_asc",   {"sort_by": "max20_asc"}),
    ("on_abs10",    {"sort_by": "on_abs10_asc"}),
    ("turn20_asc",  {"sort_by": "turn20_asc"}),
    # 反向对照: 若"越小越好"成立, 取反应当明显变差
    ("ctc_desc",    {"sort_by": "ctc_desc"}),
    # 次数: 缩短持有期(用户要"交易次数多")
    ("ctc_h5",      {"sort_by": "ctc_asc", "hold_days": 5}),
    ("imax_h5",     {"sort_by": "imax_asc", "hold_days": 5}),
    ("ctc_h3",      {"sort_by": "ctc_asc", "hold_days": 3}),
    # 池边界: 10% -> 20% / 30%(可选面变宽, 但单票纯度下降)
    ("ctc_pool20",  {"sort_by": "ctc_asc", "ivol_pct_max": 0.2}),
    ("ctc_pool30",  {"sort_by": "ctc_asc", "ivol_pct_max": 0.3}),
]


def main():
    cfg = load_config()
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    print(f"\n===== 窗口 {days} 日 =====", flush=True)
    ctx = prepare_context(cfg, days)
    print(f"实际 {len(ctx['main_dates'])} 日 {ctx['main_dates'][0]} ~ {ctx['main_dates'][-1]}",
          flush=True)
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
            print(f"{nm:12s} ERROR {ex}", flush=True)
            continue
        w["variants"][nm] = r
        print(f"{nm:12s} 总{r['总收益%']:>11.1f}% 年化{r['年化%']:7.1f}% "
              f"回撤{r['回撤%']:7.1f}% 卡玛{r['卡玛']:6.2f} 笔{r['笔数']:4d} "
              f"胜{r['胜率%']:5.1f}% 单笔{r['单笔%']:+.2f}%", flush=True)
    p = ROOT / f"output/exp_web_sort_{days}.json"
    p.write_text(json.dumps({"windows": {str(days): w}}, ensure_ascii=False, indent=1,
                            default=float), encoding="utf-8")
    print(f"\n写出 {p}", flush=True)


if __name__ == "__main__":
    main()
