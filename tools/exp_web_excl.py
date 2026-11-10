# -*- coding: utf-8 -*-
"""第184轮·D: 文献因子当"排除门槛"(不换排序键) —— 副策略

exp_web_sort 的结论是硬的: 五个文献因子直接当排序键, 全部远差于现役 mom20_ivol
(42.2% -> 10.9% / -5.2% / -8.7% / 3.3% / 2.3%)。
解释: 排序键吃的是**极端第 1 名**, 而"最低 MAX / 最低换手 / 最低 CTC"的极端端恰是
无人关注的死水票; IC 衡量的是整个截面的单调性, 与"只买第 1 名"不是一回事。

本脚本测更温和的用法: 排序键不动(仍是 mom20_ivol), 只在当日合格池里砍掉
某个因子最差的尾部 20%/30%, 看单笔与回撤是否改善。

用法: python tools/exp_web_excl.py 1500
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
    ("base",          {}),
    ("ex_max20_20",   {"excl_max20_pct": 0.2}),
    ("ex_max20_30",   {"excl_max20_pct": 0.3}),
    ("ex_onabs_20",   {"excl_onabs_pct": 0.2}),
    ("ex_onabs_30",   {"excl_onabs_pct": 0.3}),
    ("ex_imax_20",    {"excl_imax_pct": 0.2}),
    ("ex_ctc_20",     {"excl_ctc_pct": 0.2}),
    ("ex_max20+onabs", {"excl_max20_pct": 0.2, "excl_onabs_pct": 0.2}),
]


def main():
    cfg = load_config()
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
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
            print(f"{nm:16s} ERROR {ex}", flush=True)
            continue
        w["variants"][nm] = r
        print(f"{nm:16s} 总{r['总收益%']:>10.1f}% 年化{r['年化%']:7.1f}% "
              f"回撤{r['回撤%']:6.1f}% 卡玛{r['卡玛']:5.2f} 笔{r['笔数']:4d} "
              f"胜{r['胜率%']:5.1f}% 单笔{r['单笔%']:+.2f}%", flush=True)
    p = ROOT / f"output/exp_web_excl_{days}.json"
    p.write_text(json.dumps({"windows": {str(days): w}}, ensure_ascii=False, indent=1,
                            default=float), encoding="utf-8")
    print(f"写出 {p}", flush=True)


if __name__ == "__main__":
    main()
