# -*- coding: utf-8 -*-
"""第182轮·B: 执行链已到极限, 回头扫唯一没动过的层 —— 选股(新口径重扫)

为什么不是重复第 171 轮
------------------------
tools/exp_sat_pool.py(第 171 轮)扫过池子, 但那是 **4 只并发 + 无限价 + 无大盘闸门** 的旧口径。
此后四轮把执行层全换了: 单吊(max_pos=1)、信号日收盘 -1% 限价 3 日有效、持仓层闸门、
入场缓冲 MA20+0.3%。旧结论不能直接搬, 因为"池子多宽"与"买入价低 1%"是相互作用的 ——
池子越窄, 选出来的票越同质, 限价越不容易成交; 池子越宽, 个体差异越大, 限价成交后的
质量差异也越大。

本轮已否掉的执行层方向(见 output/exp_sat_exitlim.json)
-------------------------------------------------------
持满挂"收盘×(1+X%)"止盈单: 1500 日 base 42.2/-21.6/1.95 -> X=0.5% 30.2/-39.4,
1.0% 32.5/-32.8, 1.5% 33.0/-27.9, 2.0% 36.5/-24.7; 连"次日开盘卖"都只有 39.5/-26.1。
hold_days 7/12/15 = 11.0/38.0/25.6(1500 日)。执行链见顶。

本轮变体(其余参数一字不动, 同敞口)
----------------------------------
池宽: ivol_pct_max 0.05/0.10/0.15/0.20/0.30
排序: ivol / mom20_desc / dist_high20(均在 pct=0.10 下, 与 base 只差排序键)
二次过滤: dist_high20_max=-5(剔除距 20 日高点不足 5% 的票)
宽度门槛: min_cands=2(当日只有 1 只候选不买)

产物: output/exp_sat_pool2.json
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


def run_one(ctx, cfg, entry, sat=None):
    c = copy.deepcopy(cfg)
    if sat:
        c["satellite"].update(sat)
    res = sb.run(dict(ctx), c, entry_ok=entry)
    m = res["metrics"]
    return {"总收益%": m["总收益率%"], "年化%": m["年化收益率%"], "回撤%": m["最大回撤%"],
            "卡玛": m["卡玛比率"], "笔数": m["交易次数"], "胜率%": m["胜率%"],
            "单笔%": m["平均单笔收益%"], "空仓天": m["空仓天数"], "年度": yearly(res)}


def main():
    cfg = load_config()
    out = {"windows": {}}
    plan = {
        1500: [
            ("base", {}),
            ("pct05", {"ivol_pct_max": 0.05}),
            ("pct15", {"ivol_pct_max": 0.15}),
            ("pct20", {"ivol_pct_max": 0.20}),
            ("pct30", {"ivol_pct_max": 0.30}),
            ("pct50", {"ivol_pct_max": 0.50}),
            ("sort_ivol", {"sort_by": "ivol"}),
            ("sort_mom20", {"sort_by": "mom20_desc"}),
            ("sort_dh20", {"sort_by": "dist_high20"}),
            ("dh_m5", {"dist_high20_max": -5.0}),
            ("cands2", {"min_cands": 2}),
        ],
        3053: [
            ("base", {}),
            ("pct20", {"ivol_pct_max": 0.20}),
            ("pct30", {"ivol_pct_max": 0.30}),
            ("sort_ivol", {"sort_by": "ivol"}),
            ("sort_mom20", {"sort_by": "mom20_desc"}),
            ("dh_m5", {"dist_high20_max": -5.0}),
        ],
    }
    for days, items in plan.items():
        print(f"\n===== 窗口 {days} 日 =====", flush=True)
        ctx = prepare_context(cfg, days)
        entry = entry_ok_series(ctx.get("reg"), cfg)
        print(f"实际 {len(ctx['main_dates'])} 日", flush=True)
        w = {"days": len(ctx["main_dates"]), "start": ctx["main_dates"][0],
             "end": ctx["main_dates"][-1], "variants": {}}
        for nm, sat in items:
            try:
                r = run_one(ctx, cfg, entry, sat)
            except Exception as ex:
                print(f"{nm:12s} ERROR {type(ex).__name__}: {ex}", flush=True)
                w["variants"][nm] = {"error": str(ex)}
                continue
            w["variants"][nm] = r
            print(f"{nm:12s} 总{r['总收益%']:+9.1f}% 年化{r['年化%']:+6.1f}% "
                  f"回撤{r['回撤%']:7.1f}% 卡玛{r['卡玛']:5.2f} 笔{r['笔数']:4d} "
                  f"胜{r['胜率%']:5.1f}% 单笔{r['单笔%']:+.2f}%", flush=True)
        out["windows"][str(days)] = w

    p = ROOT / "output/exp_sat_pool2.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"\n写出 {p}", flush=True)


if __name__ == "__main__":
    main()
