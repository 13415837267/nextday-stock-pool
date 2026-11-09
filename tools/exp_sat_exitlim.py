# -*- coding: utf-8 -*-
"""第182轮: 执行链最后一环 —— 持满后的卖出价

现状(第181轮落地后, 1500 日)
----------------------------
106 笔里 35 笔"持满 10 日", 均值 +7.02%、胜率 74.3%, 合计贡献 +245.7%;
其余 71 笔合计 +28.4%。也就是说收益里 90% 来自"能走完 10 日"的那批票 ——
而它们在第 10 日往往还在创新高, 收盘一刀切等于把最后一段让掉。

本轮唯一的改动
--------------
持满当日不再收盘甩卖, 改挂"当日收盘 × (1+X%)"的止盈单, N 日内有效:
  · 次日开盘已过线 -> 开盘价成交(比挂单价更好);
  · 盘中触及 -> 挂单价成交;
  · 到期未触及 -> 当日收盘保底卖出(不会无限期拖下去)。
止损/移动止损/大盘闸门**不走这条路** —— 一旦触发仍然当日收盘就走。

对照组: X=0.001%(≈次日开盘必成交), 用来验证"持满即收盘卖"本身是不是已经最优。
顺带扫 hold_days 7/12/15, 看"持满 10 日"这个数在入场端变严之后是否还成立。

产物: output/exp_sat_exitlim.json
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
        k = t["note"].split(" ")[0]
        d = out.setdefault(k, {"n": 0, "sum": 0.0})
        d["n"] += 1
        d["sum"] += float(t["ret_pct"] or 0)
    return {k: {"n": v["n"], "avg": round(v["sum"] / v["n"], 2)} for k, v in out.items()}


def run_one(ctx, cfg, entry, sat=None):
    c = copy.deepcopy(cfg)
    if sat:
        c["satellite"].update(sat)
    res = sb.run(dict(ctx), c, entry_ok=entry)
    m = res["metrics"]
    return {"总收益%": m["总收益率%"], "年化%": m["年化收益率%"], "回撤%": m["最大回撤%"],
            "卡玛": m["卡玛比率"], "笔数": m["交易次数"], "胜率%": m["胜率%"],
            "单笔%": m["平均单笔收益%"], "空仓天": m["空仓天数"],
            "年度": yearly(res), "退出": exit_mix(res["trades"])}


def main():
    cfg = load_config()
    out = {"windows": {}}
    plan = {
        1500: [
            ("base", {}),
            ("lim05_3", {"exit_limit_pct": 0.5, "exit_limit_days": 3}),
            ("lim10_3", {"exit_limit_pct": 1.0, "exit_limit_days": 3}),
            ("lim15_3", {"exit_limit_pct": 1.5, "exit_limit_days": 3}),
            ("lim10_2", {"exit_limit_pct": 1.0, "exit_limit_days": 2}),
            ("lim20_3", {"exit_limit_pct": 2.0, "exit_limit_days": 3}),
            ("open_next", {"exit_limit_pct": 0.001, "exit_limit_days": 1}),
            ("hold7", {"hold_days": 7}),
            ("hold12", {"hold_days": 12}),
            ("hold15", {"hold_days": 15}),
        ],
        3053: [
            ("base", {}),
            ("lim10_3", {"exit_limit_pct": 1.0, "exit_limit_days": 3}),
            ("lim05_3", {"exit_limit_pct": 0.5, "exit_limit_days": 3}),
            ("lim15_3", {"exit_limit_pct": 1.5, "exit_limit_days": 3}),
            ("open_next", {"exit_limit_pct": 0.001, "exit_limit_days": 1}),
            ("hold12", {"hold_days": 12}),
        ],
    }
    for days, items in plan.items():
        print(f"\n===== 窗口 {days} 日 =====", flush=True)
        ctx = prepare_context(cfg, days)
        entry = entry_ok_series(ctx.get("reg"), cfg)
        print(f"实际 {len(ctx['main_dates'])} 日 {ctx['main_dates'][0]} ~ {ctx['main_dates'][-1]}",
              flush=True)
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
                  f"胜{r['胜率%']:5.1f}% 单笔{r['单笔%']:+.2f}% 空仓{r['空仓天']:4d}", flush=True)
        out["windows"][str(days)] = w

    p = ROOT / "output/exp_sat_exitlim.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"\n写出 {p}", flush=True)


if __name__ == "__main__":
    main()
