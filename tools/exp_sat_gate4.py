# -*- coding: utf-8 -*-
"""第181轮·C: "入场连续2日确认"是真机制还是单点尖峰? 顺带压回撤

上一步结果(exp_sat_gate3)
--------------------------
入场要求 gate 连续 2 日通过, 清仓一字不动:
  1500 日 base 28.1/-24.2/1.16 -> entry2 42.8/-27.9/1.54 (99 笔)
  全史    base 18.6/-42.6/0.44 -> entry2 31.5/-48.9/0.64 (208 笔)
清仓笔均值从 -0.65% 转正到 +0.16%, 说明确实过滤掉了"买在闸门临界点"的坏交易。
但 entry3(连续3日) 全史掉到 13.6/-63.0/0.22 —— 1/2/3 是倒 V, 顶点单薄, 有过拟合嫌疑。

本轮两件事
----------
A. 换两种"去抖"口径, 若同样有效则机制成立(而不是"天数=2"这个魔法数):
   buf03 / buf05 : 指数需站上 MA20*(1+0.3%/0.5%) 才允许开新仓(连续量, 不用天数)
B. 压回撤: entry2 叠加更紧的 trail(6%) / stop(-10%), 看能否把 1500 日的 -27.9%
   拉回 base 的 -24.2% 而不吐掉收益增量。

口径: 同 prepare_context、同 run(); 入场/清仓分离用 run(entry_ok=...)。
产物: output/exp_sat_gate4.json
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context  # noqa: E402
from src.config import load_config  # noqa: E402
import tools.satellite_backtest as sb  # noqa: E402


def confirm(s: pd.Series, n: int) -> pd.Series:
    return s.astype(int).rolling(n).min().fillna(0).astype(bool)


def gate_buf(reg: pd.DataFrame, buf_pct: float, breadth_min: float = 0.25) -> pd.Series:
    """指数需站上 MA20*(1+buf%) 且宽度达标、非恐慌 —— 连续量口径的入场去抖。"""
    idx = reg["mkt_idx"]
    ma = idx.rolling(20).mean()
    return (idx >= ma * (1 + buf_pct / 100.0)) & (reg["breadth"] >= breadth_min) & (~reg["panic"])


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


def run_one(ctx, cfg, entry=None, sat=None):
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
    bmin = float(cfg["regime"].get("breadth_min", 0.25))
    out = {"config": {"breadth_min": bmin}, "windows": {}}
    plan = {
        1500: [("base", {}), ("entry2", {"entry": "c2"}), ("buf03", {"entry": "b0.3"}),
               ("buf05", {"entry": "b0.5"}), ("entry2_trail6", {"entry": "c2", "sat": {"trail_pct": 6}}),
               ("entry2_stop10", {"entry": "c2", "sat": {"stop_pct": -10}})],
        3053: [("base", {}), ("entry2", {"entry": "c2"}), ("buf03", {"entry": "b0.3"}),
               ("buf05", {"entry": "b0.5"}), ("entry2_trail6", {"entry": "c2", "sat": {"trail_pct": 6}})],
    }
    for days, items in plan.items():
        print(f"\n===== 窗口 {days} 日 =====", flush=True)
        ctx = prepare_context(cfg, days)
        ok = ctx["reg"]["trade_ok"]
        print(f"实际 {len(ctx['main_dates'])} 日 {ctx['main_dates'][0]} ~ {ctx['main_dates'][-1]}"
              f" / 闸门开 {ok.mean() * 100:.1f}%", flush=True)
        w = {"days": len(ctx["main_dates"]), "start": ctx["main_dates"][0],
             "end": ctx["main_dates"][-1], "gate_on_pct": round(float(ok.mean()) * 100, 1),
             "variants": {}}
        for nm, kw in items:
            try:
                e = None
                if kw.get("entry") == "c2":
                    e = confirm(ok, 2)
                elif kw.get("entry", "").startswith("b"):
                    e = gate_buf(ctx["reg"], float(kw["entry"][1:]), bmin)
                r = run_one(ctx, cfg, entry=e, sat=kw.get("sat"))
            except Exception as ex:
                print(f"{nm:14s} ERROR {ex}", flush=True)
                w["variants"][nm] = {"error": str(ex)}
                continue
            w["variants"][nm] = r
            print(f"{nm:14s} 总{r['总收益%']:+9.1f}% 年化{r['年化%']:+6.1f}% "
                  f"回撤{r['回撤%']:7.1f}% 卡玛{r['卡玛']:5.2f} 笔{r['笔数']:4d} "
                  f"胜{r['胜率%']:5.1f}% 单笔{r['单笔%']:+.2f}% 空仓{r['空仓天']:4d}", flush=True)
        out["windows"][str(days)] = w

    p = ROOT / "output/exp_sat_gate4.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"\n写出 {p}", flush=True)


if __name__ == "__main__":
    main()
