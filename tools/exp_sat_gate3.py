# -*- coding: utf-8 -*-
"""第181轮·B: 闸门该改的不是"清仓更晚", 而是"买在临界点"

上一实验(exp_sat_gate2)的结论
------------------------------
缓冲带 / 对称迟滞 8 个变体全部跑输, 且延迟越久越差:
  1500 日 base 28.1%/-24.2% -> e0.5 25.7/-28.5 -> e1.0 25.4/-27.2 -> e2.0 14.1/-43.6
  全史     base 18.6/-42.6 -> e1.0 15.2/-55.2 -> e2.0 10.8/-55.5
说明"清仓早"不是缺陷而是特性: 全史 135 笔闸门清仓平均 -0.65%, 延迟只会放大亏损。
另一发现: 宽度阈值 0.20/0.30 与 base 逐位相同 —— **宽度条件从未单独起过作用**,
MA20 才是唯一约束(指数跌破 MA20 的日子, 宽度必然已 < 0.25)。

所以真正的漏点在入场端
----------------------
1500 日 120 笔里 28 笔(23%)"买入当日即被闸门清仓"、44 笔(37%)持有 <=2 日, 均值 -0.81%。
信号日 gate_ok=True, T+1 开盘买入, T+1 收盘就破位 —— 这是**买在临界点**。

本实验只改"什么时候允许开新仓", 清仓判定一字不动:
  A. entry2 / entry3 : 要求 gate 连续 2/3 日通过(去掉刚翻多的第一天)
  B. ma10 / ma30 / ma60 : 换闸门均线(清仓+入场同步换, 检验 MA20 是否最优)
  C. entry2_ma10 : 组合(更快的均线 + 入场确认)

口径: 同一 prepare_context、同一 run(); 入场/清仓分离靠 run(entry_ok=...) 参数。

用法: python tools/exp_sat_gate3.py
产物: output/exp_sat_gate3.json
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context  # noqa: E402
from src.config import load_config  # noqa: E402
import tools.satellite_backtest as sb  # noqa: E402


def gate_ma(reg: pd.DataFrame, n: int, breadth_min: float = 0.25) -> pd.Series:
    """把 apply_regime 的 MA20 换成 MA(n), 其余(宽度/恐慌)保持原口径。"""
    idx = reg["mkt_idx"]
    ma = idx.rolling(n).mean()
    return (idx >= ma) & (reg["breadth"] >= breadth_min) & (~reg["panic"])


def confirm(s: pd.Series, n: int) -> pd.Series:
    """连续 n 日通过才为 True(前 n-1 日样本不足按不通过处理)。"""
    return s.astype(int).rolling(n).min().fillna(0).astype(bool)


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


def run_one(ctx, cfg, reg=None, entry=None):
    ctx2 = dict(ctx)
    if reg is not None:
        r = ctx["reg"].copy()
        r["trade_ok"] = reg
        ctx2["reg"] = r
    res = sb.run(ctx2, copy.deepcopy(cfg), entry_ok=entry)
    m = res["metrics"]
    return {"总收益%": m["总收益率%"], "年化%": m["年化收益率%"], "回撤%": m["最大回撤%"],
            "卡玛": m["卡玛比率"], "笔数": m["交易次数"], "胜率%": m["胜率%"],
            "单笔%": m["平均单笔收益%"], "空仓天": m["空仓天数"],
            "年度": yearly(res), "退出": exit_mix(res["trades"])}


def main():
    cfg = load_config()
    bmin = float(cfg["regime"].get("breadth_min", 0.25))
    out = {"config": {"breadth_min": bmin, "market_ma": cfg["regime"].get("market_ma")},
           "windows": {}}
    plan = {
        1500: ["base", "entry2", "entry3", "ma10", "ma30", "ma60", "entry2_ma10"],
        3053: ["base", "entry2", "entry3", "ma10", "ma30", "entry2_ma10"],
    }
    for days, names in plan.items():
        print(f"\n===== 窗口 {days} 日 =====", flush=True)
        ctx = prepare_context(cfg, days)
        base_ok = ctx["reg"]["trade_ok"]
        print(f"实际 {len(ctx['main_dates'])} 日 {ctx['main_dates'][0]} ~ {ctx['main_dates'][-1]}"
              f" / 闸门开 {int(base_ok.sum())} 天 ({base_ok.mean() * 100:.1f}%)", flush=True)
        w = {"days": len(ctx["main_dates"]), "start": ctx["main_dates"][0],
             "end": ctx["main_dates"][-1], "gate_on_pct": round(float(base_ok.mean()) * 100, 1),
             "variants": {}}
        for nm in names:
            try:
                if nm == "base":
                    r = run_one(ctx, cfg)
                elif nm.startswith("entry"):
                    n = int(nm[5])
                    e = confirm(base_ok, n)
                    r = run_one(ctx, cfg, entry=e)
                    r["确认后允许天数"] = int(e.sum())
                elif nm.startswith("ma"):
                    n = int(nm[2:])
                    g = gate_ma(ctx["reg"], n, bmin)
                    r = run_one(ctx, cfg, reg=g)
                elif nm == "entry2_ma10":
                    g = gate_ma(ctx["reg"], 10, bmin)
                    r = run_one(ctx, cfg, reg=g, entry=confirm(g, 2))
                else:
                    continue
            except Exception as e:  # 单变体失败不拖垮整轮
                print(f"{nm:12s} ERROR {e}", flush=True)
                w["variants"][nm] = {"error": str(e)}
                continue
            w["variants"][nm] = r
            print(f"{nm:12s} 总{r['总收益%']:+9.1f}% 年化{r['年化%']:+6.1f}% "
                  f"回撤{r['回撤%']:7.1f}% 卡玛{r['卡玛']:5.2f} 笔{r['笔数']:4d} "
                  f"胜{r['胜率%']:5.1f}% 单笔{r['单笔%']:+.2f}% 空仓{r['空仓天']:4d} "
                  f"清仓{r['退出'].get('大盘转弱清仓', {}).get('n', 0)}", flush=True)
        out["windows"][str(days)] = w

    p = ROOT / "output/exp_sat_gate3.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"\n写出 {p}", flush=True)


if __name__ == "__main__":
    main()
