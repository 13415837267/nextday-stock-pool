# -*- coding: utf-8 -*-
"""第181轮 · 副策略闸门去抖: 缓冲带 / 迟滞 / 只挡入场不清仓

为什么查这个
------------
第180轮把副策略改成单吊 + 大盘闸门(MA20 & 宽度 0.25)后, 1500 日 120 笔里有 **70 笔**
(58%)是"大盘转弱清仓", 平均收益 +0.01%、胜率 41.4%; 真正持满 10 日的 38 笔平均 +5.73%、
胜率 71.1%, 贡献 +217.8pp。也就是说:
  · 闸门确实压住了回撤(关掉: 1500 日 +10.15%/-46.43% -> 打开 +28.08%/-24.22%);
  · 但它同时把 58% 的持仓在成本线附近震出局, 平均持有天数从 10 降到 5.2。
单吊下没有第二只票接力, 每一次"假破位"都是一次完整的空仓 + 重新排队(限价 -1%)。

本轮只测"闸门本身的抖动", 不动选股/持有期/仓位:
  A. 缓冲带: 指数跌破 MA20*(1-b) 才清仓 (b=0.5/1/2%), 恢复仍要求站上 MA20
  B. 对称迟滞: 跌破 -1% 清仓, 站上 +1% 才恢复 (双阈值, 防来回打脸)
  C. 宽度阈值: 0.25 -> 0.20 / 0.30
  D. 只挡入场不清仓: 闸门关闭时不开新仓, 已有持仓交给 trail/stop/持满
  E. 完全关闭闸门(对照组, 应复现第180轮的 +10.15%/-46.43%)

口径: 与 tools/satellite_backtest.py 完全同一引擎、同一次 prepare_context, 只替换 reg.trade_ok;
无未来函数 —— 迟滞状态机逐日推进, 只用当日及之前的信息。

用法: python tools/exp_sat_gate2.py
产物: output/exp_sat_gate2.json
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


def gate_series(reg: pd.DataFrame, exit_buf: float = 0.0,
                reenter_buf: float = 0.0, breadth_min: float = 0.25) -> pd.Series:
    """逐日推进的迟滞闸门(只用当日及之前信息)。

    exit_buf   : 指数跌破 MA20*(1-exit_buf%) 才从"开"转"关"
    reenter_buf: 指数站上 MA20*(1+reenter_buf%) 才从"关"转"开"
    宽度条件与 panic 熔断保持原口径(单向: 宽度不足/恐慌 -> 关; 恢复需宽度达标)。
    """
    idx = reg["mkt_idx"].to_numpy(dtype=float)
    ma = reg["mkt_ma"].to_numpy(dtype=float)
    br = reg["breadth"].to_numpy(dtype=float)
    panic = (reg["panic"].to_numpy(dtype=bool) if "panic" in reg.columns
             else np.zeros(len(idx), dtype=bool))
    out = np.zeros(len(idx), dtype=bool)
    st = True
    for i in range(len(idx)):
        x, m, b = idx[i], ma[i], br[i]
        if not np.isfinite(m):
            st = True
        elif st:
            if x < m * (1 - exit_buf / 100.0) or (np.isfinite(b) and b < breadth_min):
                st = False
        else:
            if x > m * (1 + reenter_buf / 100.0) and (not np.isfinite(b) or b >= breadth_min):
                st = True
        out[i] = st and not panic[i]
    return pd.Series(out, index=reg.index)


def flips(s: pd.Series) -> int:
    v = s.to_numpy(dtype=bool)
    return int(np.sum(v[1:] != v[:-1]))


def yearly(res: dict) -> dict:
    out = {}
    for pt in res["equity"]:
        y = pt["d"][:4]
        out.setdefault(y, [pt["nav"], pt["nav"]])
        out[y][1] = pt["nav"]
    return {y: round((v[1] / v[0] - 1) * 100, 1) for y, v in out.items()}


def run_one(ctx: dict, cfg: dict, reg: pd.DataFrame | None = None,
            gate_off: bool = False, no_exit: bool = False):
    c = copy.deepcopy(cfg)
    ctx2 = dict(ctx)
    if gate_off:
        c["satellite"]["market_gate"] = False
        c["satellite"]["gate_exit"] = False
        r = ctx["reg"].copy()
        r["trade_ok"] = True
        r["panic"] = False
        ctx2["reg"] = r
    elif reg is not None:
        ctx2["reg"] = reg
    orig = sb.should_exit
    if no_exit:
        def _patched(held_days, ret_pct, cfg_, hwm=None, close=None, gate_ok=None):
            return orig(held_days, ret_pct, cfg_, hwm=hwm, close=close, gate_ok=True)
        sb.should_exit = _patched
    try:
        res = sb.run(ctx2, c)
    finally:
        sb.should_exit = orig
    m = res["metrics"]
    return {
        "总收益%": m["总收益率%"], "年化%": m["年化收益率%"], "回撤%": m["最大回撤%"],
        "卡玛": m["卡玛比率"], "笔数": m["交易次数"], "胜率%": m["胜率%"],
        "单笔%": m["平均单笔收益%"], "空仓天": m["空仓天数"],
        "年度": yearly(res),
        "退出分布": _exit_mix(res["trades"]),
    }


def _exit_mix(trades: list) -> dict:
    out = {}
    for t in trades:
        k = t["note"].split(" ")[0] if " " in t["note"] else t["note"]
        d = out.setdefault(k, {"n": 0, "sum": 0.0})
        d["n"] += 1
        d["sum"] += float(t["ret_pct"] or 0)
    return {k: {"n": v["n"], "avg": round(v["sum"] / v["n"], 2)} for k, v in out.items()}


VARIANTS = {
    "base":     dict(),
    "e0.5":     dict(exit_buf=0.5),
    "e1.0":     dict(exit_buf=1.0),
    "e2.0":     dict(exit_buf=2.0),
    "sym1":     dict(exit_buf=1.0, reenter_buf=1.0),
    "br20":     dict(breadth_min=0.20),
    "br30":     dict(breadth_min=0.30),
    "noexit":   dict(no_exit=True),
    "gateoff":  dict(gate_off=True),
}

WINDOWS = [(1500, ["base", "e0.5", "e1.0", "e2.0", "sym1", "br20", "br30", "noexit", "gateoff"]),
           (3053, ["base", "e1.0", "e2.0", "sym1", "noexit"])]


def main():
    cfg = load_config()
    out = {"config": {"breadth_min": cfg["regime"].get("breadth_min"),
                      "market_ma": cfg["regime"].get("market_ma"),
                      "max_pos": cfg["satellite"].get("max_pos"),
                      "gate_exit": cfg["satellite"].get("gate_exit")},
           "windows": {}}
    for days, names in WINDOWS:
        print(f"\n===== 窗口 {days} 日 =====", flush=True)
        ctx = prepare_context(cfg, days)
        print(f"实际 {len(ctx['main_dates'])} 日 {ctx['main_dates'][0]} ~ {ctx['main_dates'][-1]}",
              flush=True)
        base_flip = flips(ctx["reg"]["trade_ok"])
        print(f"[gate] 基准翻转 {base_flip} 次 / 关闭 {int((~ctx['reg']['trade_ok']).sum())} 天",
              flush=True)
        w = {"days": len(ctx["main_dates"]), "start": ctx["main_dates"][0],
             "end": ctx["main_dates"][-1], "base_flips": base_flip, "variants": {}}
        for nm in names:
            kw = VARIANTS[nm]
            reg = ctx["reg"] if not kw else None
            if kw and "gate_off" not in kw and "no_exit" not in kw:
                reg = ctx["reg"].copy()
                reg["trade_ok"] = gate_series(
                    ctx["reg"], kw.get("exit_buf", 0.0),
                    kw.get("reenter_buf", 0.0), kw.get("breadth_min", 0.25))
            try:
                r = run_one(ctx, cfg, reg=reg, gate_off=kw.get("gate_off", False),
                            no_exit=kw.get("no_exit", False))
            except Exception as e:  # 单个变体失败不拖垮整轮
                print(f"{nm:10s} ERROR {e}", flush=True)
                w["variants"][nm] = {"error": str(e)}
                continue
            r["flips"] = flips(reg["trade_ok"]) if reg is not None else None
            w["variants"][nm] = r
            print(f"{nm:10s} 总{r['总收益%']:+9.1f}% 年化{r['年化%']:+6.1f}% "
                  f"回撤{r['回撤%']:7.1f}% 卡玛{r['卡玛']:5.2f} 笔{r['笔数']:4d} "
                  f"胜{r['胜率%']:5.1f}% 单笔{r['单笔%']:+.2f}% 空仓{r['空仓天']:4d} "
                  f"翻转{r.get('flips')}", flush=True)
        out["windows"][str(days)] = w

    p = ROOT / "output/exp_sat_gate2.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"\n写出 {p}", flush=True)


if __name__ == "__main__":
    main()
