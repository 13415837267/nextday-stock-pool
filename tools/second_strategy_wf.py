# -*- coding: utf-8 -*-
"""第二策略走前验证(第七十一轮, 零参数变更): "缺口优先排序"能否作为主策略的空档补充?

背景: 用户要求"不改变5只"前提下提高交易次数/收益。主策略 top5 口径十年仅 162 笔,
      信号供给(深低开事件)是结构性天花板。本探针检验唯一同源互补候选——
      同一 dip 池、同一 -2.35 承接线, 只把"排序键"从评分换成缺口(缺口越深越先买),
      且仅在主策略当日无买入时运行(资金不重叠)。

判据(沿用历轮走前纪律): 双段同向 + era前不劣化(>=主策略) + era后改善。
结论: 全带宽版本 era前 -0.545%/笔(主策略 -0.193%/笔), 联合十年 -74.7%/回撤 -75.3%,
      **四项判据全部不通过 -> 第二策略证伪**, 不接入。

复现: python tools/second_strategy_wf.py  (落盘 output/second_strategy_wf.json)
"""
from __future__ import annotations

import json
import time

import numpy as np

from src.config import load_config
from src.selector import select_stocks
from src.backtest import prepare_context

COMM, STAMP, SLIP = 0.00025, 0.0005, 0.001
ERA = "20240901"


def _day_frac(reg, cfg) -> dict[str, float]:
    """复刻 run_backtest 的参与系数: 风控触发日 0, 超限后按 rf_ramp 渐近恢复。"""
    rm = cfg.get("regime") or {}
    max_block = int(rm.get("max_block_days", 0) or 0)
    reenter = float(rm.get("block_reenter_frac", 0.0) or 0.0)
    ramp = bool(rm.get("rf_ramp", False))
    out, consec = {}, 0
    for d in reg.index:
        if bool(reg.loc[d, "trade_ok"]):
            consec, out[d] = 0, 1.0
        elif "panic" in reg.columns and bool(reg.loc[d, "panic"]):
            out[d] = 0.0
        else:
            consec += 1
            out[d] = 0.0 if consec <= max_block else (
                min(1.0, reenter + 0.05 * (consec - max_block)) if ramp else reenter)
    return out


def build_pool(cfg, main_dates, by_date, names) -> dict[str, list[tuple]]:
    """逐日 dip 池: [(code, close_adj, bias20)], 保持评分降序。"""
    pool = {}
    for d in main_dates[:-2]:
        picks, _ = select_stocks(d, by_date[d], cfg, names)
        s = by_date[d]
        mp = dict(zip(s["ts_code"], zip(s["close_adj"], s["bias20"])))
        pool[d] = [(p["code"], float(mp[p["code"]][0]), float(mp[p["code"]][1]))
                   for p in picks if p["code"] in mp]
    return pool


def simulate(dates, pool, reg, per, day_frac, floor_main, floor_block,
             rank="gap", topk=5, gap_min=-7.0, band=1.0, skip=frozenset()):
    eq, trades = 1.0, []
    for i, d in enumerate(dates[:-2]):
        if d in reg.index and not bool(reg.loc[d, "trade_ok"]) and day_frac.get(d, 1.0) <= 0:
            continue
        cands = pool.get(d, [])
        d1, d2 = dates[i + 1], dates[i + 2]
        if not cands or d1 in skip:
            continue
        floor = floor_main
        if d1 in reg.index and not bool(reg.loc[d1, "trade_ok"]):
            if day_frac.get(d1, 0.0) <= 0:
                continue
            floor = floor_block
        if rank == "score":
            sel = cands[:topk]
        else:
            sel = cands[:max(1, int(np.ceil(len(cands) * band)))]
        ready = []
        for code, ca, b in sel[:500]:
            v = per.get(code)
            if v is None or d1 not in v.index or d2 not in v.index:
                continue
            r1 = v.loc[d1]
            if r1["is_paused"] == 1 or float(r1["vol"]) <= 0:
                continue
            if float(r1["open"]) >= float(r1["high_limit"]) - 1e-9:
                continue
            gap = (float(r1["open_adj"]) / ca - 1) * 100 if ca > 0 else 0.0
            if gap >= 9.0 or gap > floor or (gap_min is not None and gap < gap_min):
                continue
            ready.append((gap, code, r1))
        if not ready:
            continue
        if rank == "gap":
            ready.sort(key=lambda x: x[0])   # 缺口优先: 缺口最深的先买
        gap, code, r1 = ready[0]             # 评分序: 保持顺延(与主引擎一致)
        r2 = per.get(code).loc[d2]
        if r2["is_paused"] == 1 or float(r2["vol"]) <= 0:
            continue
        buy = float(r1["open_adj"]) * (1 + SLIP)
        spx = float(r2["open_adj"])
        if float(r2["open"]) <= float(r2["low_limit"]) + 1e-9:
            spx = float(r2["close_adj"])
        r = spx * (1 - SLIP) * (1 - STAMP) * (1 - COMM) / (buy * (1 + COMM)) - 1
        eq *= (1 + r)
        trades.append({"buy": d1, "sell": d2, "code": code, "ret": r * 100})
    return eq, trades


def summ(trades, eq) -> dict:
    if not trades:
        return dict(n=0, avg=0.0, win=0.0, comp=0.0, mdd=0.0)
    r = np.array([t["ret"] for t in trades])
    nav = np.cumprod(1 + r / 100)
    return dict(n=len(r), avg=round(float(r.mean()), 3),
                win=round(float((r > 0).mean() * 100), 1),
                comp=round(float((eq - 1) * 100), 1),
                mdd=round(float((nav / np.maximum.accumulate(nav) - 1).min() * 100), 1))


def split(trades) -> dict:
    out = {}
    for tag, lo, hi in (("pre", "", ERA), ("post", ERA, "9")):
        t = [x for x in trades if lo <= x["buy"] < hi]
        eq = 1.0
        for x in t:
            eq *= (1 + x["ret"] / 100)
        out[tag] = summ(t, eq)
    return out


def main() -> None:
    t0 = time.time()
    cfg = load_config()
    ctx = prepare_context(cfg, days=2500)
    dates, reg, per = ctx["main_dates"], ctx["reg"], ctx["per_code"]
    print(f"context {time.time() - t0:.0f}s {dates[0]}~{dates[-1]}", flush=True)
    frac = _day_frac(reg, cfg)
    fl_main = float(cfg["execution"]["buy_gap_below"])
    fl_block = float(cfg["execution"].get("block_gap_below", fl_main))
    pool = build_pool(cfg, dates, ctx["by_date"], ctx["names"])
    print(f"pool {time.time() - t0:.0f}s", flush=True)

    out = {}
    e, mt = simulate(dates, pool, reg, per, frac, fl_main, fl_block, rank="score", gap_min=None)
    out["main"] = {"all": summ(mt, e), "split": split(mt)}
    print("MAIN", out["main"], flush=True)
    for band in (1.0, 0.5):
        e, st = simulate(dates, pool, reg, per, frac, fl_main, fl_block, band=band)
        out[f"sec_band{band}"] = {"all": summ(st, e), "split": split(st)}
        print(f"SEC band={band}", out[f"sec_band{band}"], flush=True)
        allt = sorted(mt + st, key=lambda x: x["buy"])
        eq = 1.0
        for x in allt:
            eq *= (1 + x["ret"] / 100)
        out[f"joint_band{band}"] = {"all": summ(allt, eq), "split": split(allt)}
        print(f"JOINT band={band}", out[f"joint_band{band}"], flush=True)
    with open("output/second_strategy_wf.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"elapsed {time.time() - t0:.0f}s -> output/second_strategy_wf.json")


if __name__ == "__main__":
    main()
