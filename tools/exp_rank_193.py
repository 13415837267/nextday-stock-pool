# -*- coding: utf-8 -*-
"""第 193 轮: 副策略"候选池第 1 名之后"到底还有没有 alpha —— 页面为什么只该显示 1 只。

背景
----
用户看到副策略页面候选池列了 4 行(第 1~4 名), 而实际只买第 1 名 —— 这与主策略
"候选池恒 1 只(= 表内行数)"的页面口径不一致, 观感上像"列了 4 个备选却只买一个"。
在改页面之前必须先回答一个策略问题: **第 2~4 名是不是也有 alpha?**
  · 若第 2~4 名与第 1 名同质 -> 排序键在名次间是噪声, "单吊第 1 名"是运气,
    候选池就该展示多只(甚至可以分散);
  · 若第 2~4 名显著衰减 -> 第 1 名本身就是 alpha 所在, 候选池就该只展示 1 只。

三个部分
--------
A) 名次事件研究(无资金约束, 同口径):
   对每个信号日取低波池里按 mom20/ivol 排序的前 6 名, 统一"T+1 开盘买、第 10 个
   交易日收盘卖、扣 0.3% 往返成本", 统计每个名次的均值/中位/t/胜率/分年。
   t 值用 Newey-West(lag=10) —— 持有 10 日、每日重叠, 朴素 t 会高估。
   同时给出"减去当日前 6 名均值"的截面超额, 剥掉大盘 beta。

B) 引擎级: 候选池放宽 = 分散?
   max_pos / buys_per_day = (1,1) / (2,1) / (2,2) / (3,1) / (3,3), 每槽 1/max_pos 资金,
   其余规则(限价 −1%、持 10 日、止损 −12%、移动止损 8%、MA20 闸门)全部不动。
   注意这不是"多买几只票", 而是"同一笔钱摊到几个名次上"。

C) 反事实: 只买第 2 / 第 3 名(monkeypatch 掉第 1 名), 看排名是否真的单调。

用法: python3 tools/exp_rank_193.py [days] [--part a|b|c]
输出: output/exp_rank_193.json
"""
from __future__ import annotations

import copy
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context                      # noqa: E402
from src.config import load_config                            # noqa: E402
from src.satellite import (entry_ok_series, rank_pool,        # noqa: E402
                           rank_qualified, daily_params)

_spec = importlib.util.spec_from_file_location("satbt", ROOT / "tools" / "satellite_backtest.py")
satbt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(satbt)

TOPN = 6
COST = 0.003          # 往返成本, 与 engine cost_pct=0.3 同口径
HOLD = 10             # 与 satellite.hold_days 一致


def _nw_t(x: np.ndarray, lag: int = HOLD) -> float:
    """Newey-West(lag) t 值: 重叠样本修正。"""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 10:
        return float("nan")
    m = x.mean()
    e = x - m
    g0 = float((e * e).sum() / n)
    var = g0
    for L in range(1, min(lag, n - 1) + 1):
        gl = float((e[L:] * e[:-L]).sum() / n)
        var += 2.0 * (1.0 - L / (lag + 1.0)) * gl
    if var <= 0:
        return float("nan")
    return m / math.sqrt(var / n)


def part_a(ctx: dict, cfg: dict, days: int) -> dict:
    dates, by_date, per_code = ctx["main_dates"], ctx["by_date"], ctx["per_code"]
    P = daily_params(cfg)
    recs = []
    t0 = time.time()
    for i in range(len(dates) - HOLD - 2):
        d_in, d_out = dates[i + 1], dates[i + 1 + HOLD]
        day = by_date[dates[i]]
        rk = rank_pool(day, cfg)
        if len(rk) < P["min_pool"]:
            continue
        q = rank_qualified(day, cfg, rk)
        if not len(q):
            continue
        tops = q.head(TOPN)
        got = {}
        for r, (_, row) in enumerate(tops.iterrows(), 1):
            code = str(row["ts_code"])
            v = per_code.get(code)
            if v is None or d_in not in v.index or d_out not in v.index:
                continue
            try:
                o = float(v.loc[d_in]["open_adj"])
                c = float(v.loc[d_out]["close_adj"])
            except Exception:
                continue
            if not (o > 0 and c > 0):
                continue
            got[r] = (c / o - 1.0) - COST
        if 1 not in got:
            continue
        # 排序键断层: 第 1 名 / 第 2 名 的 mom20÷ivol 比值(第 1 名是"鹤立鸡群"还是"矮子里拔将军")
        _k = [(-float(x)) for x in tops["_key"].tolist()]   # _key = -mom20/ivol
        gap = (_k[0] / _k[1]) if len(_k) > 1 and _k[1] > 0 else None
        base = float(np.mean(list(got.values())))
        for r, v in got.items():
            recs.append({"date": dates[i], "rank": r, "ret": v, "exc": v - base, "gap": gap})
        if (i % 200) == 0:
            print(f"  A: {i}/{len(dates)} {dates[i]} {time.time() - t0:.0f}s", flush=True)

    df = pd.DataFrame(recs)
    rows = []
    for r in range(1, TOPN + 1):
        s = df[df["rank"] == r]
        if not len(s):
            continue
        yr = s.groupby(s["date"].str[:4])["ret"].mean() * 100
        rows.append({
            "rank": r, "n": int(len(s)),
            "mean": round(s["ret"].mean() * 100, 3),
            "median": round(s["ret"].median() * 100, 3),
            "nw_t": round(_nw_t(s["ret"].to_numpy()), 2),
            "win": round((s["ret"] > 0).mean() * 100, 1),
            "exc": round(s["exc"].mean() * 100, 3),
            "exc_t": round(_nw_t(s["exc"].to_numpy()), 2),
            "by_year": {k: round(v, 2) for k, v in yr.items()},
        })
    # 名次差: 第 1 名 − 第 k 名(同日配对)
    piv = df.pivot_table(index="date", columns="rank", values="ret")
    sprd = []
    for k in range(2, TOPN + 1):
        if k in piv.columns:
            d = (piv[1] - piv[k]).dropna()
            sprd.append({"pair": f"1-{k}", "n": int(len(d)),
                         "mean": round(d.mean() * 100, 3),
                         "nw_t": round(_nw_t(d.to_numpy()), 2),
                         "win": round((d > 0).mean() * 100, 1)})
    # 名次间相关性: 第 1 名与第 k 名是不是同一批票 / 同涨同跌
    print(f"  A 完成 {time.time() - t0:.0f}s, {len(df)} 条", flush=True)

    # 断层分组: 只有第 1 名能成交, 所以"第 1 名领先第 2 名多少"是唯一可用的当日信息。
    g = df[df["rank"] == 1].dropna(subset=["gap"]).copy()
    gap = {"n": int(len(g)), "median": round(float(g["gap"].median()), 2),
           "p10": round(float(g["gap"].quantile(0.10)), 2),
           "p90": round(float(g["gap"].quantile(0.90)), 2), "bins": [], "abs": []}
    if len(g) > 40:
        g["bin"] = pd.qcut(g["gap"], 4, labels=["Q1 最窄", "Q2", "Q3", "Q4 最宽"])
        for lab, s in g.groupby("bin", observed=True):
            gap["bins"].append({
                "bin": str(lab), "n": int(len(s)),
                "gap_med": round(float(s["gap"].median()), 2),
                "mean": round(float(s["ret"].mean()) * 100, 3),
                "nw_t": round(_nw_t(s["ret"].to_numpy()), 2),
                "win": round(float((s["ret"] > 0).mean()) * 100, 1),
                "by_year": {k: round(float(v) * 100, 2)
                            for k, v in s.groupby(s["date"].str[:4])["ret"].mean().items()},
            })
        for lab, lo, hi in (("<1.5 无明显断层", 0.0, 1.5), ("1.5~3", 1.5, 3.0), (">3 明显断层", 3.0, 1e9)):
            s = g[(g["gap"] >= lo) & (g["gap"] < hi)]
            if not len(s):
                continue
            gap["abs"].append({
                "bin": lab, "n": int(len(s)),
                "mean": round(float(s["ret"].mean()) * 100, 3),
                "nw_t": round(_nw_t(s["ret"].to_numpy()), 2),
                "win": round(float((s["ret"] > 0).mean()) * 100, 1),
                "by_year": {k: round(float(v) * 100, 2)
                            for k, v in s.groupby(s["date"].str[:4])["ret"].mean().items()},
            })
    return {"rows": rows, "spread": sprd, "n_days": int(df["date"].nunique()), "gap": gap}


def part_b(ctx: dict, cfg: dict, entry) -> list:
    rows = []
    grid = [(1, 1), (2, 1), (2, 2), (3, 1), (3, 3)]
    for mx, by in grid:
        t0 = time.time()
        c = copy.deepcopy(cfg)
        c["satellite"]["max_pos"] = mx
        c["satellite"]["buys_per_day"] = by
        c["satellite"]["pool_show"] = max(4, by)
        res = satbt.run(ctx, c, entry_ok=entry)
        m = res["metrics"]
        r = {"max_pos": mx, "buys": by, "total": m["总收益率%"], "ann": m["年化收益率%"],
             "mdd": m["最大回撤%"], "calmar": m["卡玛比率"], "n": int(m["交易次数"]),
             "avg": m["平均单笔收益%"], "win": m["胜率%"]}
        rows.append(r)
        print(f"  B: {mx}/{by} -> {r['total']:,.1f}% 年化 {r['ann']:.1f}% 回撤 {r['mdd']:.2f}% "
              f"卡玛 {r['calmar']:.2f} 笔 {r['n']} 单笔 {r['avg']:.3f}% ({time.time() - t0:.0f}s)", flush=True)
    return rows


def part_c(ctx: dict, cfg: dict, entry) -> list:
    orig = satbt.pick_list
    rows = []
    for off in (1, 2, 3):
        def patched(day, cfg_, held=(), k=1, _off=off):
            res = orig(day, cfg_, held=held, k=k + _off)
            return res[_off:]
        satbt.pick_list = patched
        t0 = time.time()
        c = copy.deepcopy(cfg)
        c["satellite"]["max_pos"] = 1
        c["satellite"]["buys_per_day"] = 1
        res = satbt.run(ctx, c, entry_ok=entry)
        m = res["metrics"]
        r = {"skip": off, "total": m["总收益率%"], "ann": m["年化收益率%"],
             "mdd": m["最大回撤%"], "calmar": m["卡玛比率"], "n": int(m["交易次数"]),
             "avg": m["平均单笔收益%"], "win": m["胜率%"]}
        rows.append(r)
        print(f"  C: 跳过前 {off} 名 -> {r['total']:,.1f}% 年化 {r['ann']:.1f}% "
              f"回撤 {r['mdd']:.2f}% 卡玛 {r['calmar']:.2f} 笔 {r['n']} "
              f"单笔 {r['avg']:.3f}% ({time.time() - t0:.0f}s)", flush=True)
    satbt.pick_list = orig
    return rows


def main() -> None:
    days = 1500
    part = "abc"
    for a in sys.argv[1:]:
        if a.isdigit():
            days = int(a)
        elif a.startswith("--part"):
            part = a.split("=", 1)[1] if "=" in a else "abc"
    cfg = load_config()
    ctx = prepare_context(cfg, days)
    dates = ctx["main_dates"]
    print(f"窗口 {len(dates)} 日 {dates[0]} ~ {dates[-1]}", flush=True)
    entry = entry_ok_series(ctx.get("reg"), cfg)
    out = {"days": len(dates), "window": [dates[0], dates[-1]], "topn": TOPN,
           "cost": COST, "hold": HOLD}
    if "a" in part:
        out["a"] = part_a(ctx, cfg, days)
    if "b" in part:
        out["b"] = part_b(ctx, cfg, entry)
    if "c" in part:
        out["c"] = part_c(ctx, cfg, entry)
    p = ROOT / "output" / f"exp_rank_193_{days}.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"写出 {p}", flush=True)


if __name__ == "__main__":
    main()
