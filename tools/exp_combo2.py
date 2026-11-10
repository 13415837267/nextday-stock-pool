# -*- coding: utf-8 -*-
"""第185轮·C: 组合口径重算 —— 两腿都换过口径了, 第164轮的"104%是上限"已过期

为什么重算
----------
第164轮那次组合叠加用的是旧口径: 主策略 104% / 副策略 19%, 结论"钱放一起
上限就是 104%, 分散没有意义"。但之后两年半的优化全变了:
  · 主策略: 缩量阴线回调 + H1 + lev1.25 -> 498 倍(不再是 104%)
  · 副策略: 低波池 + 回踩限价 + 双向闸门 + 单吊 -> 8.27 倍(不再是 19%)
两条腿的 log 收益差 3 个数量级, 权重网格的形状必然完全不同。

口径(写清楚, 免得又被当成"加杠杆")
  · 一个账户, 资金按 w 给主策略、(1-w) 给副策略, **每日按目标权重再平衡**
  · 两腿各自满仓运行自己那份钱, 不加任何融资
  · 再平衡成本忽略(实际按月调一次, 换手 <5%/月, 成本 <0.02%/月)
产物: output/exp_combo2_<days>.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context, run_backtest  # noqa: E402
from src.config import load_config  # noqa: E402
from src.satellite import entry_ok_series  # noqa: E402
import tools.satellite_backtest as sb  # noqa: E402


def to_series(equity: list) -> pd.Series:
    return pd.Series({str(p["d"]).replace("-", ""): float(p["nav"])
                      for p in equity}).sort_index()


def metrics(nav: pd.Series) -> dict:
    nav = nav.dropna()
    if len(nav) < 30:
        return {}
    ret = nav.pct_change().dropna()
    n = len(ret)
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (252 / max(n, 1)) - 1
    dd = float((nav / nav.cummax() - 1).min())
    vol = float(ret.std() * np.sqrt(252))
    return {"倍数": round(float(nav.iloc[-1] / nav.iloc[0]), 2),
            "总收益%": round(float(nav.iloc[-1] / nav.iloc[0] - 1) * 100, 1),
            "年化%": round(float(cagr) * 100, 1),
            "回撤%": round(dd * 100, 1),
            "卡玛": round(float(cagr / abs(dd)), 2) if dd else None,
            "夏普": round(float(ret.mean() / vol * np.sqrt(252)), 2) if vol else None}


def main():
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    cfg = load_config()
    ctx = prepare_context(cfg, days)
    print(f"===== {len(ctx['main_dates'])} 日 {ctx['main_dates'][0]} ~ "
          f"{ctx['main_dates'][-1]} =====", flush=True)

    r_main = run_backtest(cfg, days, ctx=ctx)
    nm, ns = to_series(r_main["equity"]), None
    print(f"主策略 {r_main['metrics']['总收益率%']:.0f}% 回撤 "
          f"{r_main['metrics']['最大回撤%']}% 笔 {r_main['metrics']['交易次数']}", flush=True)

    r_sat = sb.run(dict(ctx), cfg, entry_ok=entry_ok_series(ctx.get("reg"), cfg))
    ns = to_series(r_sat["equity"])
    print(f"副策略 {r_sat['metrics']['总收益率%']:.0f}% 回撤 "
          f"{r_sat['metrics']['最大回撤%']}% 笔 {r_sat['metrics']['交易次数']}", flush=True)

    idx = nm.index.intersection(ns.index)
    a, b = nm.reindex(idx).ffill(), ns.reindex(idx).ffill()
    ra, rb = a.pct_change().fillna(0), b.pct_change().fillna(0)
    corr = float(ra.corr(rb))
    print(f"两腿日收益相关 {corr:+.3f} | 重叠 {len(idx)} 日", flush=True)

    out = {"days": len(idx), "start": idx[0], "end": idx[-1], "corr": round(corr, 3),
           "main": {**metrics(a), "笔数": r_main["metrics"]["交易次数"]},
           "sat": {**metrics(b), "笔数": r_sat["metrics"]["交易次数"]},
           "combo": {}}
    print("\n| 主策略权重 | 倍数 | 年化 | 回撤 | 卡玛 | 夏普 | 笔数 |", flush=True)
    for w in (1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.3, 0.0):
        rc = w * ra + (1 - w) * rb
        m = metrics((1 + rc).cumprod())
        m["主策略权重"] = w
        m["笔数"] = int(round(w * r_main["metrics"]["交易次数"]
                              + (1 - w) * r_sat["metrics"]["交易次数"]))
        out["combo"][f"w{int(round(w*100))}"] = m
        print(f"| {int(round(w*100))}% | {m['倍数']:,.2f}x | {m['年化%']:.1f}% | "
              f"{m['回撤%']:.1f}% | {m['卡玛']:.2f} | {m['夏普']:.2f} | {m['笔数']} |", flush=True)

    p = ROOT / f"output/exp_combo2_{days}.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"写出 {p}", flush=True)


if __name__ == "__main__":
    main()
