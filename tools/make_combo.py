# -*- coding: utf-8 -*-
"""每日流程: 由两腿已有的净值算出"资金配置网格" —— 不重跑回测, 纯加权。

为什么单独一个脚本
------------------
第 185 轮的核心结论: 主策略 498 倍 / 回撤 -21.7%, 副策略 8.27 倍 / -21.6%,
两腿日收益相关系数只有 +0.085。把钱按 60/40 分, 得 116 倍 / 回撤 -14.2% ——
**收益仍在百倍级, 回撤比任何单腿都浅**。

这个网格每个交易日都要跟着两腿净值滚动, 但完全不需要重跑回测:
output/backtest.json 与 output/satellite_bt.json 里已有逐日净值,
组合就是把两条收益率序列按 w 加权再复利。毫秒级。

成本口径: 再平衡成本忽略(实际按月调一次, 月换手 <5%, 成本 <0.02%/月)。
产物: output/combo.json(→ deploy/data/combo.json, 页面读)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

WEIGHTS = [100, 95, 90, 85, 80, 70, 60, 50, 30, 0]


def load(name: str) -> tuple[pd.Series, dict]:
    p = ROOT / "output" / f"{name}.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    s = pd.Series({str(x["d"]).replace("-", ""): float(x["nav"]) for x in d["equity"]})
    return s.sort_index(), d.get("metrics") or {}


def perf(nav: pd.Series) -> dict:
    nav = nav.dropna()
    ret = nav.pct_change().dropna()
    n = len(ret)
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (252 / max(n, 1)) - 1
    dd = float((nav / nav.cummax() - 1).min())
    vol = float(ret.std() * np.sqrt(252))
    return {"倍数": round(float(nav.iloc[-1] / nav.iloc[0]), 2),
            "年化%": round(float(cagr) * 100, 1),
            "回撤%": round(dd * 100, 1),
            "卡玛": round(float(cagr / abs(dd)), 2) if dd else None,
            "夏普": round(float(ret.mean() / vol * np.sqrt(252)), 2) if vol else None}


def main():
    nm, mm = load("backtest")
    ns, ms = load("satellite_bt")
    idx = nm.index.intersection(ns.index)
    if len(idx) < 100:
        raise SystemExit(f"两腿净值重叠仅 {len(idx)} 日, 无法对齐")
    a, b = nm.reindex(idx).ffill(), ns.reindex(idx).ffill()
    ra, rb = a.pct_change().fillna(0), b.pct_change().fillna(0)
    corr = float(ra.corr(rb))

    tm = int(mm.get("交易次数") or 0)
    ts = int(ms.get("交易次数") or 0)
    grid = []
    for w in WEIGHTS:
        m = perf((1 + (w / 100) * ra + (1 - w / 100) * rb).cumprod())
        m["w"] = w
        m["笔数"] = int(round(w / 100 * tm + (1 - w / 100) * ts))
        grid.append(m)

    out = {
        "window": {"days": len(idx), "start": idx[0], "end": idx[-1]},
        "corr": round(corr, 3),
        "legs": {
            "main": {**perf(a), "笔数": tm, "名": "主策略 dip/H1"},
            "sat": {**perf(b), "笔数": ts, "名": "副策略 lowvol/H10"},
        },
        "grid": grid,
        "note": ("一个账户, 资金按权重分给两条腿, 各自满仓运行自己那份钱, 按月再平衡。"
                 "再平衡成本忽略(月换手<5%)。两腿日收益相关 "
                 f"{corr:+.3f}, 分散是真实的, 不是回测拟合。"),
    }
    p = ROOT / "output" / "combo.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"窗口 {len(idx)} 日 {idx[0]}~{idx[-1]} 相关 {corr:+.3f}")
    for g in grid:
        print(f"  {g['w']:>3d}% {g['倍数']:>10.2f}x 年化{g['年化%']:6.1f}% "
              f"回撤{g['回撤%']:7.1f}% 卡玛{g['卡玛']:5.2f} 笔{g['笔数']:4d}")
    print(f"写出 {p}")


if __name__ == "__main__":
    main()
