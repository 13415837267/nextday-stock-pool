# -*- coding: utf-8 -*-
"""第199轮 · 环境切换组合: 高波做主策略, 低波做副策略

诊断结果(tools/exp_regime_199.py)在双窗口都单调:
  主策略单笔均值 按市场20日波动率分位 Q1→Q4:  0.92/0.82/1.86/2.20 (1500日)
                                              0.59/0.92/1.68/1.78 (3000日)
  → 低波环境里"深跌缩量反弹"几乎不赚钱, 高波里才是它的舒适区。
副策略(lowvol + mom20/ivol)在低波 36 笔均值 3.57%, 中波 0.72%, 高波 3.39%。

所以把两腿按"环境"而不是"固定权重"缝合: vp>=θ 全主策略, vp<θ 全副策略。
切换成本按单边 0.15% 双边 0.3% 计, 只在权重真的翻转的日子收。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SWITCH = 0.003          # 一次完整切换(卖旧腿+买新腿)的成本


def nav(name: str) -> pd.Series:
    d = json.loads((ROOT / "output" / f"{name}.json").read_text(encoding="utf-8"))
    s = pd.Series({x["d"].replace("-", ""): float(x["nav"]) for x in d["equity"]})
    return s.sort_index()


def perf(nav_s: pd.Series) -> dict:
    nav_s = nav_s.dropna()
    r = nav_s.pct_change().dropna()
    cagr = (nav_s.iloc[-1] / nav_s.iloc[0]) ** (252 / max(len(r), 1)) - 1
    dd = float((nav_s / nav_s.cummax() - 1).min())
    return {"倍数": round(float(nav_s.iloc[-1] / nav_s.iloc[0]), 2),
            "年化%": round(float(cagr) * 100, 1), "回撤%": round(dd * 100, 1),
            "卡玛": round(float(cagr / abs(dd)), 2) if dd else None}


def sim(rm: pd.Series, rs: pd.Series, w: np.ndarray, cost: float) -> pd.Series:
    """w[i]=1 表示当日资金全在主策略。权重翻转日扣一次切换成本。"""
    r = np.where(w > 0.5, rm.values, rs.values)
    flips = np.abs(np.diff(w, prepend=w[0])) > 0.5
    r = r.copy()
    r[flips] -= cost
    return pd.Series(np.cumprod(1.0 + r), index=rm.index)


def main():
    nm, ns = nav("backtest"), nav("satellite_bt")
    idx = nm.index.intersection(ns.index)
    m = pd.read_csv(ROOT / "output" / "exp_mkt_199.csv", dtype={"d": str}).set_index("d")
    vp = m.vol_pct.reindex(idx).ffill()
    xsd = m.xsd.reindex(idx).ffill()
    rm = nm.reindex(idx).pct_change().fillna(0)
    rs = ns.reindex(idx).pct_change().fillna(0)
    print(f"重叠 {len(idx)} 日 {idx[0]}~{idx[-1]}, 相关系数 {rm.corr(rs):.3f}")
    print("基线: 主", perf(nm.reindex(idx)), "副", perf(ns.reindex(idx)))

    rows = []
    for th in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        w = (vp >= th).astype(float).values
        w = np.where(np.isnan(w), 1.0, w)
        flips = int((np.abs(np.diff(w, prepend=w[0])) > 0.5).sum())
        for c in (0.0, SWITCH):
            nv = sim(rm, rs, w, c)
            p = perf(pd.concat([pd.Series([1.0]), nv.reset_index(drop=True)]))
            rows.append({"θ": th, "成本": c, "主策略占比%": round(float(w.mean()) * 100),
                         "切换次数": flips, **p})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_json(ROOT / "output" / "exp_regime_combo_199.json", orient="records",
               force_ascii=False, indent=1)

    # 同样口径的固定权重对照(月度调仓, 同 make_combo)
    def fixed(wm: float) -> dict:
        cap_m, cap_s = wm, 1 - wm
        prev, vals = idx[0][:6], [1.0]
        for i, d in enumerate(idx):
            if i and d[:6] != prev:
                tot = cap_m + cap_s
                cap_m, cap_s = tot * wm, tot * (1 - wm)
                prev = d[:6]
            cap_m *= 1 + rm.iloc[i]
            cap_s *= 1 + rs.iloc[i]
            vals.append(cap_m + cap_s)
        return perf(pd.Series(vals))
    print("\n固定权重(月度调仓):")
    for wm in (1.0, 0.8, 0.6):
        print(f"  {int(wm*100)}/{int((1-wm)*100)}", fixed(wm))


if __name__ == "__main__":
    main()
