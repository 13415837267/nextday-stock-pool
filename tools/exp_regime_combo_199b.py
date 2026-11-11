# -*- coding: utf-8 -*-
"""第199轮 · 环境切换(带滞后) —— 降低切换频率

上一版(tools/exp_regime_combo_199.py)用单一阈值, 1500 日里权重要翻转 52~66 次,
0.3% 一次的成本吃掉 5~7pp 年化, 优势全被摩擦吃光。

这版加**滞后带**(hysteresis): 只有 vp 突破上轨才切到主策略、跌破下轨才切回副策略,
中间区间维持不动。代价是反应慢, 好处是切换次数少一个量级。

再补一个"只在极端环境才切"的变体(下轨很低/上轨很高), 让大部分时间留在主策略。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def nav(name: str) -> pd.Series:
    d = json.loads((ROOT / "output" / f"{name}.json").read_text(encoding="utf-8"))
    return pd.Series({x["d"].replace("-", ""): float(x["nav"]) for x in d["equity"]}).sort_index()


def perf(v: pd.Series) -> dict:
    v = v.dropna()
    r = v.pct_change().dropna()
    cagr = (v.iloc[-1] / v.iloc[0]) ** (252 / max(len(r), 1)) - 1
    dd = float((v / v.cummax() - 1).min())
    return {"倍数": round(float(v.iloc[-1] / v.iloc[0]), 2), "年化%": round(float(cagr) * 100, 1),
            "回撤%": round(dd * 100, 1), "卡玛": round(float(cagr / abs(dd)), 2) if dd else None}


def state(vp: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """vp>=hi → 1(主策略); vp<=lo → 0(副策略); 中间保持上一状态。初始=1。"""
    s = np.ones(len(vp))
    cur = 1.0
    for i, x in enumerate(vp):
        if np.isnan(x):
            s[i] = cur
            continue
        if x >= hi:
            cur = 1.0
        elif x <= lo:
            cur = 0.0
        s[i] = cur
    return s


def main():
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    nm = nav("backtest" if days == 1500 else "backtest_3000")
    ns = nav("satellite_bt" if days == 1500 else f"satellite_bt_{days}")
    idx = nm.index.intersection(ns.index)
    m = pd.read_csv(ROOT / "output" / "exp_mkt_199.csv", dtype={"d": str}).set_index("d")
    vp = m.vol_pct.reindex(idx).ffill().values
    rm = nm.reindex(idx).pct_change().fillna(0).values
    rs = ns.reindex(idx).pct_change().fillna(0).values

    rows = []
    for lo, hi in [(0.30, 0.35), (0.35, 0.45), (0.40, 0.50), (0.30, 0.50), (0.25, 0.60),
                   (0.20, 0.70), (0.15, 0.50), (0.10, 0.40), (0.20, 0.50), (0.25, 0.45)]:
        s = state(vp, lo, hi)
        flips = int((np.abs(np.diff(s, prepend=s[0])) > 0.5).sum())
        for c in (0.0, 0.003):
            r = np.where(s > 0.5, rm, rs).copy()
            r[np.abs(np.diff(s, prepend=s[0])) > 0.5] -= c
            p = perf(pd.Series(np.concatenate([[1.0], np.cumprod(1 + r)])))
            rows.append({"下轨": lo, "上轨": hi, "成本": c, "主策略占比%": round(s.mean() * 100),
                         "切换次数": flips, **p})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_json(ROOT / "output" / f"exp_regime_combo_199b_{days}.json", orient="records",
               force_ascii=False, indent=1)


if __name__ == "__main__":
    main()
