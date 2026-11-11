# -*- coding: utf-8 -*-
"""第199轮 · 环境切换的显著性: 配对块自助

问题: θ=(0.10, 0.40) 的切换组合比 100% 主策略高 6.2pp 年化, 这是真的还是六年样本的运气?

做法: 对"切换组合 - 100%主策略"的**逐日收益差**做块自助(block=20 日, 1000 次),
      每次重采样同一条日收益轴线(两条腿在同一段日子里), 重新算年化差。
      报 P(差<=0) 与 95% 区间。这保留了策略的持仓自相关(2 日持有、10 日持有),
      比按笔重采样更保守。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from exp_regime_combo_199b import nav, state  # noqa: E402


def ann(r: np.ndarray) -> float:
    return float(np.prod(1 + r) ** (252 / max(len(r), 1)) - 1)


def boot(diff: np.ndarray, block: int = 20, n: int = 1000, seed: int = 199) -> dict:
    rng = np.random.default_rng(seed)
    L = len(diff)
    nb = int(np.ceil(L / block))
    out = np.empty(n)
    for k in range(n):
        starts = rng.integers(0, L - block, nb)
        s = np.concatenate([diff[i:i + block] for i in starts])[:L]
        out[k] = ann(s)
    return {"均值pp": round(float(out.mean()) * 100, 2),
            "P(<=0)": round(float((out <= 0).mean()), 3),
            "95%区间pp": [round(float(np.percentile(out, 2.5)) * 100, 1),
                         round(float(np.percentile(out, 97.5)) * 100, 1)]}


def main():
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    nm = nav("backtest" if days == 1500 else "backtest_3000")
    ns = nav("satellite_bt" if days == 1500 else f"satellite_bt_{days}")
    idx = nm.index.intersection(ns.index)
    m = pd.read_csv(ROOT / "output" / "exp_mkt_199.csv", dtype={"d": str}).set_index("d")
    vp = m.vol_pct.reindex(idx).ffill().values
    rm = nm.reindex(idx).pct_change().fillna(0).values
    rs = ns.reindex(idx).pct_change().fillna(0).values

    res = {}
    for lo, hi in [(0.10, 0.40), (0.15, 0.50), (0.20, 0.50), (0.35, 0.45)]:
        s = state(vp, lo, hi)
        r = np.where(s > 0.5, rm, rs).copy()
        r[np.abs(np.diff(s, prepend=s[0])) > 0.5] -= 0.003
        diff = r - rm
        res[f"{lo}-{hi}"] = {
            "切换年化%": round(ann(r) * 100, 1),
            "基线年化%": round(ann(rm) * 100, 1),
            "差pp": round((ann(r) - ann(rm)) * 100, 1),
            **boot(diff),
        }
        print(f"θ=({lo},{hi}) {res[f'{lo}-{hi}']}")
    (ROOT / "output" / f"exp_regime_boot_199_{days}.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
