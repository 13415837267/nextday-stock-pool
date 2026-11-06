# -*- coding: utf-8 -*-
"""第一百六十四轮: 主副策略资金叠加的权重网格(可复现)。

回答一个问题: 主策略 104.04% / 副策略 19.16%, 钱放一起真实复利是多少?

用法:
    python tools/exp_combo.py            # 权重网格 + 资金复用上限
    python tools/exp_combo.py --boot     # 追加 90/10 的配对 block bootstrap

依赖: output/satellite_bt.json 已由 tools/satellite_backtest.py 生成。
主策略用 BT_FULL_TRADES 取全量成交(只取每日净值, 不重建持仓时无需成交)。
"""
import os
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("BT_FULL_TRADES", "1")

import numpy as np

from src.backtest import prepare_context, run_backtest
from src.config import load_config


def perf(r: np.ndarray) -> dict:
    nav = np.cumprod(1.0 + r)
    yrs = len(r) / 252.0
    cagr = nav[-1] ** (1.0 / yrs) - 1 if nav[-1] > 0 else -1.0
    mdd = float((nav / np.maximum.accumulate(nav) - 1).min())
    vol = float(r.std() * np.sqrt(252))
    return dict(cagr=cagr, mdd=mdd, calmar=cagr / abs(mdd) if mdd else float("nan"),
                sharpe=float(r.mean() * 252 / vol) if vol else float("nan"),
                vol=vol, total=float(nav[-1] - 1))


def load_main(cfg, days):
    """主策略逐日收益。"""
    ctx = prepare_context(cfg, days)
    r = run_backtest(cfg, days, ctx)
    return np.array([x["nav"] for x in r["equity"]], dtype=float), r["metrics"]


def load_sat(dates_main):
    """副策略逐日收益, 对齐到主策略的日期轴。"""
    import pandas as pd
    d = json.load(open(ROOT / "output" / "satellite_bt.json"))
    s = pd.Series({x["d"].replace("-", ""): x["nav"] for x in d["equity"]}, dtype=float)
    key = [x["d"].replace("-", "") for x in json.load(open(ROOT / "output" / "backtest.json"))["equity"]]
    s = s.reindex(key)
    if s.isna().any():
        raise SystemExit("副策略日期轴与主策略不一致, 无法对齐")
    return s.to_numpy(), d["metrics"]


def dr(nav):
    r = np.zeros(len(nav))
    r[1:] = nav[1:] / nav[:-1] - 1
    return r


def main():
    cfg = load_config()
    days = int((cfg.get("backtest") or {}).get("trading_days", 1500))

    nav_m, mm = load_main(cfg, days)
    nav_s, ms = load_sat(None)
    rm, rs = dr(nav_m), dr(nav_s)

    print(f"主策略 {mm['年化收益率%']:.2f}% / 回撤 {mm['最大回撤%']:.2f}% / {mm['交易次数']} 笔")
    print(f"副策略 {ms['年化收益率%']:.2f}% / 回撤 {ms['最大回撤%']:.2f}% / {ms['交易次数']} 笔")
    print(f"日收益相关 {np.corrcoef(rm, rs)[0, 1]:+.4f}\n")

    print("| 主策略权重 | 年化 | 回撤 | 卡玛 | 夏普 | 期末净值 |")
    print("|---|---|---|---|---|---|")
    rows = []
    for w in range(0, 101, 10):
        p = perf(w / 100 * rm + (1 - w / 100) * rs)
        rows.append((w, p))
        print(f"| {w}% | {p['cagr']*100:.2f}% | {p['mdd']*100:.2f}% | {p['calmar']:.2f} | "
              f"{p['sharpe']:.2f} | {1+p['total']:.2f}x |")

    c = [p["cagr"] for _, p in rows]
    print(f"\n年化严格单调递增: {all(c[i] < c[i+1] for i in range(len(c)-1))}")
    bs = max(rows, key=lambda x: x[1]["sharpe"])
    bc = max(rows, key=lambda x: x[1]["calmar"])
    print(f"夏普最优 {bs[0]}% ({bs[1]['sharpe']:.2f}) | 卡玛最优 {bc[0]}% ({bc[1]['calmar']:.2f}) "
          f"| 复利最优 100% ({rows[-1][1]['cagr']*100:.2f}%)")

    # 资金完全复用: 主策略持仓日 -> 主, 空仓日 -> 副
    hold_m = np.abs(rm) > 1e-12
    print(f"\n主策略持仓日(有净值变动) {int(hold_m.sum())} / {len(rm)} = {hold_m.mean()*100:.1f}%")
    for lab, r in [("资金完全复用", np.where(hold_m, rm, rs)),
                   ("复用但副策略半仓", np.where(hold_m, rm, 0.5 * rs))]:
        p = perf(r)
        print(f"  {lab}: 年化 {p['cagr']*100:.2f}%  回撤 {p['mdd']*100:.2f}%  卡玛 {p['calmar']:.2f}")

    if "--boot" in sys.argv:
        print("\n=== 90/10 vs 100/0 配对 block bootstrap (L=10, B=3000) ===")
        r100, r90 = rm, 0.9 * rm + 0.1 * rs
        rng = np.random.default_rng(20261107)
        n, L, B = len(rm), 10, 3000
        nb = int(np.ceil(n / L))
        dc, dm = [], []
        for _ in range(B):
            st = rng.integers(0, n - L, nb)
            idx = (st[:, None] + np.arange(L)[None, :]).ravel()[:n]
            a, b = perf(r100[idx]), perf(r90[idx])
            dc.append((b["cagr"] - a["cagr"]) * 100)
            dm.append((b["mdd"] - a["mdd"]) * 100)
        print(f"  年化差 中位 {np.median(dc):+.2f}pp  P(>0)={np.mean(np.array(dc)>0)*100:.1f}%")
        print(f"  回撤差 中位 {np.median(dm):+.2f}pp  P(更浅)={np.mean(np.array(dm)>0)*100:.1f}%")


if __name__ == "__main__":
    main()
