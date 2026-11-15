"""第二百三十四轮: 单因子独立回测 —— 每个特征单独值多少收益

现有策略是 32 维特征的加权组合(且权重是 200+ 轮人肉试出来的)。本轮反过来:
把每个特征单独拿出来, 按它排序买第 1 名, 看它自己能赚多少。

这样能回答一个此前没回答过的问题: 现有那套复杂加权, 比"单看一个特征"强多少?
  若某单因子能到 60%, 说明组合的复杂度大半是冗余;
  若最好的单因子只有 25%, 说明信息确实在组合里。

口径完全对齐引擎: T日收盘定信号 -> T+1开盘买(仅低开成交) -> T+2收盘卖,
持仓互斥(T+1/T+2 不再开仓), 风控=等权指数>MA20, 成本按单边计。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "output" / "feat234"

FEATS = ["pct", "amp", "close_pos", "low_tail", "up_shadow", "gap", "intraday",
         "vr", "turnover_rate", "amt_yi", "amt_ratio", "circ_mv",
         "bias5", "bias10", "bias20", "bias60",
         "dist_high60", "dist_low60", "dist_high20",
         "ret3", "ret5", "ret10", "ret20", "vol20", "maxdd20",
         "dn_streak", "up_streak", "rel_pct", "rs20",
         "breadth", "idx_dist", "listed"]


def load_all(scope: str = "lowopen") -> pd.DataFrame:
    paths = sorted(FEAT.glob("f*.parquet"))
    cols = ["trade_date", "is_paused", "is_st", "y", "n1_ok", "pct", "listed"] + FEATS
    cols = list(dict.fromkeys(cols))
    parts = []
    for p in paths:
        d = pd.read_parquet(p, columns=cols)
        d = d[(d["is_paused"] == 0) & (d["is_st"] == 0)]
        d = d[d["pct"].between(-11.5, 11.5)]
        d = d[d["y"].between(-40, 60)]
        d = d[d["listed"] >= 250]
        if scope == "lowopen":
            d = d[d["n1_ok"] == True]      # noqa: E712
        if len(d):
            parts.append(d)
    return pd.concat(parts, ignore_index=True)


def run_factor(d: pd.DataFrame, dates: list[int], day_pos: dict[int, int],
               feat: str, ascending: bool, cost_pct: float,
               regime: pd.Series, min_pool: int = 1) -> dict:
    """单因子: 每天按 feat 排序买第 1 名, 持仓互斥。"""
    dd = d[["trade_date", feat, "y"]].dropna()
    dd = dd[np.isfinite(dd[feat])]
    if dd.empty:
        return {}
    sgn = 1.0 if ascending else -1.0
    dd = dd.assign(k=dd[feat] * sgn)
    # 每日最优一只 (k 最小)
    dd = dd.sort_values(["trade_date", "k"], kind="mergesort")
    first = dd.groupby("trade_date", as_index=False).first()
    first = first.sort_values("trade_date")

    busy_until = -1
    rets, used = [], []
    for row in first.itertuples(index=False):
        td = int(row.trade_date)
        i = day_pos.get(td)
        if i is None or i <= busy_until:
            continue
        if regime is not None and not bool(regime.get(td, False)):
            continue
        r = float(row.y) - cost_pct
        rets.append(r)
        used.append(td)
        busy_until = i + 1          # T+1 买入 -> T+2 卖出, 占用 i+1, i+2
    if not rets:
        return {}
    r = np.array(rets)
    nav = np.cumprod(1 + r / 100.0)
    yrs = max(0.5, len(dates) / 244.0)
    ann = (nav[-1] ** (1 / yrs) - 1) * 100
    peak = np.maximum.accumulate(nav)
    dd_pct = float((nav / peak - 1).min() * 100)
    return {"年化": round(ann, 2), "回撤": round(dd_pct, 2),
            "卡玛": round(ann / abs(dd_pct), 2) if dd_pct else None,
            "笔数": len(r), "单笔": round(float(r.mean()), 3),
            "胜率": round(float((r > 0).mean()) * 100, 1),
            "倍数": round(float(nav[-1]), 2)}


def main() -> None:
    cost = float(sys.argv[1]) if len(sys.argv) > 1 else 0.11
    scope = sys.argv[2] if len(sys.argv) > 2 else "lowopen"
    mkt = pd.read_csv(FEAT / "market.csv")
    dates = mkt["trade_date"].tolist()
    day_pos = {int(t): i for i, t in enumerate(dates)}
    regime = pd.Series(mkt["idx_ma20_dist"].values > 0, index=mkt["trade_date"].values)

    print(f"加载 {scope} ...", flush=True)
    d = load_all(scope)
    print(f"{len(d)} 行, {d['trade_date'].nunique()} 日\n", flush=True)

    rows = []
    for f in FEATS:
        for asc in (True, False):
            r = run_factor(d, dates, day_pos, f, asc, cost, regime)
            if not r:
                continue
            r.update({"特征": f, "方向": "小优" if asc else "大优"})
            rows.append(r)
            print(f"{f:14s} {'小优' if asc else '大优'} 年化 {r['年化']:7.2f}%  "
                  f"回撤 {r['回撤']:7.2f}%  卡玛 {r['卡玛']:5.2f}  "
                  f"笔数 {r['笔数']:4d}  单笔 {r['单笔']:+.3f}%", flush=True)

    out = pd.DataFrame(rows).sort_values("年化", ascending=False)
    out.to_csv(ROOT / "output" / f"w234_single_{scope}.csv", index=False, encoding="utf-8-sig")
    print("\n==== 单因子 Top 12 ====")
    print(out.head(12).to_string(index=False), flush=True)
    print("\n==== 单因子 Bottom 5 ====")
    print(out.tail(5).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
