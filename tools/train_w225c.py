"""第225轮C · 目标换成复利年化(右尾驱动), 其余同 225B

目标函数用 **t 值**（单笔均值 / 标准误）而不是卡玛：
  - 卡玛在满仓单票、回撤 90%+ 的离线序列里约等于"最大化年化"，会被少数极端值带跑；
  - t 值衡量"选股方向是否稳定有效"，对极端值不敏感，样本外更可能成立。
样本外规则：前半段(2010-2020)训练，后半段(2020-2026)验收，两段同向才进回测。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"

FACTORS = ["mom20", "neg_bias20", "neg_amount", "neg_vol_ratio",
           "low_tail", "neg_turnover_cool", "ivol", "neg_dist_low20"]


def stats(p: pd.DataFrame, w: np.ndarray, mask: np.ndarray | None = None) -> dict:
    X = p[FACTORS].values
    sc = X @ w
    td = p["trade_date"].values
    ret = p["ret"].values
    if mask is not None:
        sc, td, ret = sc[mask], td[mask], ret[mask]
    if len(sc) < 200:
        return {}
    order = np.lexsort((-sc, td))
    td_s, ret_s = td[order], ret[order]
    first = np.ones(len(td_s), dtype=bool)
    first[1:] = td_s[1:] != td_s[:-1]
    r = ret_s[first] / 100.0
    if len(r) < 30:
        return {}
    mu, sd = r.mean(), r.std(ddof=1)
    t = mu / (sd / np.sqrt(len(r))) if sd else 0.0
    nav = np.cumprod(1 + r)
    tail = float((r > 0.04).mean() * 100)   # 右尾占比: 单笔>4% 的交易日比例
    peak = np.maximum.accumulate(nav)
    mdd = float((nav / peak - 1).min())
    u = np.unique(td_s[first])
    yrs = max(0.5, (pd.Timestamp(str(u[-1])) - pd.Timestamp(str(u[0]))).days / 365.25)
    ann = float(nav[-1] ** (1 / yrs) - 1)
    return {"t": float(t), "ann": ann * 100, "mdd": mdd * 100, "tail": tail,
            "calmar": ann / abs(mdd) if mdd else np.nan,
            "n": int(len(r)), "avg": mu * 100, "win": float((r > 0).mean() * 100)}


def main() -> None:
    p = pd.read_parquet(OUT / "pool_225b.parquet")
    p = p.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
    dates = np.sort(p["trade_date"].unique())
    half = dates[int(len(dates) * 0.55)]
    m1 = p["trade_date"].values <= half
    m2 = ~m1
    print(f"训练集 {len(p):,} 行 / {len(dates)} 日, 分界 {half} "
          f"(前 {m1.sum():,} / 后 {m2.sum():,})", flush=True)

    cur = np.array([0.20, 0.70, 0.05, 0.02, 0.03, 0.00, 0.00, 0.00])
    print("\n=== 基准(现行权重, 低开子集内) ===")
    for nm, b in [("全", stats(p, cur)), ("前半", stats(p, cur, m1)), ("后半", stats(p, cur, m2))]:
        print(f"  {nm}: t={b['t']:5.2f} 年化{b['ann']:6.1f}% 回撤{b['mdd']:6.1f}% "
              f"单笔{b['avg']:6.3f}% 胜率{b['win']:4.1f}%", flush=True)

    print("\n=== 训练(目标: 前半段 t 值) ===", flush=True)
    rng = np.random.default_rng(22511)
    best = []
    t0 = time.time()
    for i in range(30):
        w = rng.dirichlet(np.ones(len(FACTORS)) * 1.5)
        r = stats(p, w, m1)
        if not r:
            continue
        cb, cw, step = r["ann"], w.copy(), 0.5
        for _ in range(400):
            cand = np.clip(cw + rng.normal(0, step, len(FACTORS)), -1.0, 1.0)
            rc = stats(p, cand, m1)
            if rc and rc["ann"] > cb:
                cb, cw = rc["ann"], cand
            step *= 0.994
        best.append((cb, cw))
    best.sort(key=lambda x: -x[0])
    print(f"  完成 {time.time()-t0:.0f}s, 前半最佳年化={best[0][0]:.1f}%", flush=True)

    print("\n=== top10 (前/后半均报) ===")
    rows = []
    for cb, w in best[:10]:
        a, b2 = stats(p, w), stats(p, w, m2)
        print(f"前年化{cb:6.1f}% | 全 {a['ann']:6.1f}%/{a['mdd']:6.1f}% 尾{a['tail']:4.1f}% | 后 {b2['ann']:6.1f}%/{b2['mdd']:6.1f}% 尾{b2['tail']:4.1f}%")
        print(f"   {np.round(w, 3).tolist()}")
        rows.append({"t_first": cb, "t_all": a["t"], "ann_all": a["ann"], "avg_all": a["avg"],
                     "t_back": b2["t"], "ann_back": b2["ann"], "avg_back": b2["avg"],
                     "weights": {k: float(v) for k, v in zip(FACTORS, w)}})

    # 只保留"两段 t 都为正且后半不劣"的候选
    robust = [r for r in rows if r["ann_back"] > 0 and r["t_back"] > 0.5]
    json.dump([{k: (float(v) if not isinstance(v, dict) else {kk: float(vv) for kk, vv in v.items()})
                for k, v in r.items()} for r in robust[:6]],
              open(OUT / "w225c_best.json", "w"), ensure_ascii=False, indent=1)
    print(f"\n后半年化为正且 t>0.5 的候选 {len(robust)} 组, 已存 w225c_best.json")


if __name__ == "__main__":
    main()
