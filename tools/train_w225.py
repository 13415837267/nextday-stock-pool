"""第225轮 · 用全历史样本训练 dip 排序权重（以收益为目标，不是以IC为目标）

与之前几轮的区别（这次真的是"训练提高收益"）：
  - 目标函数 = 真实复利年化 + 卡玛，不是 IC、不是 AIC。
  - 训练空间 = dip_weights 的 8 个因子权重（现行 5 个里 2 个为 0 权重的可选因子也打开）。
  - 不改变任何"是否交易"的规则（regime/仓位分层/成本全部沿用 config），
    因此回测验证时只有"每天选谁"变了，差异可归因于排序本身。
  - 全样本 412 万行，2010-2026，无幸存者偏差。

标签: ml_220 的 ret = (adj_close[T+2]/adj_open[T+1]-1)*100 - 0.30（双边成本），
      即 T+1 集合竞价买入、T+2 收盘卖出，与策略完全一致。

输出: output/w225_best.json（若干组权重）+ 控制台排序表
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

# 与 config.yaml selection 一致的 dip 硬筛选
DIP_PCT_MIN, DIP_PCT_MAX = -6.0, -3.0
VR_MAX = 4.0
MOM20_MIN, MOM20_MAX = 5.0, 25.0
BIAS20_MAX = 11.4
DIST_LOW20_MIN = 6.0
TURNOVER_MIN, TURNOVER_MAX = 2.0, 12.0

FACTORS = ["mom20", "neg_bias20", "neg_amount", "neg_vol_ratio",
           "low_tail", "neg_turnover_cool", "ivol", "neg_dist_low20"]

COLS = ["trade_date", "ts_code", "ret", "pct", "mom20", "bias20", "amt_yi",
        "vr", "dist_low20", "turnover_rate", "turnover_cool", "vol20", "mkt_ret"]


def load_pool() -> pd.DataFrame:
    t0 = time.time()
    df = pd.read_parquet(OUT / "ml_220.parquet", columns=COLS)
    print(f"  读入 {len(df):,} 行 {time.time()-t0:.0f}s", flush=True)

    m = (
        df["pct"].between(DIP_PCT_MIN, DIP_PCT_MAX)
        & (df["vr"] <= VR_MAX)
        & df["mom20"].between(MOM20_MIN, MOM20_MAX)
        & (df["bias20"] <= BIAS20_MAX)
        & (df["dist_low20"] >= DIST_LOW20_MIN)
        & df["turnover_rate"].between(TURNOVER_MIN, TURNOVER_MAX)
        & df["ret"].notna()
    )
    p = df[m].copy().reset_index(drop=True)
    del df
    print(f"  过 dip 硬筛选: {len(p):,} 行 / {p.trade_date.nunique()} 日", flush=True)

    # 逐日截面 z-score（与 selector._zs 同口径）
    g = p.groupby("trade_date", observed=True)
    z = pd.DataFrame(index=p.index)
    z["mom20"] = g["mom20"].transform(lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-9))
    z["neg_bias20"] = g["bias20"].transform(lambda s: -((s - s.mean()) / (s.std(ddof=0) + 1e-9)))
    lgamt = np.log1p(p["amt_yi"])
    z["neg_amount"] = -((lgamt - lgamt.groupby(p["trade_date"]).transform("mean"))
                        / (lgamt.groupby(p["trade_date"]).transform("std") + 1e-9))
    z["neg_vol_ratio"] = g["vr"].transform(lambda s: -((s - s.mean()) / (s.std(ddof=0) + 1e-9)))
    # close_pos / low_tail 训练集里没有，用 turnover_cool 与 vol20 补位
    z["low_tail"] = g["vol20"].transform(lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-9))
    tc = p["turnover_cool"].fillna(1.0)
    z["neg_turnover_cool"] = -((tc - tc.groupby(p["trade_date"]).transform("mean"))
                               / (tc.groupby(p["trade_date"]).transform("std") + 1e-9))
    z["ivol"] = g["vol20"].transform(lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-9))
    z["neg_dist_low20"] = g["dist_low20"].transform(lambda s: -((s - s.mean()) / (s.std(ddof=0) + 1e-9)))

    p[FACTORS] = z[["mom20", "neg_bias20", "neg_amount", "neg_vol_ratio",
                    "low_tail", "neg_turnover_cool", "ivol", "neg_dist_low20"]].values
    p[FACTORS] = p[FACTORS].fillna(0.0)
    p = p.drop(columns=["vr", "turnover_cool", "vol20", "bias20", "amt_yi",
                        "dist_low20", "turnover_rate", "pct", "mkt_ret"])
    return p


def evaluate(p: pd.DataFrame, w: np.ndarray, dates: np.ndarray | None = None,
             mask: np.ndarray | None = None) -> dict:
    """在给定样本上按权重打分, 每日取 top1, 返回复利年化/回撤/卡玛。"""
    X = p[FACTORS].values
    score = X @ w
    td = p["trade_date"].values
    ret = p["ret"].values

    if mask is not None:
        score, td, ret = score[mask], td[mask], ret[mask]

    order = np.lexsort((-score, td))
    td_s, ret_s = td[order], ret[order]
    first = np.ones(len(td_s), dtype=bool)
    first[1:] = td_s[1:] != td_s[:-1]
    r = ret_s[first] / 100.0          # 每日一笔, 满仓

    if len(r) < 30:
        return {}
    nav = np.cumprod(1 + r)
    peak = np.maximum.accumulate(nav)
    mdd = float((nav / peak - 1).min())
    u = np.unique(td_s[first])
    d0, d1 = pd.Timestamp(str(u[0])), pd.Timestamp(str(u[-1]))
    yrs = max(0.5, (d1 - d0).days / 365.25)
    ann = float(nav[-1] ** (1 / yrs) - 1)
    return {"ann": ann * 100, "mdd": mdd * 100,
            "calmar": ann / abs(mdd) if mdd else np.nan,
            "n": int(len(r)), "win": float((r > 0).mean() * 100),
            "avg": float(r.mean() * 100)}


def main() -> None:
    print("[1/3] 构建候选池 ...", flush=True)
    p = load_pool()
    p.to_parquet(OUT / "pool_225.parquet", index=False)
    dates = np.sort(p["trade_date"].unique())
    half = dates[len(dates) // 2]
    m1 = p["trade_date"].values <= half
    m2 = ~m1
    print(f"  前半段 {m1.sum():,} 行 / 后半段 {m2.sum():,} 行, 分界 {half}", flush=True)

    # 现行权重（作为基准）
    cur = np.array([0.20, 0.70, 0.05, 0.02, 0.03, 0.00, 0.00, 0.00])
    print("\n[2/3] 基准（现行权重）")
    b_all, b1, b2 = evaluate(p, cur), evaluate(p, cur, mask=m1), evaluate(p, cur, mask=m2)
    for nm, b in [("全样本", b_all), ("前半", b1), ("后半", b2)]:
        print(f"  {nm}: 年化 {b['ann']:6.1f}% / 回撤 {b['mdd']:6.1f}% / 卡玛 {b['calmar']:5.2f} "
              f"/ 笔 {b['n']:4d} / 胜率 {b['win']:4.1f}% / 单笔 {b['avg']:5.3f}%", flush=True)

    print("\n[3/3] 随机搜索 + 坐标下降（目标: 前半段卡玛, 用后半段验收）", flush=True)
    rng = np.random.default_rng(20261114)
    best = []
    # 粗随机: 20 组种子 × 每组 300 次扰动
    for i in range(20):
        w = rng.dirichlet(np.ones(len(FACTORS)) * 1.2)
        r = evaluate(p, w, mask=m1)
        if not r:
            continue
        cur_best = r["calmar"]
        cur_w = w.copy()
        step = 0.35
        for _ in range(300):
            cand = cur_w + rng.normal(0, step, len(FACTORS))
            cand = np.clip(cand, -1.5, 1.5)
            rc = evaluate(p, cand, mask=m1)
            if rc and rc["calmar"] > cur_best:
                cur_best, cur_w = rc["calmar"], cand
            step *= 0.995
        allr = evaluate(p, cur_w)
        best.append((cur_best, allr, cur_w))
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/20 组完成, 当前前半最佳卡玛 {cur_best:.2f}", flush=True)

    best.sort(key=lambda x: -x[0])
    rows = []
    for cb, allr, w in best[:8]:
        r2 = evaluate(p, w, mask=m2)
        rows.append({
            "前半卡玛": round(cb, 2), "全样本年化": round(allr["ann"], 1),
            "全样本回撤": round(allr["mdd"], 1), "全样本卡玛": round(allr["calmar"], 2),
            "后半卡玛": round(r2["calmar"], 2), "后半年化": round(r2["ann"], 1),
            "笔数": allr["n"], "胜率": round(allr["win"], 1),
            "权重": {k: round(float(v), 2) for k, v in zip(FACTORS, w)},
        })
    tab = pd.DataFrame(rows)
    print("\n=== 训练结果 top8 ===")
    for _, r in tab.iterrows():
        print(f"前半卡玛 {r['前半卡玛']:5.2f} | 全 {r['全样本年化']:6.1f}%/{r['全样本回撤']:6.1f}%/"
              f"{r['全样本卡玛']:5.2f} | 后半 {r['后半年化']:6.1f}%/{r['后半卡玛']:5.2f} | "
              f"胜率 {r['胜率']:4.1f}%")
        print(f"    {r['权重']}")

    json.dump([{"calmar_first": float(a), "ann_all": float(b["ann"]), "mdd_all": float(b["mdd"]),
                "calmar_all": float(b["calmar"]), "weights": dict(zip(FACTORS, map(float, w)))}
               for a, b, w in best[:8]],
              open(OUT / "w225_best.json", "w"), ensure_ascii=False, indent=1)
    print(f"\n已保存 {OUT/'w225_best.json'}")


if __name__ == "__main__":
    main()
