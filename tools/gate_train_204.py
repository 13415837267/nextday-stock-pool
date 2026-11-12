"""第 204 轮: 门槛(而不是权重)才是选股条件的抓手 —— 全史面板上的单因子扫描.

背景: 202 轮曾得出"评分 +0.978pp 是唯一选股 alpha"。本轮把训练集扩到 22.9 万个
股票日(硬条件池 4.5 万)后, 用现行 5 因子在同一池内取 top1, 结果**低于**池均值
—— 说明"排序"这件事本身不产生 alpha, 202 轮的 +0.978pp 是口径差(未扣成本/分档
方式)造成的。既然排序无效, 提高期望只能靠"把坏样本挡在池外", 也就是门槛。

本脚本:
  1. 复核排序 alpha: 现行权重 / 等权 / 随机(池均值) / 反向 的 top1 收益
  2. 单因子门槛扫描: 每个门槛取 3~4 档, 记池均值、top1 均值、笔数、胜率
  3. 前 8 年 / 后 8 年分别统计, 看方向是否稳定(这是能不能改 config 的前提)

输出 output/gate_train_204.json / .csv
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
COST = 0.30
FACTORS = [("mom20", "mom20", 1.0), ("neg_bias20", "bias20", -1.0),
           ("neg_amount", "logamt", -1.0), ("neg_close_pos", "close_pos", -1.0),
           ("neg_vol_ratio", "vr", -1.0), ("low_tail", "low_tail", 1.0)]
CUR_W = {"mom20": 0.20, "neg_bias20": 0.70, "neg_amount": 0.05,
         "neg_vol_ratio": 0.02, "low_tail": 0.03}


def prep() -> pd.DataFrame:
    d = pd.read_parquet(OUT / "panel_train_204.parquet")
    bad = (d["ret"].abs() > 25) | (d["gap"].abs() > 15)
    d = d[~bad].reset_index(drop=True)
    d["year"] = d["d"].str[:4].astype("int16")
    d["logamt"] = np.log1p(d["amt_yi"].astype("float64") * 1e8)
    g = d.groupby("d", sort=False)
    for k, col, sign in FACTORS:
        x = d[col].astype("float64")
        lo = g[col].transform(lambda v: v.quantile(0.01))
        hi = g[col].transform(lambda v: v.quantile(0.99))
        s = sign * x.clip(lo, hi)
        mu, sd = s.groupby(d["d"], sort=False).transform("mean"), s.groupby(d["d"], sort=False).transform("std")
        d["z_" + k] = ((s - mu) / sd.replace(0, np.nan)).fillna(0.0)
    d["_s"] = sum(CUR_W[k] * d["z_" + k] for k in CUR_W)
    d["_s_rev"] = -d["_s"]
    d["_s_eq"] = sum(d["z_" + k] for k in CUR_W)
    return d


def stat(d: pd.DataFrame, label: str) -> dict:
    if d.empty:
        return {}
    pick = d.loc[d.groupby("d", sort=False)["_s"].idxmax()]
    pool = d.groupby("d", sort=False)["ret"].mean()
    net = pick["ret"] - COST
    # 随机基线: 池均值(扣成本) —— 每天买"池里随便一只"的期望
    rnd = pool - COST
    return {"label": label, "n": int(len(d)), "days": int(d["d"].nunique()),
            "pool": round(float(rnd.mean()), 3), "top1": round(float(net.mean()), 3),
            "top1_win": round(float((net > 0).mean() * 100), 2),
            "top1_med": round(float(net.median()), 3),
            "excess": round(float(net.mean() - rnd.mean()), 3)}


def topk(d: pd.DataFrame, k: int) -> float:
    """前 k 名的平均净收益(引擎 exec_top_n=1, 但 k>1 可看排序是否有效)。"""
    if d.empty:
        return np.nan
    r = d.sort_values("_s", ascending=False).groupby("d", sort=False).head(k)
    return float((r["ret"] - COST).mean())


def main() -> None:
    d = prep()
    hard = d["hard"] == 1
    rows, js = [], {}

    # 1) 排序 alpha 复核
    h = d[hard]
    js["排序复核"] = {
        "现行权重 top1": stat(h, "cur")["top1"],
        "池均值(随机一只)": stat(h, "cur")["pool"],
        "反向权重 top1": round(float((h.loc[h.groupby("d", sort=False)["_s_rev"].idxmax(), "ret"] - COST).mean()), 3),
        "等权 top1": round(float((h.loc[h.groupby("d", sort=False)["_s_eq"].idxmax(), "ret"] - COST).mean()), 3),
        "前 3 名均值": round(topk(h, 3), 3), "前 5 名均值": round(topk(h, 5), 3),
        "全池(样本加权)": round(float((h["ret"] - COST).mean()), 3),
        "每日候选数中位": int(h.groupby("d").size().median()),
    }
    print("排序复核:", json.dumps(js["排序复核"], ensure_ascii=False))

    # 2) 单因子门槛扫描(其余条件保持现行)
    base = d[hard]
    sweeps = {
        "pct_min": ("pct", [-8.0, -7.0, -6.0, -5.5]),
        "pct_max": ("pct", [-4.0, -3.5, -3.0, -2.5]),
        "close_pos_max": ("close_pos", [0.10, 0.18, 0.25, 0.32]),
        "mom20_min": ("mom20", [5.0, 8.0, 10.0, 13.0]),
        "mom20_max": ("mom20", [20.0, 25.0, 30.0]),
        "dist_low20_min": ("dist_low20", [6.0, 10.0, 15.0, 20.0]),
        "vr_max": ("vr", [1.0, 1.2, 1.5, 2.0, 3.0, 4.0]),
        "vr_min": ("vr", [0.3, 0.5, 0.7, 0.9]),
        "amt_max": ("amt_yi", [5.0, 8.0, 12.0, 20.0, 1e9]),
        "price_max": ("price", [15.0, 25.0, 40.0, 150.0]),
        "low_tail_min": ("low_tail", [-1.0, 0.0, 0.3, 0.6]),
        "amp_max": ("amp_pct", [5.0, 7.0, 9.0, 20.0]),
        "turnover_cool_max": ("turnover_cool", [0.8, 1.0, 1.2, 9.0]),
    }
    for name, (col, vals) in sweeps.items():
        for v in vals:
            m = base
            if name.endswith("_min"):
                m = m[base[col] >= v]
            elif name.endswith("_max"):
                m = m[base[col] <= v]
            st = stat(m, f"{name}={v:g}")
            if not st:
                continue
            st["factor"], st["value"] = col, v
            early = m[m.year < 2018]
            late = m[m.year >= 2018]
            st["early_pool"] = round(float((early["ret"] - COST).mean()), 3) if len(early) > 200 else None
            st["late_pool"] = round(float((late["ret"] - COST).mean()), 3) if len(late) > 200 else None
            rows.append(st)
            print(f"  {name:18s}{v:>9g}  笔 {st['n']:>6d}  池 {st['pool']:+.3f}%  "
                  f"top1 {st['top1']:+.3f}%  胜率 {st['top1_win']:.1f}%  "
                  f"前 {st['early_pool']} / 后 {st['late_pool']}", flush=True)

    pd.DataFrame(rows).to_csv(OUT / "gate_train_204.csv", index=False)
    (OUT / "gate_train_204.json").write_text(
        json.dumps({"排序复核": js["排序复核"], "扫描": rows}, ensure_ascii=False, indent=1),
        encoding="utf-8")


if __name__ == "__main__":
    main()
