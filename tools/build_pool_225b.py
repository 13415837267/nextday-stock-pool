"""第225轮B · 重建带执行条件的训练集（修正上一版的样本分布错误）

上一版的错误：用全体 dip 候选（含 T+1 高开的票）训练排序权重，但实盘只在
"T+1 低开 0~-2.5%"时买入。低开筛选会改变收益分布（低开越多，日内修复空间越大），
所以模型学到的是"在全体样本上期望最高"的方向（放量垃圾股），一进入低开子集就失效。

修正：用 T+1 开盘相对 T 收盘的缺口做**样本筛选**（不是当特征，不构成泄漏——
实盘同样是先看到 gap 再下单），只在 gap ∈ (-2.5%, 0) 的子集上训练。
输出: output/pool_225b.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"


def main() -> None:
    print("[1/3] 读价格表算 T+1 gap ...", flush=True)
    pt = pd.read_parquet(OUT / "price_table_212.parquet",
                         columns=["ts_code", "trade_date", "open", "close", "high_limit"])
    pt = pt.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    g = pt.groupby("ts_code", observed=True, sort=False)
    pt["nopen"] = g["open"].shift(-1)          # T+1 开盘
    pt["nhl"] = g["high_limit"].shift(-1)      # T+1 涨停价
    pt["gap"] = pt["nopen"] / pt["close"] - 1
    # T+1 一字涨停 = 买不到, 剔除
    pt["unbuyable"] = pt["nopen"] >= pt["nhl"] * 0.999
    keep = pt[["ts_code", "trade_date", "gap", "unbuyable"]]
    print(f"  {len(keep):,} 行", flush=True)
    del pt

    print("[2/3] 合入候选池 ...", flush=True)
    p = pd.read_parquet(OUT / "pool_225.parquet")
    p["trade_date"] = p["trade_date"].astype("int64")
    keep["trade_date"] = keep["trade_date"].astype("int64")
    p = p.merge(keep, on=["ts_code", "trade_date"], how="left")
    print(f"  合并后 {len(p):,} 行, gap 缺失 {(p.gap.isna()).sum():,}", flush=True)

    n0 = len(p)
    p = p[p["gap"].notna() & (~p["unbuyable"].fillna(True))]
    p = p[p["gap"].between(-0.025, 0.0)]
    print(f"  低开 0~-2.5% 且非一字涨停: {len(p):,} 行 ({len(p)/n0*100:.1f}%) / "
          f"{p.trade_date.nunique()} 日", flush=True)

    print("[3/3] 检查子集内分布 ...", flush=True)
    print(f"  单笔均值 {p.ret.mean():.3f}%  胜率 {(p.ret>0).mean()*100:.1f}%", flush=True)
    p.to_parquet(OUT / "pool_225b.parquet", index=False)
    print(f"  已保存 {OUT/'pool_225b.parquet'}")


if __name__ == "__main__":
    main()
