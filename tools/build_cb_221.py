#!/usr/bin/env python3
"""第221轮: 可转债 T+0 面板构建。

为什么是可转债: A 股股票 T+1, 当天买不能当天卖, 物理上不存在"日内高频"。
可转债 T+0 且无印花税, 单次往返成本约 0.09%(股票 0.20%), 是唯一能做日内反复进出的
合法载体。日线的 open/close 刚好对应"开盘买、收盘卖"这一笔真实可执行的 T+0 交易。

关键设计:
1. 代码表用 bond_zh_cov()(含已退市券), 不用 spot(只有在市券), 否则幸存者偏差会把
   2021-2022 那批强赎退市前暴涨的券全部删掉。
2. 同时保留已退市标的: bond_zh_hs_cov_daily 对退市券仍能返回完整历史。
3. 只落 OHLCV 原样, 特征在训练脚本里算, 避免这里引入口径。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import akshare as ak
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "cb"


def load_universe() -> pd.DataFrame:
    """全量转债代码(含已退市)。"""
    d = ak.bond_zh_cov()
    d = d[["债券代码", "债券简称", "正股代码", "正股简称", "上市时间", "发行规模", "信用评级"]].copy()
    d.columns = ["code", "name", "stock_code", "stock_name", "list_date", "issue_size", "rating"]
    d["code"] = d["code"].astype(str).str.zfill(6)
    d["list_date"] = pd.to_datetime(d["list_date"], errors="coerce")
    d = d.drop_duplicates("code")
    return d


def guess_symbol(code: str) -> str:
    """东财 symbol: 沪 11x -> sh, 深 12x -> sz, 北交 8x -> bj。"""
    if code.startswith(("110", "111", "113", "118")):
        return "sh" + code
    if code.startswith(("123", "127", "128")):
        return "sz" + code
    return "bj" + code


def fetch_one(code: str) -> pd.DataFrame | None:
    for sym in (guess_symbol(code), "sz" + code, "sh" + code):
        try:
            d = ak.bond_zh_hs_cov_daily(symbol=sym)
            if d is not None and len(d):
                d = d.copy()
                d["code"] = code
                return d
        except Exception:  # noqa: BLE001
            continue
        time.sleep(0.12)
    return None


def main(limit: int | None = None) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    uni = load_universe()
    print(f"全量转债 {len(uni)} 只, 上市 {uni['list_date'].min().date()} ~ {uni['list_date'].max().date()}")
    uni.to_csv(OUT / "universe.csv", index=False)

    cache = OUT / "daily.parquet"
    old = pd.read_parquet(cache) if cache.exists() else None
    done = set(old["code"].unique()) if old is not None else set()

    todo = uni[~uni["code"].isin(done)]
    if limit:
        todo = todo.head(limit)
    print(f"待抓 {len(todo)} 只 (已有 {len(done)} 只)")

    frames, ok, bad = [], 0, 0
    t0 = time.time()
    for i, code in enumerate(todo["code"], 1):
        d = fetch_one(code)
        if d is None or len(d) == 0:
            bad += 1
        else:
            ok += 1
            frames.append(d[["code", "date", "open", "high", "low", "close", "volume"]])
        if i % 50 == 0:
            el = time.time() - t0
            print(f"  {i}/{len(todo)}  成功 {ok} 失败 {bad}  {el:.0f}s  预计还需 {el/i*(len(todo)-i):.0f}s",
                  flush=True)
        time.sleep(0.18)

    if not frames:
        print("无新数据")
        return
    new = pd.concat(frames, ignore_index=True)
    new["date"] = pd.to_datetime(new["date"])
    for c in ("open", "high", "low", "close", "volume"):
        new[c] = pd.to_numeric(new[c], errors="coerce")
    all_df = pd.concat([old, new], ignore_index=True) if old is not None else new
    all_df = all_df.drop_duplicates(["code", "date"]).sort_values(["code", "date"])
    all_df.to_parquet(cache, index=False)
    print(f"完成: 成功 {ok} 失败 {bad}; 总行 {len(all_df):,} 券数 {all_df.code.nunique()}")
    print(f"区间 {all_df.date.min().date()} ~ {all_df.date.max().date()}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
