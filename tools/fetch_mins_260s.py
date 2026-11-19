"""第260轮b · 抓取**卖出日**(T+2)分钟线 —— 卖得出去吗

第260轮量了买端: 开盘一笔只能吃全天成交额的 2.78%(首分钟占比), 容量 33 万。
卖端从没量过。策略默认 T+2 收盘卖出(15:00 收盘集合竞价), 227 轮把滑点当 0。
但收盘集合竞价同样只占全天一小部分 —— 如果它也装不下, 规模曲线要在卖端再打一次折。
"""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "mins_260s"


def main(limit: int | None = None):
    d = pickle.load(open(ROOT / "output" / "w258_skip_3200.pkl", "rb"))
    g = [x for x in d if x["name"].startswith("M ")][0]
    t = g["trades"]
    t = t[t.ret_pct.notna()]
    jobs = sorted({(c, s) for c, s in zip(t.code, t.sell_date) if isinstance(s, str) and s})
    if limit:
        jobs = jobs[:limit]
    OUT.mkdir(parents=True, exist_ok=True)
    todo = [(c, b) for c, b in jobs if not (OUT / f"{c}_{b}.csv.gz").exists()]
    print(f"共 {len(jobs)} 个(票,卖出日), 待抓 {len(todo)}", flush=True)

    from zzshare.client import DataApi
    api = DataApi()
    bad, t0 = [], time.time()
    for k, (code, day) in enumerate(todo, 1):
        got = None
        for _ in range(3):
            try:
                got = api.stk_mins(ts_code=code, trade_time=day, count=300)
                if got is not None and len(got):
                    break
            except Exception:  # noqa: BLE001
                time.sleep(0.6)
        if got is None or not len(got):
            bad.append((code, day))
            continue
        got.to_csv(OUT / f"{code}_{day}.csv.gz", index=False, compression="gzip")
        if k % 100 == 0:
            el = time.time() - t0
            print(f"  {k}/{len(todo)}  {el:.0f}s", flush=True)
    print(f"完成, 失败 {len(bad)}")
    if bad:
        print(bad[:8])


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
