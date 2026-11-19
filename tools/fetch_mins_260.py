"""第260轮 · 抓取**买入日**(T+1)分钟线, 用来量"开盘价成交"的真实容量

为什么: 第129轮把容量算成 5% × **全天**成交额 = 1250 万。
但那笔钱不是"全天慢慢买", 是 **9:25 集合竞价一笔吃进**。
集合竞价(含 9:30 首分钟)成交额通常只占全天个位数百分比,
所以真实可装资金 = 5% × 竞价成交额, 而不是 5% × 全天。
这个比例从来没量过, 本轮把它量出来。日线数据里没有, 只能看分钟线。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "output" / "mins_260"
OUT.mkdir(parents=True, exist_ok=True)


def main(limit: int | None = None):
    import pickle
    d = pickle.load(open(ROOT / "output" / "w258_skip_3200.pkl", "rb"))
    g = [x for x in d if x["name"].startswith("M ")][0]
    t = g["trades"]
    t = t[t.ret_pct.notna()]
    jobs = sorted({(c, b) for c, b in zip(t.code, t.buy_date)})
    if limit:
        jobs = jobs[:limit]
    todo = [(c, b) for c, b in jobs if not (OUT / f"{c}_{b}.csv.gz").exists()]
    print(f"共 {len(jobs)} 个(票,买入日), 待抓 {len(todo)}", flush=True)

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
            except Exception as e:  # noqa: BLE001
                if k <= 3:
                    print("  err", code, day, type(e).__name__, e, flush=True)
                time.sleep(0.6)
        if got is None or not len(got):
            bad.append((code, day))
            continue
        got.to_csv(OUT / f"{code}_{day}.csv.gz", index=False, compression="gzip")
        if k % 50 == 0:
            el = time.time() - t0
            print(f"  {k}/{len(todo)}  {el:.0f}s  预计剩 {el / k * (len(todo) - k):.0f}s", flush=True)
    print(f"完成, 失败 {len(bad)}")
    if bad:
        print(bad[:10])


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
