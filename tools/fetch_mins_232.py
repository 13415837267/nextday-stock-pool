"""第232轮 · 抓取逐笔的**信号日**分钟行情

为什么是信号日而不是买入日：策略的 edge 是"T 日深跌 → T+1 浅低开 → T+2 修复"。
日线只告诉我们 T 日跌了多少，看不到**什么时候跌的**。
尾盘 30 分钟急砸（被动卖出/强平/减持）与全天均匀阴跌（趋势下行）在日线上是同一根 K 线，
但在次日修复的概率上可能完全不同 —— 这是日线数据天然缺失的一维。

信号日数据在 T 日收盘后即完全可知，用它做 T+1 的决策不含未来函数。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "output" / "mins_232"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    t = pd.read_csv(ROOT / "output" / "w232_trades.csv", dtype={"signal": str})
    jobs = sorted({(c, d) for c, d in zip(t.code, t.signal)})
    todo = [(c, d) for c, d in jobs if not (OUT / f"{c}_{d}.csv.gz").exists()]
    print(f"共 {len(jobs)} 个(票,信号日), 待抓 {len(todo)}", flush=True)

    from zzshare.client import DataApi
    api = DataApi()
    bad, t0 = [], time.time()
    for k, (code, d) in enumerate(todo, 1):
        got = None
        for _ in range(3):
            try:
                got = api.stk_mins(ts_code=code, trade_time=d, count=300)
                if got is not None and len(got):
                    break
            except Exception:
                time.sleep(0.6)
        if got is None or not len(got):
            bad.append((code, d))
        else:
            got.to_csv(OUT / f"{code}_{d}.csv.gz", index=False, compression="gzip")
        if k % 50 == 0:
            el = time.time() - t0
            print(f"{k}/{len(todo)} 失败{len(bad)} 已用{el/60:.1f}min "
                  f"预计还需{el/k*(len(todo)-k)/60:.1f}min", flush=True)
        time.sleep(0.12)
    print(f"完成, 失败 {len(bad)}", flush=True)
    if bad:
        print(bad[:10], flush=True)


if __name__ == "__main__":
    main()
