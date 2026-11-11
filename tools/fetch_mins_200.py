# -*- coding: utf-8 -*-
"""第200轮 · 抓取主策略逐笔的分钟行情（买入日 + 卖出日）

为什么现在才做: 前几轮一直把"分钟级"列为"需要外部系统"。实际上 zzshare 的
`stk_mins(ts_code, trade_time=YYYYMMDD, count=300)` 能取**任意历史交易日**的
完整 240 根 1 分钟 K 线（已用 2015/2018/2021/2025 四个时点验证），
且与日线 OHLC 逐位一致（未复权价）。

存储: output/mins_200/{code}_{date}.csv.gz，可复跑(命中即跳过)。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "output" / "mins_200"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    t = pd.read_csv(ROOT / "output" / f"exp_factor_197_trades_{days}.csv",
                    dtype={"buy": str, "sell": str})
    t["buy"] = t.buy.str.replace("-", "")
    t["sell"] = t.sell.str.replace("-", "")
    jobs = sorted({(c, d) for c, b, s in zip(t.code, t.buy, t.sell) for d in (b, s)})
    todo = [(c, d) for c, d in jobs if not (OUT / f"{c}_{d}.csv.gz").exists()]
    print(f"共 {len(jobs)} 个(票,日), 已有 {len(jobs)-len(todo)}, 待抓 {len(todo)}")

    from zzshare.client import DataApi
    api = DataApi()
    bad, t0 = [], time.time()
    for k, (code, d) in enumerate(todo, 1):
        for attempt in range(3):
            try:
                x = api.stk_mins(ts_code=code, trade_time=d, freq="1min", count=300)
                if x is None or not len(x):
                    bad.append((code, d, "empty"))
                    break
                x = x.sort_values("trade_time").reset_index(drop=True)
                x.to_csv(OUT / f"{code}_{d}.csv.gz", index=False, compression="gzip")
                break
            except Exception as e:
                if attempt == 2:
                    bad.append((code, d, str(e)[:60]))
                else:
                    time.sleep(1.5)
        if k % 50 == 0:
            print(f"  {k}/{len(todo)}  累计 {time.time()-t0:.0f}s  失败 {len(bad)}", flush=True)
    print(f"完成: 抓 {len(todo)-len(bad)}, 失败 {len(bad)}, 用时 {time.time()-t0:.0f}s")
    if bad:
        pd.DataFrame(bad, columns=["code", "date", "err"]).to_csv(
            ROOT / "output" / f"mins_200_bad_{days}.csv", index=False)
        print("失败清单 ->", f"output/mins_200_bad_{days}.csv")


if __name__ == "__main__":
    main()
