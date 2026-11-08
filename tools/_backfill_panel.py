# -*- coding: utf-8 -*-
"""本地补齐全市场面板缓存(data/panel/)。

用途: 本地只读回测需要连续交易日快照; 仓库里只版本化了部分年份(2012/2017/
2020-2022/2025-2026), 中间年份缺失。本脚本按交易日历补齐缺失日, 命中缓存即跳过。

注意: 补出来的文件**不进 git**(体积 80MB 量级, 且可随时重建), 只作本地回测用。

用法: python3 tools/_backfill_panel.py --days 1561
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config          # noqa: E402
from src.data_api import ZZClient           # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=1561)
    args = ap.parse_args()
    cfg = load_config()
    c = ZZClient(cfg["data"])
    days = c.trade_days(days=args.days)
    print(f"日历 {len(days)} 日: {days[0]} ~ {days[-1]}", flush=True)
    miss = [d for d in days if not (c.panel_dir / f"{d}.csv.gz").exists()]
    print(f"缺失 {len(miss)} 日, 预计 {len(miss) * 0.5 / 60:.1f} 分钟", flush=True)
    t0 = time.time()
    ok = bad = 0
    for i, d in enumerate(miss, 1):
        try:
            df = c.market_day(d, use_cache=False)
            if len(df):
                c._save_panel(d, df)
                ok += 1
            else:
                bad += 1
        except Exception as e:  # noqa: BLE001
            bad += 1
            print(f"  ! {d} {e}", flush=True)
        if i % 50 == 0:
            print(f"  {i}/{len(miss)} 用时 {time.time() - t0:.0f}s", flush=True)
    print(f"完成: 补齐 {ok} 日, 失败/空 {bad} 日, 用时 {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
