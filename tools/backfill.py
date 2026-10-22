# -*- coding: utf-8 -*-
"""历史数据回填脚本(在 GitHub Actions 中运行):
按 [start, end] 区间拉取缺失的每日全市场快照并入库, 每 batch 天提交推送一次。
用法: python tools/backfill.py --start 20120409 --end 20261022 --batch 150
"""
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import load_config
from src.data_api import ZZClient


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)


def git_push_retry(tries=6):
    for i in range(tries):
        r = sh("git push origin HEAD:main")
        if r.returncode == 0:
            return True
        print(f"  push retry {i+1}: {r.stderr.strip()[-200:]}", flush=True)
        sh("git pull --rebase origin main")
        time.sleep(8)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="20120409", help="开始日期YYYYMMDD")
    ap.add_argument("--end", default="", help="结束日期YYYYMMDD(默认今天)")
    ap.add_argument("--batch", type=int, default=150, help="每批提交天数")
    args = ap.parse_args()
    end = args.end or datetime.now().strftime("%Y%m%d")

    cfg = load_config()
    client = ZZClient(cfg)

    days_total = (datetime.strptime(end, "%Y%m%d") - datetime.strptime(args.start, "%Y%m%d")).days
    cal = client.trade_days(days=int(days_total * 1.7) + 40)
    cal = [str(d).replace("-", "") for d in cal]
    cal = sorted(d for d in cal if args.start <= d <= end)
    print(f"交易日历: {len(cal)}个交易日 ({cal[0]}~{cal[-1]})", flush=True)

    existing = set(os.listdir("data/panel"))
    missing = [d for d in cal if f"{d}.csv.gz" not in existing]
    print(f"已有 {len(existing)} 文件, 缺失 {len(missing)} 个交易日", flush=True)

    t0 = time.time()
    batch = []
    for i, d in enumerate(missing, 1):
        f = f"data/panel/{d}.csv.gz"
        if not (os.path.exists(f) and os.path.getsize(f) > 500):
            try:
                df = client.market_day(d)
                if not len(df):
                    print(f"  {d}: 空数据(非交易日?), 跳过", flush=True)
                    continue
            except Exception as e:
                print(f"  {d}: FAIL {str(e)[:80]}", flush=True)
                continue
        batch.append(d)
        if len(batch) >= args.batch:
            sh("git add -f " + " ".join(f"data/panel/{x}.csv.gz" for x in batch))
            sh(f'git commit -m "历史数据回填: +{len(batch)}个交易日({batch[0]}~{batch[-1]}) [skip ci]"')
            ok = git_push_retry()
            print(f"  批次提交 {batch[0]}~{batch[-1]} ({len(batch)}天) pushed={ok} "
                  f"({i}/{len(missing)}, {time.time()-t0:.0f}s)", flush=True)
            batch = []
        if i % 100 == 0:
            print(f"  进度 {i}/{len(missing)}, 累计 {time.time()-t0:.0f}s", flush=True)
    if batch:
        sh("git add -f " + " ".join(f"data/panel/{x}.csv.gz" for x in batch))
        sh(f'git commit -m "历史数据回填补齐: +{len(batch)}个交易日 [skip ci]"')
        git_push_retry()
        print(f"  收尾批次 {len(batch)}天 已提交", flush=True)

    total = len([x for x in os.listdir("data/panel") if x.endswith(".csv.gz")])
    print(f"BACKFILL DONE: panel 文件总数 = {total}, 耗时 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
