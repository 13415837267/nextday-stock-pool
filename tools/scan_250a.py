"""第250轮 A: 实盘链路(daily_run 口径)下补位腿的产出校验。

用 data/panel 里最近 620 个交易日, 完全按 src/daily_run.py 的顺序算:
  market_dist(清洗前) -> build_indicators -> fillin.add_indicators -> fillin.plan
与 tools/backfill_daily.py 同一天(2026-11-17)的名单逐只对比。
"""

from __future__ import annotations

import glob
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import fillin                      # noqa: E402
from src.config import load_config          # noqa: E402
from src.selector import build_indicators   # noqa: E402

WARM = 620


def main() -> None:
    cfg = load_config()
    fp = fillin.params(cfg)
    print(f"fillin.enabled={fp['enabled']}  let_way={fp['let_way']}")

    days = sorted(int(Path(f).name.replace(".csv.gz", ""))
                  for f in glob.glob(str(ROOT / "data" / "panel" / "*.csv.gz")))[-WARM:]
    print(f"面板 {days[0]} ~ {days[-1]}  共 {len(days)} 天")

    t0 = time.time()
    frames = [pd.read_csv(ROOT / "data" / "panel" / f"{d}.csv.gz") for d in days]
    panel = pd.concat(frames, ignore_index=True)
    del frames
    print(f"拼接 {len(panel):,} 行  {time.time()-t0:.1f}s")

    # ① 清洗前算指数 —— 与 daily_run 一致
    t1 = time.time()
    dist = fillin.market_dist(panel)
    print(f"market_dist {time.time()-t1:.1f}s  index dtype={dist.index.dtype}")
    last = days[-1]
    print(f"  最新日 {last} 距20日线 {dist.loc[last]:.2f}%")

    # ② 指标
    t2 = time.time()
    panel = build_indicators(panel)
    print(f"build_indicators {time.time()-t2:.1f}s")

    t3 = time.time()
    panel = fillin.add_indicators(panel)
    print(f"add_indicators {time.time()-t3:.1f}s  列: {[c for c in panel.columns if c.startswith('dh')]}")

    day = panel[panel["trade_date"] == last]
    t4 = time.time()
    rk = fillin.rank_day(day, fp)
    print(f"rank_day {time.time()-t4:.1f}s  池子 {len(rk)} 只")

    pl = fillin.plan(day, fillin.idx_dist_at(dist, last), fp)
    if pl is None:
        print("❌ plan 返回 None")
        return
    print(f"\n路径 {pl['path']}   候选 {len(pl['cands'])} 只")
    for i, c in enumerate(pl["cands"], 1):
        print(f"  {i:>2}  {c['code']}  rk={c['rk']:>3}  收盘 {c['close_adj']:>8.2f}  dh60 {c['dh60']:>7.1f}%")

    # ③ 与 backfill_daily.py 名单对比
    f = ROOT / "output" / "backfill_latest.csv"
    print(f"\nbackfill 名单文件存在: {f.exists()}")
    alt = ["603256.SH", "605376.SH", "002384.SZ", "002484.SZ", "603986.SH",
           "003036.SZ", "603516.SH", "603773.SH", "600105.SH", "603115.SH"]
    got = [c["code"] for c in pl["cands"]]
    print(f"  backfill 前10: {alt}")
    print(f"  fillin   前10: {got}")
    print(f"  一致 {len(set(alt) & set(got))}/10")


if __name__ == "__main__":
    main()
