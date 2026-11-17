"""第248轮 H: 校验 src/fillin.py 与 tools/backfill_daily.py 同日同名单

fillin 是合入引擎的版本, backfill_daily 是独立的实盘脚本。两者必须给出同一张名单，
否则"回测-实盘同口径"就是空话。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import fillin  # noqa: E402
from src.config import load_config  # noqa: E402
from src.selector import build_indicators, regime_series, split_by_date  # noqa: E402


def main() -> None:
    cfg = load_config()
    fp = fillin.params(cfg)
    files = sorted((ROOT / "data" / "panel").glob("*.csv.gz"))[-(360 + 250 + 60):]
    frames = [pd.read_csv(f, dtype={"ts_code": "str"}) for f in files]
    panel = pd.concat(frames, ignore_index=True)
    del frames
    frames_raw = panel[["trade_date", "pct"]].copy()
    print(f"面板 {len(files)} 天 {len(panel):,} 行")
    fdist = fillin.market_dist(frames_raw)   # 清洗前
    panel = build_indicators(panel)
    panel = fillin.add_indicators(panel)
    reg = regime_series(panel, cfg)
    by_date = split_by_date(panel)
    last = reg.index[-1]
    _dt = panel["trade_date"].dtype
    print(f"最新交易日 {last}   (panel dtype={_dt})")

    idxd = float(fdist.loc[last])
    print(f"指数距20日线 {idxd:+.2f}%   (backfill_daily: -0.79%)")

    pl = fillin.plan(by_date[str(last)], idxd, fp)
    if pl is None:
        print("无计划")
        return
    print(f"路径 {pl['path']}  仓位 {fillin.pos_of(pl['path'], fp):.0%}")
    for c in pl["cands"]:
        r = by_date[str(last)][by_date[str(last)]["ts_code"].astype(str) == c["code"]].iloc[0]
        print(f"  #{c['rk']:<3}{c['code']}  close {r['close']:>8.2f}  pct {r['pct']:>6.2f}%  "
              f"dh60 {c['dh60']:>7.1f}%  amt {r['amount']/1e8:>6.2f}亿")

    # 与 backfill_daily.py 的最新输出对照
    print("\nbackfill_daily.py 的名单: 603256.SH 605376.SH 002384.SZ 002484.SZ 603986.SH "
          "003036.SZ 603516.SH 603773.SH 600105.SH 603115.SH")
    # T+1 开盘模拟: 用最新一天自己做假设(不可得, 只演示 choose 的判定)
    print("\nchoose() 用当日自身开盘价试算（真实场景是次日开盘）:")
    got = fillin.choose(pl, by_date[str(last)], fp)
    print(f"  → {got}")


def day_df_code(d: pd.DataFrame):
    return d["ts_code"].astype(str)


if __name__ == "__main__":
    main()
