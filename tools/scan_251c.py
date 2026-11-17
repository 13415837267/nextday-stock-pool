"""第251轮 C: 页面名单与 backfill 名单为什么不同

现象: 2026-11-17 CI 的 latest.json 第一位是 603629.SH 利通电子,
      tools/backfill_daily.py 的最新表第一位是 603256.SH 宏和科技, 且 603629 不在前 10。
两者都号称"9 窗口超跌排名", 本脚本用同一份面板分别跑两条实现, 定位差异。
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import fillin  # noqa: E402
from src.selector import build_indicators  # noqa: E402

D = 20261117


def main() -> None:
    files = sorted(glob.glob(str(ROOT / "data" / "panel" / "*.csv.gz")))[-1200:]
    p = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    print(f"面板 {len(files)} 天, {len(p):,} 行")
    p = build_indicators(p)
    p = fillin.add_indicators(p)
    day = p[p.trade_date == D]

    prm = fillin.params({})
    r = fillin.rank_day(day, prm)
    print(f"\n[fillin.rank_day] 池 {len(r)}")
    cols = ["ts_code", "close", "pct", "mr"] + fillin.DH_COLS
    print(r[cols].head(10).to_string(index=False))
    t = r[r.ts_code == "603629.SH"]
    if len(t):
        print(f"\n603629.SH 在 fillin 里: rk={int(t.rk.iloc[0])} mr={t.mr.iloc[0]:.6f} "
              f"amount={t.amount.iloc[0]/1e8:.2f}亿")

    # backfill 口径: 同样池子, 但 dh 全部非空 + bars>=250 才进(它是在 win 上算排名)
    w = day[(day.is_paused == 0) & (day.is_st == 0)]
    w = w[w.pct.between(-11.5, 11.5)]
    w = w[w.ts_code.astype(str).str[:2].isin(["60", "00"])]
    w = w[(w.bars >= 250) & w.dh60.notna()].copy()
    w = w.dropna(subset=fillin.DH_COLS).copy()
    den = {c: max(int(w[c].notna().sum()), 1) for c in fillin.DH_COLS}
    rk = pd.DataFrame({c: w[c].rank(ascending=True) / den[c] for c in fillin.DH_COLS})
    w["mr"] = rk.mean(axis=1, skipna=True)
    w = w.sort_values(["mr", "amount"], ascending=[True, False]).reset_index(drop=True)
    print(f"\n[backfill 口径] 池 {len(w)}")
    print(w[cols].head(10).to_string(index=False))
    tt = w[w.ts_code == "603629.SH"]
    if len(tt):
        print(f"\n603629.SH 在 backfill 口径: 第 {int(tt.index[0])+1} 名 mr={tt.mr.iloc[0]:.6f}")
    else:
        print("\n603629.SH 不在 backfill 的池里")
        raw = day[day.ts_code == "603629.SH"]
        if len(raw):
            x = raw.iloc[0]
            print(f"  原始: is_paused={x.get('is_paused')} is_st={x.get('is_st')} "
                  f"pct={x.pct:.2f} bars={x.get('bars')} dh60={x.get('dh60')}")
            print("  dh 缺失:", [c for c in fillin.DH_COLS if pd.isna(x.get(c))])


if __name__ == "__main__":
    main()
