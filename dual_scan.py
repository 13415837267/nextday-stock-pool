# -*- coding: utf-8 -*-
"""双策略互补性扫描: dip 与 momentum 两种模式的全周期信号级对比。
输出 dual_strategy.csv, 用于分析互补性与组合价值。"""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import copy
import numpy as np
import pandas as pd

from src.config import load_config
from src.selector import build_indicators, regime_base, select_stocks, split_by_date

CFG = load_config()
EX = CFG["execution"]
FEE = (1 - EX["slippage"]) * (1 - EX["commission_rate"] - EX["stamp_tax"]) \
      / ((1 + EX["slippage"]) * (1 + EX["commission_rate"]))
PANEL_DIR = Path("data/panel")
OUT = "dual_strategy.csv"
SEG_MAIN = 900
WARMUP = CFG["backtest"]["warmup_days"]

# momentum 信号执行的买入条件(与 dip 不同的执行口径): 无缺口过滤, 开盘直接买(高开≥9%放弃)
M_SKIP_GAIN = EX.get("skip_open_gain", 9.0)


def scan_segment(all_dates, s, e, cfg_dip, cfg_mom):
    lo = max(0, s - WARMUP)
    hi = min(len(all_dates), e + 2)
    dates = all_dates[lo:hi]
    main_range = all_dates[s:e]

    def read_one(d):
        return pd.read_csv(PANEL_DIR / f"{d}.csv.gz", compression="gzip",
                           dtype={"trade_date": str, "is_paused": "int8", "is_st": "int8"})
    with ThreadPoolExecutor(8) as x:
        frames = list(x.map(read_one, dates))
    panel = pd.concat(frames, ignore_index=True)
    panel = build_indicators(panel)
    by_date = split_by_date(panel)
    reg = regime_base(panel, CFG)
    reg["mkt_vol_20"] = reg["mkt_pct"].rolling(20).std()
    reg["m20"] = (reg["mkt_idx"] / reg["mkt_idx"].shift(20) - 1) * 100
    per_code = {}
    for code, sub in panel.groupby("ts_code", sort=False):
        per_code[code] = sub.set_index("trade_date").sort_index()
    del panel, frames
    pos_of = {d: i for i, d in enumerate(dates)}
    names = {}

    rows = []
    for d in main_range:
        day_df = by_date.get(d)
        if day_df is None or d not in pos_of:
            continue
        ip = pos_of[d]
        buy_d = dates[ip + 1] if ip + 1 < len(dates) else None
        sell_d = dates[ip + 2] if ip + 2 < len(dates) else None
        env = {}
        if d in reg.index:
            er = reg.loc[d]
            env = {c: er[c] for c in ["breadth", "mkt_vol_20", "m20"]}
            env["year"] = str(d)[:4]
        for mode_name, cfg in [("dip", cfg_dip), ("mom", cfg_mom)]:
            row = {"date": d, "mode": mode_name, **env}
            try:
                picks, _ = select_stocks(d, day_df, cfg, names, top_n=3)
            except Exception:
                picks = []
            row["n_cand"] = len(picks)
            row["top1_code"] = picks[0]["code"] if picks else ""
            row["top1_name"] = picks[0].get("name", "") if picks else ""
            row["top1_score"] = round(float(picks[0]["score"]), 3) if picks else np.nan
            ret = np.nan
            if picks and buy_d and sell_d:
                code = picks[0]["code"]
                sub = per_code.get(code)
                rb = sub.loc[buy_d] if sub is not None and buy_d in sub.index else None
                rs = sub.loc[sell_d] if sub is not None and sell_d in sub.index else None
                rp = day_df[day_df["ts_code"] == code]
                rp = rp.iloc[0] if len(rp) else None
                if rb is not None and rp is not None and rp["close_adj"] > 0:
                    gap = (rb["open_adj"] / rp["close_adj"] - 1) * 100
                    row["gap_pct"] = round(float(gap), 2)
                    if mode_name == "dip":
                        ok = (rb["is_paused"] == 0 and rb["vol"] > 0
                              and rb["open"] < rb["high_limit"] - 1e-9
                              and gap <= cfg["execution"]["buy_gap_below"]
                              and gap < M_SKIP_GAIN)
                    else:  # mom: 开盘直接买, 仅挡高开近一字
                        ok = (rb["is_paused"] == 0 and rb["vol"] > 0
                              and rb["open"] < rb["high_limit"] - 1e-9
                              and gap < M_SKIP_GAIN)
                    row["tradable"] = int(ok)
                    if ok and rs is not None and rs["is_paused"] == 0 and rs["vol"] > 0:
                        ret = (rs["open_adj"] / rb["open_adj"]) * FEE - 1
                else:
                    row["tradable"] = 0
            else:
                row["gap_pct"] = np.nan
                row["tradable"] = 0
            row["ret_raw_pct"] = round(float(ret) * 100, 3) if not np.isnan(ret) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    all_dates = sorted(p.name[:8] for p in PANEL_DIR.glob("*.csv.gz"))
    cfg_dip = copy.deepcopy(CFG)                       # dip: 当前定稿
    cfg_mom = copy.deepcopy(CFG)
    cfg_mom["selection"]["mode"] = "momentum"          # momentum: config 内置参数

    done = set()
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT, dtype={"date": str})
        done = set(prev["date"].unique())
        print(f"续扫: 已有 {len(done)} 天", flush=True)
    segs = []
    seg_start = WARMUP
    while seg_start < len(all_dates):
        segs.append((seg_start, min(seg_start + SEG_MAIN, len(all_dates))))
        seg_start += SEG_MAIN
    t0 = time.time()
    for si, (s, e) in enumerate(segs, 1):
        seg_dates = [d for d in all_dates[s:e] if d not in done]
        if not seg_dates:
            continue
        t1 = time.time()
        df = scan_segment(all_dates, s, e, cfg_dip, cfg_mom)
        df = df[~df["date"].isin(done)]
        header = not os.path.exists(OUT)
        df.to_csv(OUT, mode="a", header=header, index=False)
        done.update(df["date"])
        print(f"seg{si}/{len(segs)}: {len(df)}行 ({df['date'].iloc[0]}~{df['date'].iloc[-1]}) "
              f"{time.time()-t1:.0f}s 总{time.time()-t0:.0f}s", flush=True)
    print(f"ALL DONE: {len(done)} days", flush=True)


if __name__ == "__main__":
    main()
