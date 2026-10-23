# -*- coding: utf-8 -*-
"""分段全周期信号-环境扫描: 每900主日一段(带160预热), 规避8GB内存限制。
输出 regime_signals.csv (追加式, 断点续扫)。"""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd

from src.config import load_config
from src.selector import build_indicators, regime_base, select_stocks, split_by_date

CFG = load_config()
EX = CFG["execution"]
FEE = (1 - EX["slippage"]) * (1 - EX["commission_rate"] - EX["stamp_tax"]) \
      / ((1 + EX["slippage"]) * (1 + EX["commission_rate"]))
FLOOR = EX["buy_gap_below"]
SKIP_GAIN = EX.get("skip_open_gain", 9.0)
PANEL_DIR = Path("data/panel")
OUT = "regime_signals.csv"
SEG_MAIN = 900    # 每段主日数
WARMUP = CFG["backtest"]["warmup_days"]


def build_env(reg_base: pd.DataFrame) -> pd.DataFrame:
    env = reg_base.copy()
    for w in [20, 60, 120, 250]:
        env[f"ma_ratio_{w}"] = (env["mkt_idx"] / env["mkt_idx"].rolling(w).mean() - 1) * 100
    env["breadth_ma10"] = env["breadth"].rolling(10).mean()
    env["mkt_ret_20"] = (env["mkt_idx"] / env["mkt_idx"].shift(20) - 1) * 100
    env["mkt_vol_20"] = env["mkt_pct"].rolling(20).std()
    env["dist_250h"] = (env["mkt_idx"] / env["mkt_idx"].rolling(250).max() - 1) * 100
    env["year"] = env.index.astype(str).str[:4]
    env["month"] = env.index.astype(str).str[:6]
    return env


def scan_segment(all_dates, s, e):
    """扫描主日区间 [s, e) 的日期, 读入 [s - warmup, e + 2] 的数据构建上下文。"""
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
    reg = build_env(regime_base(panel, CFG))
    per_code = {}
    for code, sub in panel.groupby("ts_code", sort=False):
        per_code[code] = sub.set_index("trade_date").sort_index()
    del panel, frames
    pos_of = {d: i for i, d in enumerate(dates)}
    cfg = CFG.copy()
    names = {}

    rows = []
    for d in main_range:
        day_df = by_date.get(d)
        if day_df is None or d not in pos_of:
            continue
        ip = pos_of[d]
        buy_d = dates[ip + 1] if ip + 1 < len(dates) else None
        sell_d = dates[ip + 2] if ip + 2 < len(dates) else None
        try:
            picks, _ = select_stocks(d, day_df, cfg, names, top_n=5)
        except Exception:
            picks = []
        row = {"date": d, "n_cand": len(picks),
               "top1_code": picks[0]["code"] if picks else "",
               "top1_name": picks[0].get("name", "") if picks else "",
               "top1_score": round(float(picks[0]["score"]), 3) if picks else np.nan,
               "top2_code": picks[1]["code"] if len(picks) > 1 else "",
               "top2_score": round(float(picks[1]["score"]), 3) if len(picks) > 1 else np.nan}
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
                tradable = (rb["is_paused"] == 0 and rb["vol"] > 0
                            and rb["open"] < rb["high_limit"] - 1e-9
                            and gap <= FLOOR and gap < SKIP_GAIN)
                row["tradable"] = int(tradable)
                if tradable and rs is not None and rs["is_paused"] == 0 and rs["vol"] > 0:
                    ret = (rs["open_adj"] / rb["open_adj"]) * FEE - 1
                    row["ret_raw_pct"] = round(float(ret) * 100, 3)
                else:
                    row["ret_raw_pct"] = np.nan
            else:
                row["gap_pct"] = np.nan; row["tradable"] = 0; row["ret_raw_pct"] = np.nan
        else:
            row["gap_pct"] = np.nan; row["tradable"] = 0; row["ret_raw_pct"] = np.nan
        if d in reg.index:
            er = reg.loc[d]
            for c in ["breadth", "ma_ratio_20", "ma_ratio_60", "ma_ratio_120", "ma_ratio_250",
                      "breadth_ma10", "mkt_ret_20", "mkt_vol_20", "dist_250h", "year", "month"]:
                row[c] = er[c]
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    all_dates = sorted(p.name[:8] for p in PANEL_DIR.glob("*.csv.gz"))
    done = set()
    if os.path.exists(OUT):
        done = set(pd.read_csv(OUT, dtype={"date": str})["date"])
        print(f"续扫: 已有 {len(done)} 天", flush=True)
    main_dates = [d for d in all_dates[WARMUP:] if d not in done]
    idx_of = {d: i for i, d in enumerate(all_dates)}
    # 主日区间(按 all_dates 索引切段)
    segs = []
    seg_start = WARMUP
    while seg_start < len(all_dates):
        segs.append((seg_start, min(seg_start + SEG_MAIN, len(all_dates))))
        seg_start += SEG_MAIN
    t0 = time.time()
    for si, (s, e) in enumerate(segs, 1):
        seg_dates = [d for d in all_dates[s:e] if d not in done]
        if not seg_dates:
            print(f"seg{si} 全部已完成, 跳过", flush=True)
            continue
        t1 = time.time()
        df = scan_segment(all_dates, s, e)
        df = df[~df["date"].isin(done)]
        header = not os.path.exists(OUT)
        df.to_csv(OUT, mode="a", header=header, index=False)
        done.update(df["date"])
        print(f"seg{si}/{len(segs)}: {len(df)}天 ({df['date'].iloc[0]}~{df['date'].iloc[-1]}) "
              f"耗时{time.time()-t1:.0f}s 总{time.time()-t0:.0f}s", flush=True)
        del df
    print(f"ALL SEGS DONE: {len(done)} days, {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
