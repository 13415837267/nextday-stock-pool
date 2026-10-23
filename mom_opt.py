# -*- coding: utf-8 -*-
"""momentum 参数 h1 粗扫(走前纪律: 仅用前半数据评估)。"""
import copy, os, sys, time
sys.path.insert(0, ".")
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from src.config import load_config
from src.selector import build_indicators, select_stocks, split_by_date

CFG = load_config()
EX = CFG["execution"]
FEE = (1 - EX["slippage"]) * (1 - EX["commission_rate"] - EX["stamp_tax"]) / ((1 + EX["slippage"]) * (1 + EX["commission_rate"]))
PANEL_DIR = Path("data/panel")
WARMUP = CFG["backtest"]["warmup_days"]

all_dates = sorted(p.name[:8] for p in PANEL_DIR.glob("*.csv.gz"))
H1_END = all_dates.index([d for d in all_dates if d >= "20190901"][0])  # h1 = 2012→2019-08
print(f"h1: {all_dates[WARMUP]}~{all_dates[H1_END-1]} ({H1_END-WARMUP} main days)")

def run_variant(name, sel_over, breadth_min=None):
    cfg = copy.deepcopy(CFG)
    cfg["selection"]["mode"] = "momentum"
    for k, v in sel_over.items():
        cfg["selection"][k] = v
    # 分3段扫描h1(内存)
    rets = []
    seg_bounds = [(WARMUP, WARMUP + 600), (WARMUP + 600, WARMUP + 1200), (WARMUP + 1200, H1_END)]
    for (s, e) in seg_bounds:
        if s >= e: continue
        lo, hi = max(0, s - WARMUP), min(len(all_dates), e + 2)
        dates = all_dates[lo:hi]
        def read_one(d):
            return pd.read_csv(PANEL_DIR / f"{d}.csv.gz", compression="gzip",
                               dtype={"trade_date": str, "is_paused": "int8", "is_st": "int8"})
        with ThreadPoolExecutor(8) as x:
            frames = list(x.map(read_one, dates))
        panel = pd.concat(frames, ignore_index=True)
        panel = build_indicators(panel)
        by_date = split_by_date(panel)
        reg = regime_base_ = panel.groupby("trade_date").agg(mkt_pct=("pct","mean")).sort_index()
        reg["breadth"] = panel[(panel["is_paused"]==0)&(panel["is_st"]==0)&(panel["bars"]>=21)].groupby("trade_date")["above_ma20"].mean()
        per_code = {}
        for code, sub in panel.groupby("ts_code", sort=False):
            per_code[code] = sub.set_index("trade_date").sort_index()
        del panel, frames
        pos_of = {d: i for i, d in enumerate(dates)}
        names = {}
        for d in dates[s-lo:e-lo]:
            day_df = by_date.get(d)
            if day_df is None: continue
            if breadth_min is not None:
                br = reg["breadth"].get(d)
                if br is None or br < breadth_min: continue
            ip = pos_of[d]
            buy_d = dates[ip+1] if ip+1 < len(dates) else None
            sell_d = dates[ip+2] if ip+2 < len(dates) else None
            if not buy_d or not sell_d: continue
            try: picks, _ = select_stocks(d, day_df, cfg, names, top_n=2)
            except Exception: continue
            if not picks: continue
            code = picks[0]["code"]
            sub = per_code.get(code)
            if sub is None: continue
            rb = sub.loc[buy_d] if buy_d in sub.index else None
            rs = sub.loc[sell_d] if sell_d in sub.index else None
            rp = day_df[day_df["ts_code"] == code]
            rp = rp.iloc[0] if len(rp) else None
            if rb is None or rp is None or rp["close_adj"] <= 0: continue
            gap = (rb["open_adj"] / rp["close_adj"] - 1) * 100
            if rb["is_paused"] == 1 or rb["vol"] <= 0 or rb["open"] >= rb["high_limit"] - 1e-9: continue
            if gap >= EX.get("skip_open_gain", 9.0): continue
            if rs is None or rs["is_paused"] == 1 or rs["vol"] <= 0: continue
            ret = (rs["open_adj"] / rb["open_adj"]) * FEE - 1
            rets.append((d, float(ret) * 100))
        del by_date, per_code
    if not rets:
        print(f"[{name}] h1: 0笔"); return 0
    r = pd.Series([x[1] for x in rets])
    print(f"[{name}] h1: {len(r)}笔 均值{r.mean():+.3f}% 中位{r.median():+.3f}% 胜率{(r>0).mean()*100:.1f}%", flush=True)
    return r.mean()

B = dict(pct_min=1.0, pct_max=7.0, vol_ratio_min=1.2, vol_ratio_max=5.0,
         ma5_slope_up=True, weights=dict(vol_ratio=0.35, mom3=0.25, mom20=0.20, amount=0.10, neg_bias20=0.10))
t0 = time.time()
run_variant("base", {})
run_variant("strong_pct", dict(pct_min=3.0))
run_variant("limit_pct", dict(pct_min=5.0, pct_max=9.5))
run_variant("vol2", dict(vol_ratio_min=2.0))
run_variant("mom3up", dict(weights=dict(vol_ratio=0.25, mom3=0.40, mom20=0.20, amount=0.05, neg_bias20=0.10)))
run_variant("base_breadth44", dict(), breadth_min=0.44)
run_variant("strong_vol2", dict(pct_min=3.0, vol_ratio_min=2.0))
print(f"done {time.time()-t0:.0f}s")
