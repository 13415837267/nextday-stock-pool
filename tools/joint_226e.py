"""第226轮E · 选股条件联合优化（训练段搜索 / 验证段确认）

为什么必须联合而不是单参数: 单参数扫描在 355 笔上的置信区间本身就是几十个百分点,
测出来的 ±2pp 全是噪声(price_max 1500日+6.5/750日-1.9 就是例子)。
但参数之间有交互 —— 放宽 dip_pct 区间后, 量比上限的最优点会移动。

协议:
  同一段 1500 日的净值切成前 750 / 后 750 两段;
  随机搜索 6 维(40 组), 目标=前段年化; 逐组记录后段年化与卡玛;
  只有当**前段优于基准 且 后段也优于基准 且 后段卡玛不降**时才进入候选池。
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context, run_backtest  # noqa: E402
from src.config import load_config  # noqa: E402


def seg_metrics(res: dict, lo: float, hi: float) -> dict:
    eq = pd.DataFrame(res["equity"])
    n = len(eq)
    seg = eq.iloc[int(n * lo):int(n * hi)]
    if len(seg) < 60:
        return {}
    nav = seg["nav"].values / seg["nav"].values[0]
    peak, mdd = 1.0, 0.0
    for v in nav:
        peak = max(peak, v)
        mdd = min(mdd, v / peak - 1)
    yrs = max(0.4, (pd.Timestamp(seg["d"].iloc[-1]) - pd.Timestamp(seg["d"].iloc[0])).days
              / 365.25)
    ann = nav[-1] ** (1 / yrs) - 1
    return {"ann": ann * 100, "mdd": mdd * 100,
            "calmar": ann / abs(mdd) if mdd else float("nan")}


def main() -> None:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    n_try = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    cfg = load_config()
    ctx = prepare_context(cfg, days)
    print(f"上下文 {len(ctx['main_dates'])} 日", flush=True)

    def eval_cfg(c):
        r = run_backtest(c, days, ctx)
        return seg_metrics(r, 0.0, 0.5), seg_metrics(r, 0.5, 1.0)

    b1, b2 = eval_cfg(cfg)
    print(f"基准  前段 年化 {b1['ann']:6.2f}% 卡玛 {b1['calmar']:5.2f} | "
          f"后段 年化 {b2['ann']:6.2f}% 卡玛 {b2['calmar']:5.2f}", flush=True)

    rng = np.random.default_rng(22655)
    rows = []
    t0 = time.time()
    for i in range(n_try):
        c = copy.deepcopy(cfg)
        pmin = float(rng.uniform(-8.0, -4.0))
        pmax = float(rng.uniform(-4.5, -1.0))
        if pmax <= pmin + 0.5:
            pmax = pmin + 1.0
        c["selection"]["dip_pct_min"] = round(pmin, 2)
        c["selection"]["dip_pct_max"] = round(pmax, 2)
        c["selection"]["dip_vol_ratio_max"] = round(float(rng.uniform(1.5, 8.0)), 2)
        c["selection"]["dip_mom20_min"] = round(float(rng.uniform(0.0, 12.0)), 1)
        c["selection"]["dip_mom20_max"] = round(float(rng.uniform(15.0, 45.0)), 1)
        c["universe"]["amount_min_yi"] = round(float(rng.uniform(0.4, 2.0)), 2)
        try:
            s1, s2 = eval_cfg(c)
        except Exception as e:  # noqa: BLE001
            print(f"  {i+1}: FAIL {e}", flush=True)
            continue
        if not s1 or not s2:
            continue
        ok = (s1["ann"] > b1["ann"] and s2["ann"] > b2["ann"]
              and s2["calmar"] >= b2["calmar"])
        rows.append({
            "pct_min": c["selection"]["dip_pct_min"], "pct_max": c["selection"]["dip_pct_max"],
            "vr_max": c["selection"]["dip_vol_ratio_max"],
            "m20_min": c["selection"]["dip_mom20_min"], "m20_max": c["selection"]["dip_mom20_max"],
            "amt": c["universe"]["amount_min_yi"],
            "前段年化": round(s1["ann"], 2), "前段卡玛": round(s1["calmar"], 2),
            "后段年化": round(s2["ann"], 2), "后段卡玛": round(s2["calmar"], 2),
            "两段皆赢": bool(ok)})
        if ok or (i + 1) % 10 == 0:
            print(f"  {i+1:2d}/{n_try} 前 {s1['ann']:6.1f}%/{s1['calmar']:4.2f} "
                  f"后 {s2['ann']:6.1f}%/{s2['calmar']:4.2f} "
                  f"pct[{pmin:.1f},{pmax:.1f}] vr{c['selection']['dip_vol_ratio_max']:.1f} "
                  f"amt{c['universe']['amount_min_yi']:.2f} {'✓' if ok else ''}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "output" / "s226e_joint.csv", index=False, encoding="utf-8-sig")
    print(f"\n耗时 {time.time()-t0:.0f}s, 两段皆赢 {int(df['两段皆赢'].sum())}/{len(df)} 组")
    if df["两段皆赢"].any():
        print(df[df["两段皆赢"]].to_string(index=False))
        best = df[df["两段皆赢"]].sort_values("后段卡玛", ascending=False).iloc[0]
        json.dump(best.to_dict(), open(ROOT / "output" / "s226e_best.json", "w"),
                  ensure_ascii=False, indent=1, default=float)
    else:
        print("\n无任何组合在验证段同时赢过年化与卡玛。")


if __name__ == "__main__":
    main()
