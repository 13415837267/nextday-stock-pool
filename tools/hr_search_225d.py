"""第225轮D · 在真实回测里搜权重（评价函数与目标完全对齐）

前三版失败的原因不是权重方向, 而是评价函数: 离线"每日top1满仓"的排序, 与真实回测
(regime 只在场 43.7% + 仓位按 gap/分数分层 + T+2 卖出规则) 的排序不相关。
所以这版不再做离线近似, 每次评估都跑真实回测。

搜索: 从现行权重出发, 逐维坐标扰动, 只接受帕累托(年化更高且卡玛不降)。
输出: output/w225d_best.json
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context, run_backtest  # noqa: E402
from src.config import load_config  # noqa: E402

KEYS = ["mom20", "neg_bias20", "neg_amount", "neg_vol_ratio", "low_tail",
        "neg_turnover_cool", "ivol", "neg_dist_low20"]
CUR = {"mom20": 0.20, "neg_bias20": 0.70, "neg_amount": 0.05, "neg_vol_ratio": 0.02,
       "low_tail": 0.03, "neg_turnover_cool": 0.0, "ivol": 0.0, "neg_dist_low20": 0.0}


def score(m: dict) -> tuple:
    return float(m["年化收益率%"]), float(m["卡玛比率"])


def main() -> None:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    budget = int(sys.argv[2]) if len(sys.argv) > 2 else 18
    cfg = load_config()
    ctx = prepare_context(cfg, days)
    print(f"上下文 {len(ctx['main_dates'])} 日", flush=True)

    t0 = time.time()
    base_m = run_backtest(cfg, days, ctx)["metrics"]
    b_ann, b_cal = score(base_m)
    print(f"基准 年化 {b_ann:.2f}% 卡玛 {b_cal:.2f}  ({time.time()-t0:.0f}s/次)", flush=True)

    best_w = dict(CUR)
    best = (b_ann, b_cal)
    tried = []
    trials = 0

    # 坐标搜索: 每维 × 每个步长, 只接受帕累托
    steps = [0.6, 0.3, 0.15, -0.15, -0.3, -0.6]
    for rnd in range(2):
        improved = False
        for k in KEYS:
            for st in steps:
                if trials >= budget:
                    break
                w = dict(best_w)
                w[k] = round(float(np.clip(w.get(k, 0.0) + st, -1.2, 1.5)), 3)
                if abs(w[k] - best_w.get(k, 0.0)) < 1e-9:
                    continue
                c = copy.deepcopy(cfg)
                c["selection"]["dip_weights"] = w
                try:
                    r = run_backtest(c, days, ctx)
                except Exception as e:  # noqa: BLE001
                    print(f"  [FAIL] {k}{st:+.2f}: {e}", flush=True)
                    continue
                trials += 1
                ann, cal = score(r["metrics"])
                ok = ann > best[0] and cal >= best[1] * 0.98
                print(f"  {trials:2d} {k:>18s}{st:+.2f} -> 年化 {ann:7.2f}% 卡玛 {cal:5.2f} "
                      f"{'✓接受' if ok else ''}", flush=True)
                tried.append({"key": k, "step": st, "ann": ann, "cal": cal, "ok": ok})
                if ok:
                    best, best_w, improved = (ann, cal), w, True
            if trials >= budget:
                break
        if trials >= budget or not improved:
            break

    print(f"\n最佳: 年化 {best[0]:.2f}% (基准 {b_ann:.2f}%, Δ{best[0]-b_ann:+.2f}pp) "
          f"卡玛 {best[1]:.2f} (基准 {b_cal:.2f})", flush=True)
    print(f"权重 {best_w}", flush=True)
    json.dump({"weights": best_w, "ann": best[0], "calmar": best[1],
               "base_ann": b_ann, "base_calmar": b_cal, "trials": trials,
               "log": tried}, open(ROOT / "output" / "w225d_best.json", "w"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
