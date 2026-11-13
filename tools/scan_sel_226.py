"""第226轮 · 选股条件网格扫描（只看年化/卡玛，不做其他）

已经确认到顶的: regime(64组)、排序权重(4类目标)、仓位折减、持仓只数、卖出顺延。
这一层从未系统扫过: 决定"哪只票进入候选池"的硬阈值本身。
   dip_pct 区间 / vol_ratio 上限 / mom20 区间 / dist_low20 下限 / bias20 上限
   universe: 成交额下限 / 换手区间 / 价格上限
方法: 单维扫描(一次一个参数), 1500 日粗筛, 只对改善的做 750 日复核。
"""
from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context, run_backtest  # noqa: E402
from src.config import load_config  # noqa: E402

# (分组名, 路径, 参数名, 候选值)
GRID = [
    ("dip_pct_min", "selection", "dip_pct_min", [-7.5, -6.0, -5.0, -4.0]),
    ("dip_pct_max", "selection", "dip_pct_max", [-4.5, -3.0, -2.0, -1.0]),
    ("vol_ratio_max", "selection", "dip_vol_ratio_max", [1.5, 2.5, 4.0, 6.0]),
    ("mom20_min", "selection", "dip_mom20_min", [0.0, 5.0, 12.0]),
    ("mom20_max", "selection", "dip_mom20_max", [15.0, 25.0, 45.0]),
    ("dist_low20_min", "selection", "dip_dist_low20_min", [0.0, 3.0, 6.0, 12.0]),
    ("bias20_max", "selection", "bias20_max", [7.0, 11.4, 16.0]),
    ("close_pos_max", "selection", "dip_close_pos_max", [0.2, 0.32, 0.5]),
    ("amt_min_yi", "universe", "amount_min_yi", [0.4, 0.8, 1.5, 3.0]),
    ("to_min", "universe", "turnover_min", [1.0, 2.0, 4.0]),
    ("to_max", "universe", "turnover_max", [8.0, 12.0, 20.0]),
    ("price_max", "universe", "price_max", [50.0, 100.0, 150.0, 300.0]),
]


def run_one(cfg, days, ctx) -> dict:
    m = run_backtest(cfg, days, ctx)["metrics"]
    return {"年化": float(m["年化收益率%"]), "回撤": float(m["最大回撤%"]),
            "卡玛": float(m["卡玛比率"]), "笔数": int(m["交易次数"]),
            "胜率": float(m["胜率%"])}


def main() -> None:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    cfg = load_config()
    t0 = time.time()
    ctx = prepare_context(cfg, days)
    print(f"上下文 {len(ctx['main_dates'])} 日, {time.time()-t0:.0f}s", flush=True)

    base = run_one(cfg, days, ctx)
    print(f"\n基准 {base}", flush=True)

    rows = []
    for name, sect, key, vals in GRID:
        print(f"\n--- {name} (当前 {cfg[sect][key]}) ---", flush=True)
        for v in vals:
            if v == cfg[sect].get(key):
                continue
            c = copy.deepcopy(cfg)
            c[sect][key] = v
            try:
                r = run_one(c, days, ctx)
            except Exception as e:  # noqa: BLE001
                print(f"  {v}: FAIL {e}", flush=True)
                continue
            d = r["年化"] - base["年化"]
            flag = "★" if d > 3 and r["卡玛"] >= base["卡玛"] else ""
            print(f"  {key}={v:<7} 年化 {r['年化']:7.2f}% ({d:+6.2f}) 回撤 {r['回撤']:7.2f}% "
                  f"卡玛 {r['卡玛']:5.2f} 笔 {r['笔数']:4d} {flag}", flush=True)
            rows.append({"组": name, "值": v, **r, "Δ年化": round(d, 2),
                         "Δ卡玛": round(r["卡玛"] - base["卡玛"], 2)})

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "output" / "s226_grid.csv", index=False, encoding="utf-8-sig")
    print("\n=== 全部按 Δ年化 排序 top15 ===")
    print(df.sort_values("Δ年化", ascending=False).head(15).to_string(index=False))


if __name__ == "__main__":
    main()
