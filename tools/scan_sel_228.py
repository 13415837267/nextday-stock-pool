"""第228轮 · 选股条件重扫（新成本口径：买0/卖0）

为什么必须重扫：
  226 轮的选股网格是在"千1 双边滑点"下做的，那时单笔 +1.435%、每笔往返成本约 0.26%。
  本轮成本口径改为双端滑点 0（机制上成立），单笔升到 +1.647%、往返成本降到约 0.06%。
  **成本占比变了 4 倍，最优选股条件会整体移动** —— 高成本时"少而准"更优，
  低成本时"多而宽"可能更优。这是重扫的正当理由，不是重复劳动。

方法：单维扫描（一次只动一个参数），1500 日，与基准逐位对照。
  只跑与当前值不同的档位，减少无效回测。
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

OUT = ROOT / "output" / "w228_sel.csv"

# 第一批：回调形态（决定买什么样的回调）
GRID_A = [
    ("dip_pct_min", "selection", "dip_pct_min", [-7.5, -5.0, -4.0]),
    ("dip_pct_max", "selection", "dip_pct_max", [-4.5, -2.0, -1.0]),
    ("dip_vol_ratio_max", "selection", "dip_vol_ratio_max", [1.5, 2.5, 6.0]),
    ("dip_mom20_min", "selection", "dip_mom20_min", [0.0, 12.0]),
    ("dip_mom20_max", "selection", "dip_mom20_max", [15.0, 45.0]),
    ("dip_dist_low20_min", "selection", "dip_dist_low20_min", [0.0, 3.0, 12.0]),
]
# 第二批：安全边际 + 股票池门槛
GRID_B = [
    ("bias20_max", "selection", "bias20_max", [7.0, 16.0]),
    ("dip_close_pos_max", "selection", "dip_close_pos_max", [0.2, 0.5]),
    ("amount_min_yi", "universe", "amount_min_yi", [0.4, 1.5, 3.0]),
    ("turnover_min", "universe", "turnover_min", [1.0, 4.0]),
    ("turnover_max", "universe", "turnover_max", [8.0]),
]


def run_one(cfg, days, ctx) -> dict:
    m = run_backtest(cfg, days, ctx)["metrics"]
    return {"年化": round(float(m["年化收益率%"]), 2),
            "回撤": round(float(m["最大回撤%"]), 2),
            "卡玛": round(float(m["卡玛比率"] or 0), 2),
            "笔数": int(m["交易次数"]),
            "胜率": round(float(m["胜率%"]), 1),
            "单笔": round(float(m["平均单笔收益%"]), 3)}


# 第三批：把 1500 日上"卡玛改善"或"跌幅最小"的关键值, 放到 2500 日复验
GRID_C = [
    ("dip_pct_min", "selection", "dip_pct_min", [-7.5]),
    ("bias20_max", "selection", "bias20_max", [16.0]),
    ("amount_min_yi", "universe", "amount_min_yi", [1.5]),
    ("turnover_min", "universe", "turnover_min", [1.0]),
    ("dip_vol_ratio_max", "selection", "dip_vol_ratio_max", [2.5]),
    ("dip_close_pos_max", "selection", "dip_close_pos_max", [0.5]),
]


def main() -> None:
    batch = sys.argv[1] if len(sys.argv) > 1 else "A"
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
    grid = {"A": GRID_A, "B": GRID_B, "C": GRID_C}.get(batch.upper(), GRID_A)

    cfg = load_config()
    t0 = time.time()
    ctx = prepare_context(cfg, days)
    print(f"ctx {time.time()-t0:.0f}s, {days} 日, 批次 {batch}\n", flush=True)

    base = run_one(cfg, days, ctx)
    print(f"基准(现行)  年化{base['年化']:7.2f}% 回撤{base['回撤']:7.2f}% 卡玛{base['卡玛']:5.2f} "
          f"{base['笔数']}笔 单笔{base['单笔']:.3f}%\n", flush=True)

    rows = []
    for name, sect, key, vals in grid:
        cur = cfg[sect][key]
        print(f"--- {name} (当前 {cur}) ---", flush=True)
        for v in vals:
            if isinstance(cur, (int, float)) and abs(float(v) - float(cur)) < 1e-9:
                continue
            c = copy.deepcopy(cfg)
            c[sect][key] = type(cur)(v) if not isinstance(cur, bool) else v
            t = time.time()
            r = run_one(c, days, ctx)
            d_ann = r["年化"] - base["年化"]
            rows.append({"参数": name, "取值": v, "当前": cur, **r,
                         "Δ年化": round(d_ann, 2), "Δ卡玛": round(r["卡玛"] - base["卡玛"], 2),
                         "Δ笔数": r["笔数"] - base["笔数"]})
            pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8-sig")
            flag = "★" if d_ann > 0 and r["卡玛"] >= base["卡玛"] else " "
            print(f"  {v:<8} 年化{r['年化']:7.2f}% ({d_ann:+6.2f}) 回撤{r['回撤']:7.2f}% "
                  f"卡玛{r['卡玛']:5.2f} ({r['卡玛']-base['卡玛']:+.2f}) "
                  f"{r['笔数']:4d}笔 单笔{r['单笔']:.3f}% {flag} {time.time()-t:.0f}s", flush=True)

    d = pd.DataFrame(rows)
    print("\n=== 双改善(年化↑ 且 卡玛↑) ===")
    w = d[(d["Δ年化"] > 0) & (d["Δ卡玛"] >= 0)].sort_values("Δ年化", ascending=False)
    print(w[["参数", "取值", "年化", "卡玛", "笔数", "单笔", "Δ年化", "Δ卡玛"]].to_string(index=False)
          if len(w) else "(无)")


if __name__ == "__main__":
    main()
