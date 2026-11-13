"""第226轮D · 卖出执行维度扫描（从未在本轮动过）

现行: T+2 收盘全卖(sell_open_frac=0), 唯一例外是开盘浮盈>=15% 全卖、开盘跌>=2% 提前砍。
未扫过的两个自由度:
  sell_open_frac  开盘卖出的比例(0=全在收盘, 1=全在开盘, 0.5=两条腿各一半)
  sell_close_if_open_drop  T+2 开盘跌幅达到该值即当日开盘砍掉
    (config 注释: 2021-24 日内正向、2025-26 负向 —— 押单一时点等于押结构不变)
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context, run_backtest  # noqa: E402
from src.config import load_config  # noqa: E402


def main() -> None:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    cfg = load_config()
    ctx = prepare_context(cfg, days)
    print(f"窗口 {len(ctx['main_dates'])} 日", flush=True)

    base = run_backtest(cfg, days, ctx)["metrics"]
    b_ann, b_cal = float(base["年化收益率%"]), float(base["卡玛比率"])
    print(f"基准 年化 {b_ann:.2f}% 卡玛 {b_cal:.2f}\n", flush=True)

    rows = []
    for key, vals in [("sell_open_frac", [0.0, 0.3, 0.5, 0.7, 1.0]),
                      ("sell_close_if_open_drop", [0.0, 1.0, 2.0, 3.0, 5.0])]:
        print(f"--- {key} (当前 {cfg['execution'].get(key)}) ---", flush=True)
        for v in vals:
            c = copy.deepcopy(cfg)
            c["execution"][key] = v
            m = run_backtest(c, days, ctx)["metrics"]
            ann, cal = float(m["年化收益率%"]), float(m["卡玛比率"])
            r = {"参数": key, "值": v, "年化": round(ann, 2),
                 "回撤": float(m["最大回撤%"]), "卡玛": round(cal, 2),
                 "笔数": int(m["交易次数"]), "胜率": float(m["胜率%"]),
                 "Δ年化": round(ann - b_ann, 2), "Δ卡玛": round(cal - b_cal, 2)}
            rows.append(r)
            tag = " ← 现行" if v == cfg["execution"].get(key) else (
                " ★" if r["Δ年化"] > 3 and r["Δ卡玛"] >= 0 else "")
            print(f"  {v:<5} 年化 {ann:7.2f}% ({r['Δ年化']:+6.2f}) 回撤 {r['回撤']:7.2f}% "
                  f"卡玛 {cal:5.2f} 笔 {r['笔数']:4d} 胜率 {r['胜率']:4.1f}%{tag}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "output" / "s226d_sell.csv", index=False, encoding="utf-8-sig")
    print("\n=== 按 Δ年化 排序 ===")
    print(df.sort_values("Δ年化", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
