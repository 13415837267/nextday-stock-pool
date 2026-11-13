"""第226轮G · 低波动空仓阈值扫描

依据: 全史 3.4 万笔低开样本按市场20日波动五分档, 单笔 -0.005% ~ +1.370%, 单调向上。
2023/2024/2026 波动偏低, 单笔 ≈ 0 甚至为负 —— 策略没有失效, 是波动环境不在。
新增 regime.vol_min(默认不配置=不启用), 扫描 {0.5, 0.6, 0.7, 0.8, 0.9}。
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
    print(f"基准 年化 {b_ann:.2f}% 回撤 {base['最大回撤%']}% 卡玛 {b_cal:.2f} "
          f"笔 {base['交易次数']}\n", flush=True)

    rows = []
    for v in [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        c = copy.deepcopy(cfg)
        if v > 0:
            c["regime"]["vol_min"] = v
        m = run_backtest(c, days, ctx)["metrics"]
        ann, cal = float(m["年化收益率%"]), float(m["卡玛比率"])
        r = {"vol_min": v, "年化": round(ann, 2), "回撤": float(m["最大回撤%"]),
             "卡玛": round(cal, 2), "笔数": int(m["交易次数"]),
             "胜率": float(m["胜率%"]), "Δ年化": round(ann - b_ann, 2),
             "Δ卡玛": round(cal - b_cal, 2)}
        rows.append(r)
        tag = " ← 现行" if v == 0 else ""
        print(f"  vol_min={v:<4} 年化 {ann:7.2f}% ({r['Δ年化']:+7.2f}) 回撤 {r['回撤']:7.2f}% "
              f"卡玛 {cal:5.2f} 笔 {r['笔数']:4d} 胜率 {r['胜率']:4.1f}%{tag}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "output" / "s226g_volmin.csv", index=False, encoding="utf-8-sig")
    print("\n" + df.to_string(index=False))


if __name__ == "__main__":
    main()
