"""第226轮F · 持有期结构扫描（改变收益结构，不是调参）

现行: T+1 开盘买 → T+2 收盘卖, 固定持有 1 个交易日。
分年数据显示 2021-22 是异常大年(190%/243%), 2023-26 均值落到 74.4%,
说明"隔日反转"这一结构在近期被削弱。持有期是唯一还没系统扫过的结构维度:
  hold_extra_days: T+2 开盘后继续持有的天数(0=现行)
  hold_if_open_gain: T+2 开盘浮盈达到该值才顺延(0=一律顺延)
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
    print(f"基准 年化 {b_ann:.2f}% 回撤 {base['最大回撤%']}% 卡玛 {b_cal:.2f}\n", flush=True)

    rows = []
    for extra in (0, 1, 2, 3, 5):
        for gate in (0.0, 3.0, 5.0):
            if extra == 0 and gate != 0.0:
                continue
            c = copy.deepcopy(cfg)
            c["execution"]["hold_extra_days"] = extra
            c["execution"]["hold_if_open_gain"] = gate
            m = run_backtest(c, days, ctx)["metrics"]
            ann, cal = float(m["年化收益率%"]), float(m["卡玛比率"])
            r = {"顺延天数": extra, "浮盈门槛": gate, "年化": round(ann, 2),
                 "回撤": float(m["最大回撤%"]), "卡玛": round(cal, 2),
                 "笔数": int(m["交易次数"]), "胜率": float(m["胜率%"]),
                 "Δ年化": round(ann - b_ann, 2), "Δ卡玛": round(cal - b_cal, 2)}
            rows.append(r)
            print(f"  顺延{extra}天 门槛{gate:<4} 年化 {ann:7.2f}% ({r['Δ年化']:+7.2f}) "
                  f"回撤 {r['回撤']:7.2f}% 卡玛 {cal:5.2f} 笔 {r['笔数']:4d} "
                  f"胜率 {r['胜率']:4.1f}%", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "output" / "s226f_hold.csv", index=False, encoding="utf-8-sig")
    print("\n=== 按 Δ年化 ===")
    print(df.sort_values("Δ年化", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
