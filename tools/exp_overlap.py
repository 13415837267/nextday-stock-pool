# -*- coding: utf-8 -*-
"""主/副策略的日历互补性 —— 第一百七十六轮。

用户要"每天有一只票可看"。两条腿各自只有 20~26% 的交易日有票, 关键是**重叠多少**:
  · 主有副无 / 副有主无 = 真正互补的部分;
  · 都有 / 都无 = 冗余与空窗。

本脚本不重跑资金曲线(那需要新的合并引擎), 只做"信号日历 + 单笔收益"的静态对齐 ——
能回答"互补有多大、补上的那部分是不是正期望", 但**不外推成合并组合的年化**。

用法: python3 tools/exp_overlap.py [--days 1500]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context          # noqa: E402
from src.config import load_config                # noqa: E402
import tools.satellite_backtest as sb             # noqa: E402


def _stat(vals):
    a = np.asarray(vals, dtype=float)
    if not len(a):
        return {"n": 0}
    return {"n": int(len(a)), "mean": round(float(a.mean()), 3),
            "median": round(float(np.median(a)), 3),
            "win": round(float((a > 0).mean() * 100), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=1500)
    args = ap.parse_args()
    cfg = load_config()
    print("构建上下文 ...", flush=True)
    ctx = prepare_context(cfg, args.days)
    dates = ctx["main_dates"]
    print(f"窗口 {dates[0]} ~ {dates[-1]} ({len(dates)} 日)", flush=True)

    # 主策略: 直接用已跑好的回测交易明细(含 gap_pct)
    bt = json.loads((ROOT / "output" / "backtest.json").read_text(encoding="utf-8"))
    mt = [t for t in bt.get("trades_all", []) if t.get("ret_pct") is not None]
    m_days = {str(t["buy_date"]).replace("-", "") for t in mt}

    r = sb.run(ctx, cfg)
    st = [t for t in r["trades"] if t.get("ret_pct") is not None]
    s_days = {str(t["entry_date"]).replace("-", "") for t in st}

    span = set(dates)
    m_days &= span
    s_days &= span
    both, m_only, s_only = m_days & s_days, m_days - s_days, s_days - m_days
    none = span - m_days - s_days
    n = len(span)

    def by_day(days):
        return [t["ret_pct"] for t in st if str(t["entry_date"]).replace("-", "") in days]

    def by_day_m(days):
        return [t["ret_pct"] for t in mt if str(t["buy_date"]).replace("-", "") in days]

    out = {
        "window": [dates[0], dates[-1]], "days": n,
        "cover": {
            "main_days": len(m_days), "sat_days": len(s_days),
            "both": len(both), "main_only": len(m_only), "sat_only": len(s_only),
            "none": len(none),
            "main_pct": round(len(m_days) / n * 100, 1),
            "sat_pct": round(len(s_days) / n * 100, 1),
            "union_pct": round((len(both) + len(m_only) + len(s_only)) / n * 100, 1),
        },
        "sat_on_main_empty": _stat(by_day(s_only)),
        "sat_on_main_active": _stat(by_day(both)),
        "sat_all": _stat([t["ret_pct"] for t in st]),
        "main_all": _stat([t["ret_pct"] for t in mt]),
        "main_in_both": _stat(by_day_m(both)),
    }
    print(json.dumps(out, ensure_ascii=False, indent=1), flush=True)
    p = ROOT / "output" / "exp_overlap.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("写出", p, flush=True)


if __name__ == "__main__":
    main()
