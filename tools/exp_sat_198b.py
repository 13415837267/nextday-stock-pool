# -*- coding: utf-8 -*-
"""第 198 轮(第二批): 副策略「持仓层闸门」到底是在救火还是在割肉。

第一批(exp_sat_198.py)的双窗口结论:
  · 加并发只数(mp4) 只在近六年改善卡玛(2.04→2.34), 全史反而 0.62→0.57;
  · 持有期、入场缓冲、限价深度、池宽全部维持原值;
  · 固定止损(-8/-15)与基准逐位相同 —— 闸门总是先砍, 止损根本轮不到。

而逐笔统计给了新线索: 1500 日 106 笔里 **61 笔是"大盘转弱清仓", 均 +0.38%**,
真正持满的 35 笔均 **+7.02%**。也就是说闸门把大量本来会跑完 10 天的仓位提前卖掉了。
本批只动这一处: 清仓口径(all / loser / 关)、闸门均线(MA20 / MA60)、以及和并发只数的组合。
"""
from __future__ import annotations

import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context          # noqa: E402
from src.config import load_config               # noqa: E402
from src.satellite import entry_ok_series        # noqa: E402
from tools.satellite_backtest import run         # noqa: E402

OUT = ROOT / "output"

VARIANTS: list[tuple[str, dict, int]] = [
    # (名字, satellite 覆盖, 闸门均线周期; 0 = 用 config 里的 regime 原值)
    ("base",        {}, 0),
    ("gxloser",     {"gate_exit_mode": "loser"}, 0),
    ("gxoff",       {"gate_exit": False}, 0),
    ("ma60",        {}, 60),
    ("ma60_loser",  {"gate_exit_mode": "loser"}, 60),
    ("mp4_loser",   {"max_pos": 4, "buys_per_day": 4, "gate_exit_mode": "loser"}, 0),
    ("mp4_ma60",    {"max_pos": 4, "buys_per_day": 4}, 60),
    ("mp4_pb2",     {"max_pos": 4, "buys_per_day": 4, "pullback_pct": 2.0}, 0),
    ("mp4_b2",      {"max_pos": 4, "buys_per_day": 2}, 0),
    ("gxloser_pb2", {"gate_exit_mode": "loser", "pullback_pct": 2.0}, 0),
]

KEYS = ("总收益率%", "年化收益率%", "最大回撤%", "卡玛比率", "夏普比率",
        "交易次数", "胜率%", "平均单笔收益%", "空仓天数")


def gate_ctx(ctx: dict, period: int) -> dict:
    """把 regime 的闸门均线换成 period 日均线, 并重算 trade_ok(与引擎同一套条件)。"""
    if period <= 0 or ctx.get("reg") is None:
        return ctx
    reg = ctx["reg"].copy()
    idx = reg["mkt_idx"].astype(float)
    reg["mkt_ma"] = idx.rolling(period, min_periods=period).mean()
    ok = reg["mkt_idx"] >= reg["mkt_ma"]
    bmin = 0.25
    if "breadth" in reg.columns:
        ok = ok & (reg["breadth"] >= bmin)
    if "panic" in reg.columns:
        ok = ok & (~reg["panic"].astype(bool))
    reg["trade_ok"] = ok.fillna(False)
    out = dict(ctx)
    out["reg"] = reg
    return out


def main() -> None:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    cfg0 = load_config()
    cfg0["backtest"]["trading_days"] = days
    t0 = time.time()
    ctx0 = prepare_context(cfg0, days)
    print(f"[sat198b] 窗口 {len(ctx0['main_dates'])} 日 "
          f"{ctx0['main_dates'][0]}~{ctx0['main_dates'][-1]} 准备 {time.time() - t0:.0f}s", flush=True)

    res: dict[str, dict] = {}
    for name, ov, period in VARIANTS:
        cfg = copy.deepcopy(cfg0)
        cfg["satellite"].update(ov)
        cfg["backtest"]["trading_days"] = days
        ctx = gate_ctx(ctx0, period)
        _entry = entry_ok_series(ctx.get("reg"), cfg)
        t1 = time.time()
        r = run(ctx, cfg, entry_ok=_entry)
        m = r["metrics"]
        res[name] = {k: m.get(k) for k in KEYS}
        res[name]["overrides"] = ov
        res[name]["gate_ma"] = period or 20
        # 退出原因分布: 直接看闸门是不是在割肉
        from collections import Counter
        c = Counter(t["note"].split(" ")[0] for t in r["trades"])
        agg: dict[str, list] = {}
        for t in r["trades"]:
            agg.setdefault(t["note"].split(" ")[0], []).append(t["ret_pct"])
        res[name]["exits"] = {k: [len(v), round(float(np.mean(v)), 2)] for k, v in agg.items()}
        print(f"{name:12s} 年化 {m['年化收益率%']:7.2f}%  回撤 {m['最大回撤%']:7.2f}%  "
              f"卡玛 {m['卡玛比率']:5.2f}  笔数 {m['交易次数']:4d}  胜率 {m['胜率%']:5.2f}%  "
              f"单笔 {m['平均单笔收益%']:+6.3f}%  {dict(c.most_common(4))}  ({time.time() - t1:.0f}s)",
              flush=True)
        (OUT / f"exp_sat_198b_{days}.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")

    b = res["base"]
    print("\n=== 相对基线 ===", flush=True)
    for name, m in res.items():
        if name == "base":
            continue
        print(f"{name:12s} 年化 {m['年化收益率%'] - b['年化收益率%']:+7.2f}pp  "
              f"回撤 {m['最大回撤%'] - b['最大回撤%']:+7.2f}pp  "
              f"卡玛 {m['卡玛比率'] - b['卡玛比率']:+5.2f}", flush=True)
    print(f"[sat198b] 总耗时 {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("BT_FULL_TRADES", "1")
    main()
