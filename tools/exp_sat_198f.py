# -*- coding: utf-8 -*-
"""第 198 轮(第六批): 长周期趋势过滤 —— 只在中期趋势之上开新仓。

前五批的结论是: 所有"日内/短周期"旋钮都在噪声里(±1pp), 唯一没解释的窟窿是
**3000 日全史 -48.6% 的回撤全落在 2015-11~2016-06**(133 个交易日, 只有 11 笔交易)。
那段时间指数在 MA20 上下反复, 每次反弹都触发开仓、每次下跌都触发清仓 —— 典型绞肉机。

本批在**不清仓口径**的前提下, 给入场再加一层慢趋势条件(指数站上 MA60/MA120):
清仓仍按原 MA20 闸门, 只拦"熊市反弹里开的新仓"。
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
KEYS = ("总收益率%", "年化收益率%", "最大回撤%", "卡玛比率", "夏普比率",
        "交易次数", "胜率%", "平均单笔收益%", "空仓天数")


def main() -> None:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    cfg0 = load_config()
    cfg0["backtest"]["trading_days"] = days
    t0 = time.time()
    ctx = prepare_context(cfg0, days)
    print(f"[sat198f] 窗口 {len(ctx['main_dates'])} 日 "
          f"{ctx['main_dates'][0]}~{ctx['main_dates'][-1]} 准备 {time.time() - t0:.0f}s", flush=True)

    reg = ctx["reg"]
    idx = reg["mkt_idx"].astype(float)
    base = entry_ok_series(reg, cfg0)
    if base is None:
        b = float((cfg0.get("satellite") or {}).get("gate_entry_buffer") or 0.0)
        base = (idx >= reg["mkt_ma"] * (1 + b / 100.0))
        base = base & (reg["breadth"].astype(float) >= 0.25) & (~reg["panic"].astype(bool))
    base = base.fillna(False)

    ma60 = idx.rolling(60, min_periods=40).mean()
    ma120 = idx.rolling(120, min_periods=80).mean()
    ma250 = idx.rolling(250, min_periods=150).mean()
    print(f"   均线可用: MA60 {int(ma60.notna().sum())} / MA120 {int(ma120.notna().sum())} / "
          f"MA250 {int(ma250.notna().sum())} 天(共 {len(reg)})", flush=True)

    variants = [
        ("base", base, {}),
        ("xma60", base & (idx >= ma60), {}),
        ("xma120", base & (idx >= ma120), {}),
        ("xma250", base & (idx >= ma250), {}),
        ("xma120_mp4", base & (idx >= ma120), {"max_pos": 4, "buys_per_day": 4}),
    ]
    res: dict[str, dict] = {}
    for name, ok, ov in variants:
        cfg = copy.deepcopy(cfg0)
        cfg["satellite"].update(ov)
        ok = ok.fillna(False)
        t1 = time.time()
        r = run(ctx, cfg, entry_ok=ok)
        m = r["metrics"]
        res[name] = {k: m.get(k) for k in KEYS}
        res[name]["overrides"] = ov
        res[name]["entry_days"] = int(ok.sum())
        print(f"{name:11s} 年化 {m['年化收益率%']:7.2f}%  回撤 {m['最大回撤%']:7.2f}%  "
              f"卡玛 {m['卡玛比率']:5.2f}  笔数 {m['交易次数']:4d}  胜率 {m['胜率%']:5.2f}%  "
              f"单笔 {m['平均单笔收益%']:+6.3f}%  可入场 {int(ok.sum()):4d} 天  ({time.time() - t1:.0f}s)",
              flush=True)
        (OUT / f"exp_sat_198f_{days}.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")

    b = res["base"]
    print("\n=== 相对基线 ===", flush=True)
    for name, m in res.items():
        if name == "base":
            continue
        print(f"{name:11s} 年化 {m['年化收益率%'] - b['年化收益率%']:+7.2f}pp  "
              f"回撤 {m['最大回撤%'] - b['最大回撤%']:+7.2f}pp  "
              f"卡玛 {m['卡玛比率'] - b['卡玛比率']:+5.2f}  笔数 {m['交易次数'] - b['交易次数']:+d}",
              flush=True)
    print(f"[sat198f] 总耗时 {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("BT_FULL_TRADES", "1")
    main()
