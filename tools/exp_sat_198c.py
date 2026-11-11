# -*- coding: utf-8 -*-
"""第 198 轮(第三批): 只动「入场」不改「清仓」的四个过滤器。

第二批证明了一件反直觉的事: 持仓层闸门**不是在割肉** —— 关掉它(gxoff)近六年年化
从 43.69% 掉到 24.21%、回撤从 -21.45% 扩到 -36.31%; 换成 MA60(ma60) 直接 -4.71%。
被闸门提前卖掉的 61 笔均 +0.38%, 但留下来的那 10 天里跌得更多。
所以闸门是对的, 只能从"什么条件下允许开新仓"这一侧再找增量。

本批全部通过 entry_ok 单独注入(只过滤入场), reg.trade_ok 保持原样, 清仓口径不变:
  · 宽度地板 0.30 / 0.35 (config 现值 0.25)
  · 指数 20 日年化波动 > 25% / 30% 时不开新仓
  · 指数须在 60 日高点的 95% 以内
  · 池子最少 30 只(min_pool=20)
  · 跳空上限 3%(现值 5%) —— 这一条走 entry_gate, 不是 entry_ok
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
from tools.satellite_backtest import run         # noqa: E402

OUT = ROOT / "output"

KEYS = ("总收益率%", "年化收益率%", "最大回撤%", "卡玛比率", "夏普比率",
        "交易次数", "胜率%", "平均单笔收益%", "空仓天数")


def base_entry(reg: pd.DataFrame, cfg: dict) -> pd.Series:
    """现行入场闸门: 指数 >= MA20×(1+0.3%) 且 宽度 >= 0.25 且 非恐慌。"""
    b = float((cfg.get("satellite") or {}).get("gate_entry_buffer") or 0.0)
    ok = reg["mkt_idx"] >= reg["mkt_ma"] * (1 + b / 100.0)
    if "breadth" in reg.columns:
        ok = ok & (reg["breadth"] >= 0.25)
    if "panic" in reg.columns:
        ok = ok & (~reg["panic"].astype(bool))
    return ok.fillna(False)


def main() -> None:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    cfg0 = load_config()
    cfg0["backtest"]["trading_days"] = days
    t0 = time.time()
    ctx = prepare_context(cfg0, days)
    reg = ctx["reg"]
    print(f"[sat198c] 窗口 {len(ctx['main_dates'])} 日 "
          f"{ctx['main_dates'][0]}~{ctx['main_dates'][-1]} 准备 {time.time() - t0:.0f}s", flush=True)

    idx = reg["mkt_idx"].astype(float)
    vol20 = idx.pct_change().rolling(20).std() * np.sqrt(252)
    hi60 = idx.rolling(60, min_periods=40).max()
    bd = reg["breadth"].astype(float) if "breadth" in reg.columns else pd.Series(1.0, index=reg.index)

    base = base_entry(reg, cfg0)
    variants: list[tuple[str, pd.Series, dict]] = [
        ("base", base, {}),
        ("e_bmin30", base & (bd >= 0.30), {}),
        ("e_bmin35", base & (bd >= 0.35), {}),
        ("e_vol25", base & (vol20 <= 0.25), {}),
        ("e_vol30", base & (vol20 <= 0.30), {}),
        ("e_nearhi", base & (idx >= 0.95 * hi60), {}),
        ("e_pool30", base, {"min_pool": 30}),
        ("e_gap3", base, {"gap_max": 3.0}),
        ("e_gap2", base, {"gap_max": 2.0}),
    ]

    res: dict[str, dict] = {}
    for name, ok, ov in variants:
        cfg = copy.deepcopy(cfg0)
        cfg["satellite"].update(ov)
        cfg["backtest"]["trading_days"] = days
        t1 = time.time()
        r = run(ctx, cfg, entry_ok=ok)
        m = r["metrics"]
        res[name] = {k: m.get(k) for k in KEYS}
        res[name]["overrides"] = ov
        res[name]["gate_pass"] = int(ok.sum())
        print(f"{name:10s} 年化 {m['年化收益率%']:7.2f}%  回撤 {m['最大回撤%']:7.2f}%  "
              f"卡玛 {m['卡玛比率']:5.2f}  笔数 {m['交易次数']:4d}  胜率 {m['胜率%']:5.2f}%  "
              f"单笔 {m['平均单笔收益%']:+6.3f}%  可入场日 {int(ok.sum())}  ({time.time() - t1:.0f}s)",
              flush=True)
        (OUT / f"exp_sat_198c_{days}.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")

    b = res["base"]
    print("\n=== 相对基线 ===", flush=True)
    for name, m in res.items():
        if name == "base":
            continue
        print(f"{name:10s} 年化 {m['年化收益率%'] - b['年化收益率%']:+7.2f}pp  "
              f"回撤 {m['最大回撤%'] - b['最大回撤%']:+7.2f}pp  "
              f"卡玛 {m['卡玛比率'] - b['卡玛比率']:+5.2f}  "
              f"笔数 {m['交易次数'] - b['交易次数']:+d}", flush=True)
    print(f"[sat198c] 总耗时 {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("BT_FULL_TRADES", "1")
    main()
