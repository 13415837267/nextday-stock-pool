# -*- coding: utf-8 -*-
"""第 198 轮(第四批): 清仓闸门的时间维度 —— 缓冲带与确认日。

前三批把所有"数值旋钮"扫完了(并发只数/持有期/止损/移动止损/入场缓冲/限价深度/池宽/
宽度地板/波动过滤/接近高点/跳空上限), 除 mp4 在近六年略有改善外全部持平或更差。
剩下唯一没动过的是**清仓闸门的判定时机**:

  · 现值: 指数收盘跌破 MA20 当日就全部清仓(不等确认)。1500 日 61 笔这样出场, 均 +0.38%。
  · exit_buf: 指数须跌破 MA20×(1-b%) 才清仓 —— 给一点缓冲, 少砍在临界点上。
  · exit_confirm2: 连续 2 日跌破才清仓 —— 过滤单日假摔。

只改 reg.trade_ok(清仓用), entry_ok 保持现行(+0.3% 缓冲), 入场完全不变。
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


def exit_series(reg: pd.DataFrame, buf: float, confirm: int) -> pd.Series:
    """清仓闸门: 指数 >= MA20×(1-buf%) 视为"可持有"; confirm>1 时需连续 N 日跌破才关。"""
    idx = reg["mkt_idx"].astype(float)
    ok = idx >= reg["mkt_ma"].astype(float) * (1 - buf / 100.0)
    if "breadth" in reg.columns:
        ok = ok & (reg["breadth"].astype(float) >= 0.25)
    if "panic" in reg.columns:
        ok = ok & (~reg["panic"].astype(bool))
    ok = ok.fillna(False)
    if confirm > 1:
        # 连续 confirm 日"跌破"才关门: 用前 confirm-1 日也都不满足来判定
        bad = (~ok).astype(int)
        closed = bad.rolling(confirm).sum() >= confirm
        ok = ~closed.fillna(False)
    return ok


def main() -> None:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    cfg0 = load_config()
    cfg0["backtest"]["trading_days"] = days
    t0 = time.time()
    ctx = prepare_context(cfg0, days)
    print(f"[sat198d] 窗口 {len(ctx['main_dates'])} 日 "
          f"{ctx['main_dates'][0]}~{ctx['main_dates'][-1]} 准备 {time.time() - t0:.0f}s", flush=True)
    entry = entry_ok_series(ctx["reg"], cfg0)

    variants = [
        ("base", 0.0, 1),
        ("xb05", 0.5, 1),
        ("xb10", 1.0, 1),
        ("xb20", 2.0, 1),
        ("xc2", 0.0, 2),
        ("xc3", 0.0, 3),
        ("xb10_c2", 1.0, 2),
    ]
    res: dict[str, dict] = {}
    for name, buf, conf in variants:
        cfg = copy.deepcopy(cfg0)
        ctx2 = dict(ctx)
        reg2 = ctx["reg"].copy()
        reg2["trade_ok"] = exit_series(ctx["reg"], buf, conf)
        ctx2["reg"] = reg2
        t1 = time.time()
        r = run(ctx2, cfg, entry_ok=entry)
        m = r["metrics"]
        res[name] = {k: m.get(k) for k in KEYS}
        res[name]["exit_buf"] = buf
        res[name]["confirm"] = conf
        res[name]["hold_ok_days"] = int(reg2["trade_ok"].sum())
        print(f"{name:9s} 年化 {m['年化收益率%']:7.2f}%  回撤 {m['最大回撤%']:7.2f}%  "
              f"卡玛 {m['卡玛比率']:5.2f}  笔数 {m['交易次数']:4d}  胜率 {m['胜率%']:5.2f}%  "
              f"单笔 {m['平均单笔收益%']:+6.3f}%  可持有日 {int(reg2['trade_ok'].sum())}  "
              f"({time.time() - t1:.0f}s)", flush=True)
        (OUT / f"exp_sat_198d_{days}.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")

    b = res["base"]
    print("\n=== 相对基线 ===", flush=True)
    for name, m in res.items():
        if name == "base":
            continue
        print(f"{name:9s} 年化 {m['年化收益率%'] - b['年化收益率%']:+7.2f}pp  "
              f"回撤 {m['最大回撤%'] - b['最大回撤%']:+7.2f}pp  "
              f"卡玛 {m['卡玛比率'] - b['卡玛比率']:+5.2f}  笔数 {m['交易次数'] - b['交易次数']:+d}",
              flush=True)
    print(f"[sat198d] 总耗时 {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("BT_FULL_TRADES", "1")
    main()
