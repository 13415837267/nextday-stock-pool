# -*- coding: utf-8 -*-
"""第 198 轮: 副策略结构参数复扫(双窗口)。

副策略现存问题(第 197 轮结论): 近六年卡玛 2.04, 全史只有 0.62 / 回撤 -48.6%
(2015-11 峰 → 2016-06 谷, 133 个交易日, 期间只有 11 笔交易, 全是慢慢磨掉的)。
它被定义为"回撤保险", 但保险本身在全史上是亏的, 所以这一轮先修它自己。

一次 prepare_context, 之后在同一份面板上跑多个变体(避免重复抓数据)。
用法: python tools/exp_sat_198.py [1500|3000]
"""
from __future__ import annotations

import copy
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context          # noqa: E402
from src.config import load_config               # noqa: E402
from src.satellite import entry_ok_series        # noqa: E402
from tools.satellite_backtest import run         # noqa: E402

OUT = ROOT / "output"

# 变体: 名字 -> satellite 段覆盖
VARIANTS: dict[str, dict] = {
    "base":        {},
    # ── 并发只数(分散) ─────────────────────────────────────────────
    "mp2":         {"max_pos": 2, "buys_per_day": 2},
    "mp3":         {"max_pos": 3, "buys_per_day": 3},
    "mp4":         {"max_pos": 4, "buys_per_day": 4},
    # ── 持有期 ────────────────────────────────────────────────────
    "h5":          {"hold_days": 5},
    "h7":          {"hold_days": 7},
    "h15":         {"hold_days": 15},
    "h20":         {"hold_days": 20},
    # ── 风控 ──────────────────────────────────────────────────────
    "stop8":       {"stop_pct": -8.0},
    "stop15":      {"stop_pct": -15.0},
    "trail0":      {"trail_pct": 0.0},
    "trail12":     {"trail_pct": 12.0},
    # ── 大盘闸门(入场缓冲) ─────────────────────────────────────────
    "buf0":        {"gate_entry_buffer": 0.0},
    "buf10":       {"gate_entry_buffer": 1.0},
    "buf20":       {"gate_entry_buffer": 2.0},
    # ── 买价(限价深度) ────────────────────────────────────────────
    "pb0":         {"pullback_pct": 0.0},
    "pb2":         {"pullback_pct": 2.0},
    # ── 池子宽度 ──────────────────────────────────────────────────
    "ivol20":      {"ivol_pct_max": 0.2},
}

KEYS = ("总收益率%", "年化收益率%", "最大回撤%", "卡玛比率", "夏普比率",
        "交易次数", "胜率%", "平均单笔收益%", "空仓天数")


def main() -> None:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    cfg0 = load_config()
    cfg0["backtest"]["trading_days"] = days
    t0 = time.time()
    ctx = prepare_context(cfg0, days)
    print(f"[sat198] 窗口 {len(ctx['main_dates'])} 日 "
          f"{ctx['main_dates'][0]}~{ctx['main_dates'][-1]} 准备耗时 {time.time() - t0:.0f}s", flush=True)

    res: dict[str, dict] = {}
    for name, ov in VARIANTS.items():
        cfg = copy.deepcopy(cfg0)
        cfg["satellite"].update(ov)
        cfg["backtest"]["trading_days"] = days
        _entry = entry_ok_series(ctx.get("reg"), cfg)
        t1 = time.time()
        r = run(ctx, cfg, entry_ok=_entry)
        m = r["metrics"]
        res[name] = {k: m.get(k) for k in KEYS}
        res[name]["overrides"] = ov
        res[name]["rules"] = r["meta"].get("rules")
        print(f"{name:10s} 年化 {m['年化收益率%']:7.2f}%  回撤 {m['最大回撤%']:7.2f}%  "
              f"卡玛 {m['卡玛比率']:5.2f}  笔数 {m['交易次数']:4d}  胜率 {m['胜率%']:5.2f}%  "
              f"单笔 {m['平均单笔收益%']:+6.3f}%  ({time.time() - t1:.0f}s)", flush=True)
        # 每跑完一个就落盘, 崩了也不丢
        (OUT / f"exp_sat_198_{days}.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")

    # 与基线对比的年化差(简单相减, 不做显著性 —— 笔数少时 t 检验没有意义)
    b = res["base"]
    print("\n=== 相对基线 ===", flush=True)
    for name, m in res.items():
        if name == "base":
            continue
        print(f"{name:10s} 年化 {m['年化收益率%'] - b['年化收益率%']:+7.2f}pp  "
              f"回撤 {m['最大回撤%'] - b['最大回撤%']:+7.2f}pp  "
              f"卡玛 {m['卡玛比率'] - b['卡玛比率']:+5.2f}", flush=True)
    print(f"[sat198] 总耗时 {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("BT_FULL_TRADES", "1")
    main()
