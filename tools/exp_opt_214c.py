#!/usr/bin/env python3.11
"""第 214 轮: 主策略提盈实验(第三波) —— 优化方案的分窗口稳健性验证。

全史(3664)已见: gap+砍pullback 年化 75.73→80.72、卡玛 3.35→3.66、回撤不变。
本波在 1500/3000 日窗口复验, 排除"全史特有、近窗口退化"的窗口效应。
"""
from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtest import prepare_context, run_backtest
from src.config import load_config


def _metrics(m: dict) -> dict:
    return {
        "年化%": m["年化收益率%"],
        "回撤%": m["最大回撤%"],
        "卡玛": m["卡玛比率"],
        "笔数": m["交易次数"],
        "胜率%": m["胜率%"],
        "均单%": m["平均单笔收益%"],
    }


def _opt_cfg(base: dict) -> dict:
    cfg = copy.deepcopy(base)
    cfg["execution"]["pick_order"] = "gap"
    fbs = cfg["selection"].get("fallbacks") or []
    cfg["selection"]["fallbacks"] = [f for f in fbs if f.get("mode") != "pullback"]
    return cfg


def main() -> None:
    base = load_config()
    rows = []
    for days in [1500, 3000]:
        print(f"[0] 构建上下文({days} 日)…", flush=True)
        ctx = prepare_context(base, days)
        for name, cfg in [("基线", copy.deepcopy(base)), ("gap砍pullback", _opt_cfg(base))]:
            r = run_backtest(cfg, days, ctx)
            row = _metrics(r["metrics"])
            row.update({"窗口": days, "变体": name})
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)

    out = Path(__file__).resolve().parent.parent / "output" / "opt_214c.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n结果已写入 {out}", flush=True)


if __name__ == "__main__":
    main()
