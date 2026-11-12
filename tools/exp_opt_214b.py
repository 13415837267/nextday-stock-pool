#!/usr/bin/env python3.11
"""第 214 轮: 主策略提盈实验(第二波) —— fallback 备用池消融 + pick_gap 确认。

第一波结论: pick_order=gap 全史年化 75.73→76.93(+1.2pp)、回撤 -22.60→-22.03、卡玛 3.35→3.49。
本波验证: 备用池(pullback/oversold)全史负贡献, 砍掉后是否提升。
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

DAYS = int(os.environ.get("OPT_DAYS", "3664"))


def _metrics(m: dict) -> dict:
    return {
        "年化%": m["年化收益率%"],
        "回撤%": m["最大回撤%"],
        "卡玛": m["卡玛比率"],
        "笔数": m["交易次数"],
        "胜率%": m["胜率%"],
        "均单%": m["平均单笔收益%"],
        "空仓天": m["空仓天数"],
        "跳过": m["跳过买入次数"],
    }


def _drop_fb(cfg: dict, modes: set[str]) -> None:
    fbs = cfg["selection"].get("fallbacks") or []
    cfg["selection"]["fallbacks"] = [f for f in fbs if f.get("mode") not in modes]


def main() -> None:
    base = load_config()
    print(f"[0] 构建全史上下文({DAYS} 日)…", flush=True)
    ctx = prepare_context(base, DAYS)

    def mk(name: str, pick_order: str, drop: set[str] | None) -> dict:
        cfg = copy.deepcopy(base)
        cfg["execution"]["pick_order"] = pick_order
        if drop:
            _drop_fb(cfg, drop)
        return cfg

    variants = [
        ("gap_保留3备用", mk("gap", "gap", None)),
        ("gap_砍pullback", mk("gap", "gap", {"pullback"})),
        ("gap_砍pullback+oversold", mk("gap", "gap", {"pullback", "oversold"})),
        ("gap_无备用", mk("gap", "gap", {"momentum", "oversold", "pullback"})),
        ("score_砍pullback+oversold", mk("score", "score", {"pullback", "oversold"})),
    ]

    rows = []
    for name, cfg in variants:
        try:
            r = run_backtest(cfg, DAYS, ctx)
            row = _metrics(r["metrics"])
            row["变体"] = name
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"变体 {name} 失败: {type(e).__name__}: {e}", flush=True)
            rows.append({"变体": name, "错误": f"{type(e).__name__}: {e}"})

    out = Path(__file__).resolve().parent.parent / "output" / "opt_214b.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n结果已写入 {out}", flush=True)


if __name__ == "__main__":
    main()
