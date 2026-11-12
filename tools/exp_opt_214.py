#!/usr/bin/env python3.11
"""第 214 轮: 主策略提盈实验(第一波) —— 复用一次全史面板, 批量跑变体对比。

基线(全史 3664 日): 年化 75.73% / 回撤 -22.6% / 卡玛 3.35 / 844 笔。
本脚本一次 prepare_context, 然后对每个变体 deepcopy 后 run_backtest(ctx 复用)。
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


def _set(cfg: dict, path: str, value):
    """按 a.b.c 路径设置嵌套 dict 值。"""
    keys = path.split(".")
    d = cfg
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


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


def main() -> None:
    base = load_config()
    print(f"[0] 构建全史上下文({DAYS} 日)…", flush=True)
    ctx = prepare_context(base, DAYS)

    # 变体定义: (名称, {路径: 值})
    variants = [
        ("基线_score", {}),
        ("pick_gap", {"execution.pick_order": "gap"}),
        ("pick_gap_band", {"execution.pick_order": "gap_band",
                           "execution.pick_gap_band": [-3.0, -1.0]}),
        ("dip_pct_-7~-2", {"selection.dip_pct_min": -7.0,
                           "selection.dip_pct_max": -2.0}),
        ("dip_pct_-6~-2", {"selection.dip_pct_max": -2.0}),
        ("close_pos_0.45", {"selection.dip_close_pos_max": 0.45}),
        ("mom20_到35", {"selection.dip_mom20_max": 35.0}),
        ("dist_low20_到4", {"selection.dip_dist_low20_min": 4.0}),
    ]

    rows = []
    for name, patch in variants:
        cfg = copy.deepcopy(base)
        for path, value in patch.items():
            _set(cfg, path, value)
        try:
            r = run_backtest(cfg, DAYS, ctx)
            row = _metrics(r["metrics"])
            row["变体"] = name
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"变体 {name} 失败: {type(e).__name__}: {e}", flush=True)
            rows.append({"变体": name, "错误": f"{type(e).__name__}: {e}"})

    out = Path(__file__).resolve().parent.parent / "output" / "opt_214.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n结果已写入 {out}", flush=True)


if __name__ == "__main__":
    main()
