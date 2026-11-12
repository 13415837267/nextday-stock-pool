#!/usr/bin/env python3.11
"""第 214 轮: 1500 日窗口 4 变体消融 —— 分离「砍 pullback」与「gap 排序」各自贡献。"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtest import prepare_context, run_backtest
from src.config import load_config


def _M(m: dict) -> dict:
    return {"年化%": m["年化收益率%"], "回撤%": m["最大回撤%"], "卡玛": m["卡玛比率"],
            "笔数": m["交易次数"], "胜率%": m["胜率%"], "均单%": m["平均单笔收益%"]}


def _drop_pb(b: dict) -> dict:
    c = copy.deepcopy(b)
    c["selection"]["fallbacks"] = [f for f in c["selection"].get("fallbacks", [])
                                   if f.get("mode") != "pullback"]
    return c


def _gap(b: dict) -> dict:
    c = copy.deepcopy(b)
    c["execution"]["pick_order"] = "gap"
    return c


def main() -> None:
    base = load_config()
    ctx = prepare_context(base, 1500)
    rows = []
    for name, c in [("基线", copy.deepcopy(base)),
                    ("只砍pullback", _drop_pb(base)),
                    ("只改gap", _gap(base)),
                    ("gap+砍pullback", _gap(_drop_pb(base)))]:
        r = run_backtest(c, 1500, ctx)
        row = _M(r["metrics"])
        row["变体"] = name
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    out = Path(__file__).resolve().parent.parent / "output" / "opt_214d.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"结果写入 {out}", flush=True)


if __name__ == "__main__":
    main()
