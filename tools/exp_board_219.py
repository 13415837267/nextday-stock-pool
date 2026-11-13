"""第219轮: 板块涨跌幅制度 —— 10% 池 vs 20% 池的赔率放大效应

背景:
  主策略在 `selection.mode: dip` 下赚的是"深跌 + 缩量回调 + 次日深低开承接"后的
  T+1→T+2 反抽。反抽幅度有一个机械性上限: **当日涨跌幅限制**。
  10% 板两天最多 +21%; 20% 板两天最多 +44%。同样的信号, 在 20% 板上赔率天然翻倍。

  当前 `universe.boards: [主板]`(第八十九轮用户约束), 创业板/科创被排除。
  本轮不改任何策略参数, 只测"把池子换成/加上 20% 涨跌幅板块"会怎样。

口径:
  · 数据源自带正确的历史涨跌幅规则(已核: 2018 创业板 high/pre_close=1.100,
    2023 后=1.200), 引擎用 high_limit 判定涨停, 无需特殊处理。
  · 引擎/参数/风控/执行全部逐位不变, 只改 boards 列表。
  · 双窗口: 1500 日 + 全史 3664 日。只有双窗口同向且回撤不恶化才算候选。
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.backtest import run_backtest  # noqa: E402

VARIANTS = [
    ("主板(基线)", ["主板"]),
    ("主板+创业板", ["主板", "创业板"]),
    ("仅创业板", ["创业板"]),
    ("主板+创业板+科创", ["主板", "创业板", "科创板"]),
]


def metrics_row(res: dict) -> dict:
    m = res["metrics"]
    return {
        "年化%": m.get("年化收益率%"),
        "总收益%": m.get("总收益率%"),
        "回撤%": m.get("最大回撤%"),
        "卡玛": m.get("卡玛比率"),
        "夏普": m.get("夏普比率"),
        "笔数": m.get("交易次数"),
        "胜率%": m.get("胜率%"),
        "单笔%": m.get("平均单笔收益%"),
    }


def board_of(code: str) -> str:
    c = str(code).split(".")[0]
    if c.startswith("688"):
        return "科创板"
    if c.startswith("30"):
        return "创业板"
    return "主板"


def trade_split(res: dict) -> dict:
    """按板块拆交易: 判断恶化来自"票变差"还是"挤出主板优质票"。"""
    tr = [x for x in (res.get("trades_all") or []) if x.get("ret_pct") is not None]
    out: dict = {"n": len(tr)}
    for b in ("主板", "创业板", "科创板"):
        ss = [x for x in tr if board_of(x.get("code")) == b]
        if ss:
            out[b] = {
                "笔数": len(ss),
                "单笔%": round(sum(x["ret_pct"] for x in ss) / len(ss), 3),
                "胜率%": round(100.0 * sum(1 for x in ss if x["ret_pct"] > 0) / len(ss), 1),
            }
    return out


def main(days_list: list[int], only: list[str] | None = None, with_trades: bool = False) -> None:
    base = load_config()
    out: dict = {"variants": {}, "days": days_list}
    for name, boards in VARIANTS:
        if only and name not in only:
            continue
        out["variants"][name] = {"boards": boards, "runs": {}}
        for days in days_list:
            cfg = copy.deepcopy(base)
            cfg["universe"]["boards"] = boards
            t0 = time.time()
            res = run_backtest(cfg, days)
            row = metrics_row(res)
            row["秒"] = round(time.time() - t0, 1)
            if with_trades:
                row["分板块"] = trade_split(res)
            out["variants"][name]["runs"][str(days)] = row
            print(f"[{name}] {days}日  " + "  ".join(f"{k}={v}" for k, v in row.items()),
                  flush=True)
    p = ROOT / "output" / "exp_board_219.json"
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved", p)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=str, default="1500")
    ap.add_argument("--only", type=str, default=None)
    ap.add_argument("--trades", action="store_true")
    a = ap.parse_args()
    ds = [int(x) for x in a.days.split(",")]
    ol = a.only.split(",") if a.only else None
    if a.trades:
        import os
        os.environ["BT_FULL_TRADES"] = "1"
    main(ds, ol, a.trades)
