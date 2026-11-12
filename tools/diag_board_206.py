"""第206轮: 板块可入池性诊断 —— 加创业板之前, 先看现行门槛会砍掉多少.

现行 universe 门槛: price 3~150 / amount>=0.8亿 / turnover 2~12% / bars>=150 / 非ST非停.
创业板(20cm)的价格中枢更高、换手更高, 12% 换手上限可能把绝大多数票挡在门外。
先算通过率, 再决定是"直接加板块"还是"必须给创业板单独门槛"。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import board_of, load_config          # noqa: E402
from src.backtest import prepare_context              # noqa: E402

DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 3645
cfg = load_config()
ctx = prepare_context(cfg, DAYS)
u = cfg["universe"]
boards = {}

for d in ctx["main_dates"][-1500:]:                    # 近六年抽样即可, 不必全史
    day = ctx["by_date"][d]
    if day is None or not len(day):
        continue
    df = day[(day["is_paused"] == 0) & (day["is_st"] == 0) & (day["bars"] >= u["listed_min_bars"])]
    b = df["ts_code"].map(board_of)
    steps = {
        "原始": pd.Series(True, index=df.index),
        "价 3~150": (df["close"] >= u["price_min"]) & (df["close"] <= u["price_max"]),
        "额>=0.8亿": df["amount"] >= u["amount_min_yi"] * 1e8,
        "换手 2~12%": (df["turnover_rate"] >= u["turnover_min"]) & (df["turnover_rate"] <= u["turnover_max"]),
    }
    cum = pd.Series(True, index=df.index)
    for name, ok in steps.items():
        cum = cum & ok
        for bd in ("主板", "创业板", "科创板"):
            m = (b == bd)
            if m.sum() == 0:
                continue
            boards.setdefault(bd, {}).setdefault(name, []).append(
                float((cum & m).sum()) / max(int(m.sum()), 1) * 100)
    for bd in ("主板", "创业板", "科创板"):
        m = (b == bd)
        if m.sum():
            boards.setdefault(bd, {}).setdefault("每日票数", []).append(int((cum & m).sum()))

rows = []
for bd, dd in boards.items():
    rows.append({"板块": bd,
                 **{k: round(sum(v) / len(v), 1) for k, v in dd.items()}})
out = pd.DataFrame(rows)
print(out.to_string(index=False))
out.to_csv(ROOT / "output" / "diag_board_206.csv", index=False)
