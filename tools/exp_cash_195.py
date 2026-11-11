# -*- coding: utf-8 -*-
"""第一百九十五轮: 闲置现金计息(cash_yield)的双窗口验证 + 分年。

1.0x 不用融资之后, 空仓期那部分钱是真闲置(1500 日里 1133 天收盘无持仓)。
实盘上可以买国债逆回购 / 货币 ETF(年化 1.5%~2%), 回测里由 execution.cash_yield
按"正现金余额 × 年化 / 252"逐日计提, 不改变任何买卖判定 —— 是纯增益, 不是 alpha。

用法: python3 tools/exp_cash_195.py 3000
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context, run_backtest   # noqa: E402
from src.config import load_config                       # noqa: E402

VARIANTS = [
    ("基准(现金 0%)", 0.0),
    ("现金 1.5%(国债逆回购)", 0.015),
    ("现金 2.5%(上限敏感度)", 0.025),
]


def main() -> None:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    cfg = load_config()
    print(f"窗口 {days} · 杠杆 {cfg['execution']['leverage']}", flush=True)
    ctx = prepare_context(cfg, days)
    rows = []
    for name, y in VARIANTS:
        c = copy.deepcopy(cfg)
        c["execution"]["cash_yield"] = y
        res = run_backtest(c, days, ctx)
        m = res["metrics"]
        rows.append({"name": name, "cash_yield": y,
                     "metrics": {k: (float(v) if isinstance(v, (int, float)) else v)
                                 for k, v in m.items()},
                     "monthly": res["monthly"]})
        print(f"  {name:24s} 总 {m['总收益率%']:>14,.0f}%  年化 {m['年化收益率%']:6.2f}%  "
              f"回撤 {m['最大回撤%']:6.2f}%  卡玛 {m['卡玛比率']:5.2f}  笔 {int(m['交易次数'])}",
              flush=True)
    p = ROOT / "output" / f"exp_cash_195_{days}.json"
    p.write_text(json.dumps({"days": days, "rows": rows}, ensure_ascii=False), encoding="utf-8")
    print(f"→ {p}", flush=True)


if __name__ == "__main__":
    main()
