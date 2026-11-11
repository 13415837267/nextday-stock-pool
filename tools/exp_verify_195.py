# -*- coding: utf-8 -*-
"""第一百九十五轮: 把 GA 搜出来的两个基因变动放到长窗口做样本外验证。

GA(仅 1500 日)只动了两个基因: bias20_max 11.4->12.0、sell_full_if_open_gain 15->12。
两者在历史注释里都留过"全史更差"的记录, 所以必须逐项拆开在 3000 日上复验。

用法: python3 tools/exp_verify_195.py 3000
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
    ("基线(现行参数)", {}),
    ("GA最优(bias12.0 + sfg12)", {"selection.bias20_max": 12.0,
                                 "execution.sell_full_if_open_gain": 12.0}),
    ("只改 bias 11.4->12.0", {"selection.bias20_max": 12.0}),
    ("只改 sfg 15->12", {"execution.sell_full_if_open_gain": 12.0}),
]


def apply(cfg: dict, ov: dict) -> dict:
    c = copy.deepcopy(cfg)
    for k, v in ov.items():
        sec, key = k.split(".")
        c[sec][key] = v
    return c


def main() -> None:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    cfg = load_config()
    print(f"窗口 {days} · 杠杆 {cfg['execution']['leverage']}", flush=True)
    t0 = time.time()
    ctx = prepare_context(cfg, days)
    print(f"ctx {time.time() - t0:.0f}s", flush=True)
    rows = []
    for name, ov in VARIANTS:
        t = time.time()
        res = run_backtest(apply(cfg, ov), days, ctx)
        m = res["metrics"]
        rows.append({"name": name, "ov": ov,
                     "metrics": {k: (float(v) if isinstance(v, (int, float)) else v)
                                 for k, v in m.items()},
                     "monthly": res["monthly"]})
        print(f"  {name:26s} 总 {m['总收益率%']:>14,.0f}%  年化 {m['年化收益率%']:6.2f}%  "
              f"回撤 {m['最大回撤%']:6.2f}%  卡玛 {m['卡玛比率']:5.2f}  "
              f"笔 {m['交易次数']:4d}  单笔 {m['平均单笔收益%']:6.3f}%  ({time.time() - t:.0f}s)",
              flush=True)
    p = ROOT / "output" / f"exp_verify_195_{days}.json"
    p.write_text(json.dumps({"days": days, "rows": rows}, ensure_ascii=False), encoding="utf-8")
    print(f"→ {p}", flush=True)


if __name__ == "__main__":
    main()
