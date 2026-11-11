# -*- coding: utf-8 -*-
"""第一百九十五轮: 无杠杆(lev=1.0)下的执行侧复扫。

杠杆从 1.25 降到 1.0 之后, 有三类参数的目标函数变了:
  ① 闲置现金的价值上升(账户不再靠融资放大, 空仓期的钱是真闲置);
  ② 风控期"半仓参与"(blocked_frac)的相对代价变了(以前是 1.25x 下的 0.4 仓);
  ③ 分仓(2×50%)以前要跟 1.25x 单吊比, 现在两者都用满 100% 权益, 才是同口径对比。

用法: python3 tools/exp_nolev_195.py 1500
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
    # (标签, 覆盖项)
    ("基准(lev=1.0)", {}),
    ("闲置现金 国债逆回购 1.5%", {"execution.cash_yield": 0.015}),
    ("闲置现金 货币基金 2.0%", {"execution.cash_yield": 0.020}),
    ("风控期不参与(blocked_frac 0)", {"execution.blocked_frac": 0.0}),
    ("风控期两成(blocked_frac 0.2)", {"execution.blocked_frac": 0.2}),
    ("风控期六成(blocked_frac 0.6)", {"execution.blocked_frac": 0.6}),
    ("分仓 2×50%", {"execution.alloc_frac_pct": 50.0, "execution.max_buy_per_day": 2,
                    "selection.exec_top_n": 2, "selection.display_top_n": 2}),
    ("分仓 2×50% + 现金 1.5%", {"execution.alloc_frac_pct": 50.0, "execution.max_buy_per_day": 2,
                              "selection.exec_top_n": 2, "selection.display_top_n": 2,
                              "execution.cash_yield": 0.015}),
    ("半仓单吊(alloc 70%)", {"execution.alloc_frac_pct": 70.0}),
]


def apply(cfg: dict, ov: dict) -> dict:
    c = copy.deepcopy(cfg)
    for k, v in ov.items():
        sec, key = k.split(".")
        c[sec][key] = v
    return c


def main() -> None:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
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
                                 for k, v in m.items()}})
        print(f"  {name:28s} 总 {m['总收益率%']:>12,.0f}%  年化 {m['年化收益率%']:6.2f}%  "
              f"回撤 {m['最大回撤%']:6.2f}%  卡玛 {m['卡玛比率']:5.2f}  "
              f"笔 {m['交易次数']:4d}  单笔 {m['平均单笔收益%']:6.3f}%  ({time.time() - t:.0f}s)",
              flush=True)
    p = ROOT / "output" / f"exp_nolev_195_{days}.json"
    p.write_text(json.dumps({"days": days, "rows": rows,
                             "base": rows[0]["metrics"]}, ensure_ascii=False), encoding="utf-8")
    print(f"→ {p}", flush=True)


if __name__ == "__main__":
    main()
