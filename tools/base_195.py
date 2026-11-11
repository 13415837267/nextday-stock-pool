# -*- coding: utf-8 -*-
"""第一百九十五轮: 无杠杆基线 (leverage=1.0) —— 双窗口落盘 metrics / equity / trades_all。

用法: python3 tools/base_195.py 1500|3000
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ["BT_FULL_TRADES"] = "1"
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context, run_backtest   # noqa: E402
from src.config import load_config                       # noqa: E402


def main() -> None:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    cfg = load_config()
    ex = cfg["execution"]
    print(f"杠杆 {ex.get('leverage')} · 副策略 {cfg['satellite'].get('leverage')}", flush=True)
    t0 = time.time()
    ctx = prepare_context(cfg, days)
    res = run_backtest(cfg, days, ctx)
    m = res["metrics"]
    print(f"窗口 {res['meta']['start']} ~ {res['meta']['end']} ({res['meta']['days']} 日) "
          f"{time.time() - t0:.0f}s", flush=True)
    for k, v in m.items():
        print(f"  {k}: {v}", flush=True)
    out = {
        "days": days,
        "leverage": ex.get("leverage"),
        "meta": res["meta"],
        "metrics": m,
        "equity": res["equity"],
        "trades_all": res["trades_all"],
        "monthly": res["monthly"],
    }
    p = ROOT / "output" / f"base_195_{days}.json"
    p.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"→ {p}", flush=True)


if __name__ == "__main__":
    main()
