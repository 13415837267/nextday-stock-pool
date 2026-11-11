# -*- coding: utf-8 -*-
"""第 198 轮(第七批): 把「指数清仓」换成「个股移动止损」。

第二批的 gxoff 证明清仓闸门是有价值的(关掉它年化 -19.5pp、回撤扩 15pp), 但那不等于
"必须用指数判定"。闸门真正做的事是**在下跌初期把仓位撤出来**, 换成个股自己的移动止损
(收盘自持仓最高点回撤 X%)在逻辑上等价, 而且不带指数噪声。

本批: 关掉清仓闸门(gate_exit=False), 把移动止损从 8% 收紧到 3/4/5/6%, 看能不能
用"个股自己的价格"复现甚至超过"指数 MA20"的效果。入场闸门保持现行不变。
"""
from __future__ import annotations

import copy
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context          # noqa: E402
from src.config import load_config               # noqa: E402
from src.satellite import entry_ok_series        # noqa: E402
from tools.satellite_backtest import run         # noqa: E402

OUT = ROOT / "output"
KEYS = ("总收益率%", "年化收益率%", "最大回撤%", "卡玛比率", "夏普比率",
        "交易次数", "胜率%", "平均单笔收益%", "空仓天数")

VARIANTS = [
    ("base", {}),
    ("t3", {"gate_exit": False, "trail_pct": 3.0}),
    ("t4", {"gate_exit": False, "trail_pct": 4.0}),
    ("t5", {"gate_exit": False, "trail_pct": 5.0}),
    ("t6", {"gate_exit": False, "trail_pct": 6.0}),
    ("t4_gate_loser", {"trail_pct": 4.0, "gate_exit_mode": "loser"}),
    ("t5_gate_all", {"trail_pct": 5.0}),
]


def main() -> None:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    cfg0 = load_config()
    cfg0["backtest"]["trading_days"] = days
    t0 = time.time()
    ctx = prepare_context(cfg0, days)
    print(f"[sat198g] 窗口 {len(ctx['main_dates'])} 日 "
          f"{ctx['main_dates'][0]}~{ctx['main_dates'][-1]} 准备 {time.time() - t0:.0f}s", flush=True)

    res: dict[str, dict] = {}
    for name, ov in VARIANTS:
        cfg = copy.deepcopy(cfg0)
        cfg["satellite"].update(ov)
        ok = entry_ok_series(ctx.get("reg"), cfg)
        t1 = time.time()
        r = run(ctx, cfg, entry_ok=ok)
        m = r["metrics"]
        from collections import Counter
        c = Counter(t["note"].split(" ")[0] for t in r["trades"])
        res[name] = {k: m.get(k) for k in KEYS}
        res[name]["overrides"] = ov
        res[name]["exits"] = dict(c)
        print(f"{name:14s} 年化 {m['年化收益率%']:7.2f}%  回撤 {m['最大回撤%']:7.2f}%  "
              f"卡玛 {m['卡玛比率']:5.2f}  笔数 {m['交易次数']:4d}  胜率 {m['胜率%']:5.2f}%  "
              f"单笔 {m['平均单笔收益%']:+6.3f}%  {dict(c.most_common(3))}  ({time.time() - t1:.0f}s)",
              flush=True)
        (OUT / f"exp_sat_198g_{days}.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")

    b = res["base"]
    print("\n=== 相对基线 ===", flush=True)
    for name, m in res.items():
        if name == "base":
            continue
        print(f"{name:14s} 年化 {m['年化收益率%'] - b['年化收益率%']:+7.2f}pp  "
              f"回撤 {m['最大回撤%'] - b['最大回撤%']:+7.2f}pp  "
              f"卡玛 {m['卡玛比率'] - b['卡玛比率']:+5.2f}", flush=True)
    print(f"[sat198g] 总耗时 {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("BT_FULL_TRADES", "1")
    main()
