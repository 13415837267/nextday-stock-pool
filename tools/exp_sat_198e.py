# -*- coding: utf-8 -*-
"""第 198 轮(第五批): 两个真正没扫过的旋钮 —— 挂单等待天数、持满时的限价止盈。

前面四批把常规旋钮扫完了(并发/持有期/止损/移动止损/入场缓冲/清仓口径/清仓确认/
限价深度/池宽/宽度地板/波动过滤/跳空), 唯一还没动过的是**成交机制**:

  · pullback_days: 挂"收盘-1%"的限价单等几天有效(现值 3)。等得久成交率高但信号过期。
  · exit_limit_pct: 持满 10 日时, 不直接市价卖, 而是挂"收盘+X%"的止盈单等几天 ——
    35 笔持满的平均 +7.02%, 如果能在更高价出货就是白捡。

另外补一个组合: max_pos=2 但每天最多买 1 只(现值是 1/1)。
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
    ("live_old", {"pullback_days": 1}),   # 第198轮前的实盘口径: 只撮合 T+1 一次
    ("wait4", {"pullback_days": 4}),
    ("wait8", {"pullback_days": 8}),
]


def main() -> None:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    cfg0 = load_config()
    cfg0["backtest"]["trading_days"] = days
    t0 = time.time()
    ctx = prepare_context(cfg0, days)
    print(f"[sat198e] 窗口 {len(ctx['main_dates'])} 日 "
          f"{ctx['main_dates'][0]}~{ctx['main_dates'][-1]} 准备 {time.time() - t0:.0f}s", flush=True)

    res: dict[str, dict] = {}
    for name, ov in VARIANTS:
        cfg = copy.deepcopy(cfg0)
        cfg["satellite"].update(ov)
        _entry = entry_ok_series(ctx.get("reg"), cfg)
        t1 = time.time()
        r = run(ctx, cfg, entry_ok=_entry)
        m = r["metrics"]
        res[name] = {k: m.get(k) for k in KEYS}
        res[name]["overrides"] = ov
        from collections import Counter
        c = Counter(t["note"].split(" ")[0] for t in r["trades"])
        res[name]["exits"] = dict(c)
        print(f"{name:10s} 年化 {m['年化收益率%']:7.2f}%  回撤 {m['最大回撤%']:7.2f}%  "
              f"卡玛 {m['卡玛比率']:5.2f}  笔数 {m['交易次数']:4d}  胜率 {m['胜率%']:5.2f}%  "
              f"单笔 {m['平均单笔收益%']:+6.3f}%  {dict(c.most_common(3))}  ({time.time() - t1:.0f}s)",
              flush=True)
        (OUT / f"exp_sat_198e_{days}.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")

    b = res["base"]
    print("\n=== 相对基线 ===", flush=True)
    for name, m in res.items():
        if name == "base":
            continue
        print(f"{name:10s} 年化 {m['年化收益率%'] - b['年化收益率%']:+7.2f}pp  "
              f"回撤 {m['最大回撤%'] - b['最大回撤%']:+7.2f}pp  "
              f"卡玛 {m['卡玛比率'] - b['卡玛比率']:+5.2f}  笔数 {m['交易次数'] - b['交易次数']:+d}",
              flush=True)
    print(f"[sat198e] 总耗时 {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("BT_FULL_TRADES", "1")
    main()
