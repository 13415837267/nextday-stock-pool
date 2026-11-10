# -*- coding: utf-8 -*-
"""第184轮·C: 主策略(dip, H1)加文献因子门槛 —— 双窗口

主策略是全市场扫 dip(缩量阴线回调), 每天只取 top1, 1500 日 379 笔。
与副策略不同, 这里的池子很宽(每天几十~几百只), 所以"加一道门槛"的代价是
**笔数减少**, 收益可能来自"每笔更干净" —— 必须两个方向一起看, 不能只看倍数。

四个门槛(列由 selector.build_indicators 新增, 见第184轮注释):
  dip_max20_max    20 日最大单日涨幅上限(彩票股过滤)
  dip_imax20_max   20 日最大超额收益上限(IMAX)
  dip_on_abs10_max 10 日隔夜跳空绝对值上限(华安: |跳空|大者次月负 alpha)
  dip_ctc20_max    20 日价量相关性上限(低 CTC 更优)

口径: 与现役完全一致(杠杆 1.25 / 成本 / T+1 开盘买 / T+2 收盘卖), 只叠加门槛。
      base 必须复现 498x(1500 日)才说明口径没跑偏。

用法: python tools/exp_web_main.py 1500
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context, run_backtest  # noqa: E402
from src.config import load_config  # noqa: E402

PLAN = [
    ("base", {}),
    ("max20_le8", {"dip_max20_max": 8.0}),
    ("max20_le9.5", {"dip_max20_max": 9.5}),
    ("imax_le6", {"dip_imax20_max": 6.0}),
    ("imax_le7.5", {"dip_imax20_max": 7.5}),
    ("onabs_le1.5", {"dip_on_abs10_max": 1.5}),
    ("onabs_le2", {"dip_on_abs10_max": 2.0}),
    ("ctc_le0.3", {"dip_ctc20_max": 0.3}),
    ("ctc_le0.5", {"dip_ctc20_max": 0.5}),
]


def main():
    cfg = load_config()
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    ctx = prepare_context(cfg, days)
    print(f"实际 {len(ctx['main_dates'])} 日 {ctx['main_dates'][0]} ~ {ctx['main_dates'][-1]}",
          flush=True)
    out = {"days": len(ctx["main_dates"]), "start": ctx["main_dates"][0],
           "end": ctx["main_dates"][-1], "variants": {}}
    for nm, ov in PLAN:
        c = copy.deepcopy(cfg)
        c["selection"].update(ov)
        try:
            r = run_backtest(c, days, ctx=ctx)
        except Exception as ex:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            out["variants"][nm] = {"error": str(ex)}
            continue
        m = r["metrics"]
        out["variants"][nm] = {k: m[k] for k in m if isinstance(m[k], (int, float))}
        print(f"{nm:12s} 总{m['总收益率%']:>10.1f}% 年化{m['年化收益率%']:7.1f}% "
              f"回撤{m['最大回撤%']:6.1f}% 卡玛{m['卡玛比率']:5.2f} 笔{m['交易次数']:4d} "
              f"胜{m['胜率%']:5.1f}% 单笔{m['平均单笔收益%']:+.2f}%", flush=True)
    p = ROOT / f"output/exp_web_main_{days}.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"写出 {p}", flush=True)


if __name__ == "__main__":
    main()
