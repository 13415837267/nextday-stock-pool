# -*- coding: utf-8 -*-
"""第 192 轮: 副策略"持有期"重训 —— 笔数太少是不是可以靠缩短持有期解决。

背景
----
副策略现行口径: 低波前 10% → mom20/ivol 排序 → 单吊 → 持 10 个交易日收盘卖。
1500 日只有 106 笔(每年 17 笔), 单笔 +2.55%, 总 728% / 年化 42.2%。
单吊 + 长持有 = 资金周转率 1/10, 数学上把"笔数"锁死了:
  · 单笔 2.55% × 106 笔 = 728%
  · 若 H 减半, 笔数约翻倍, 单笔若只掉 30%, 总收益仍能翻 1.4 倍
第 150 轮测过 H=10/20/40(结论 H=10 最好), 但 3~9 日这一档从未扫过 ——
那是"T+1 买、T+2/T+3 卖"的短线区, 与主策略的 H1 结构最接近, 也是唯一可能把
副策略年化从 40% 抬到 100%+ 的维度。

口径
----
与 tools/satellite_backtest.py 完全一致(同一个 run()), 只覆盖 satellite.hold_days。
entry_ok 序列由 entry_ok_series 统一生成, 与实盘账本同源。
trail/stop/闸门/限价全部不动, 所以这是"结构"实验, 不是参数搜索。

用法: python3 tools/exp_sat_hold_192.py [days]
输出: output/exp_sat_hold_192.json
"""
from __future__ import annotations

import copy
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context                      # noqa: E402
from src.config import load_config                            # noqa: E402
from src.satellite import entry_ok_series                     # noqa: E402

_spec = importlib.util.spec_from_file_location("satbt", ROOT / "tools" / "satellite_backtest.py")
satbt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(satbt)


def main() -> None:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    cfg = load_config()
    ctx = prepare_context(cfg, days)
    dates = ctx["main_dates"]
    print(f"窗口 {len(dates)} 日 {dates[0]} ~ {dates[-1]}", flush=True)
    entry = entry_ok_series(ctx.get("reg"), cfg)

    rows = []
    print(f"{'持有':>4} {'总收益%':>12} {'年化%':>7} {'回撤%':>8} {'卡玛':>6} "
          f"{'笔数':>5} {'单笔%':>7} {'胜率%':>6} {'周转/年':>7}", flush=True)
    for hold in (3, 5, 7, 8, 10, 12, 15, 20):
        t0 = time.time()
        c = copy.deepcopy(cfg)
        c["satellite"]["hold_days"] = hold
        res = satbt.run(ctx, c, entry_ok=entry)
        m = res["metrics"]
        n = int(m["交易次数"])
        r = dict(hold=hold, total=m["总收益率%"], ann=m["年化收益率%"], mdd=m["最大回撤%"],
                 calmar=m["卡玛比率"], n=n, avg=m["平均单笔收益%"], win=m["胜率%"],
                 days=len(dates), turns=round(n / max(len(dates), 1) * 250, 1))
        rows.append(r)
        print(f"{hold:>4} {r['total']:>12,.1f} {r['ann']:>7.1f} {r['mdd']:>8.2f} "
              f"{r['calmar']:>6.2f} {n:>5} {r['avg']:>7.3f} {r['win']:>6.1f} "
              f"{r['turns']:>7.1f}  ({time.time() - t0:.0f}s)", flush=True)

    out = ROOT / "output" / "exp_sat_hold_192.json"
    out.write_text(json.dumps({"days": len(dates), "window": [dates[0], dates[-1]],
                               "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"写出 {out}", flush=True)


if __name__ == "__main__":
    main()
