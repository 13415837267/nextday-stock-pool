# -*- coding: utf-8 -*-
"""第 192 轮: 副策略杠杆前沿 —— 把它推到百倍需要多少倍杠杆、代价是什么。

为什么做
--------
持有期网格(exp_sat_hold_192)证明 H10 已是收益最高的档: H3 单笔 0.079% → H10 2.546%,
缩短持有期只会把单笔期望打没, 不会靠"提笔数"把年化抬上去。也就是说副策略的
106 笔 × 2.546% 就是它的结构上限(728%)。要摸到 100 倍, 只剩"放大单笔"一条路。

口径
----
与 tools/satellite_backtest.py 完全一致, 只覆盖 satellite.leverage
(第 171 轮已实现: lev>1 时允许透支 (lev-1)*nav, 按 margin_rate 年化逐日计提)。
同时给出"历史最坏时刻的维持担保比例"= L*(1+mdd)/(L-1), 与主策略第 191 轮同口径。

用法: python3 tools/exp_sat_lev_192.py [days]
输出: output/exp_sat_lev_192.json
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


def ratio(L: float, mdd_pct: float) -> float | None:
    """维持担保比例(%): 总资产/负债 = L*(1+dd)/(L-1)。dd 为负。"""
    if L <= 1.0:
        return None
    return round(L * (1 + mdd_pct / 100.0) / (L - 1.0) * 100, 0)


def main() -> None:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    cfg = load_config()
    ctx = prepare_context(cfg, days)
    dates = ctx["main_dates"]
    entry = entry_ok_series(ctx.get("reg"), cfg)
    print(f"窗口 {len(dates)} 日 {dates[0]} ~ {dates[-1]}", flush=True)

    rows = []
    print(f"{'杠杆':>5} {'总倍数':>10} {'年化%':>7} {'回撤%':>8} {'卡玛':>6} "
          f"{'笔数':>5} {'单笔%':>7} {'担保%':>7} {'强平?':>6}", flush=True)
    for lev in (1.0, 1.25, 1.5, 1.75, 2.0, 2.5):
        t0 = time.time()
        c = copy.deepcopy(cfg)
        c["satellite"]["leverage"] = lev
        res = satbt.run(ctx, c, entry_ok=entry)
        m = res["metrics"]
        tot = float(m["总收益率%"])
        mdd = float(m["最大回撤%"])
        rt = ratio(lev, mdd)
        r = dict(lev=lev, total_pct=tot, mult=round(1 + tot / 100.0, 2), ann=m["年化收益率%"],
                 mdd=mdd, calmar=m["卡玛比率"], n=int(m["交易次数"]),
                 avg=m["平均单笔收益%"], ratio=rt, margin_call=(rt is not None and rt < 130))
        rows.append(r)
        print(f"{lev:>5.2f} {r['mult']:>10,.1f} {r['ann']:>7.1f} {mdd:>8.2f} "
              f"{r['calmar']:>6.2f} {r['n']:>5} {r['avg']:>7.3f} "
              f"{(f'{rt:.0f}' if rt else '—'):>7} {('YES' if r['margin_call'] else ''):>6}  ({time.time()-t0:.0f}s)",
              flush=True)

    out = ROOT / "output" / "exp_sat_lev_192.json"
    out.write_text(json.dumps({"days": len(dates), "window": [dates[0], dates[-1]],
                               "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"写出 {out}", flush=True)


if __name__ == "__main__":
    main()
