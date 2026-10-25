# -*- coding: utf-8 -*-
"""第四十六轮: 风控贡献拆解补全 + edge季度退潮探针(事前注册, 零参数变更)。

背景:
  - 第四十五轮审计后风控贡献矩阵缺最后一块: 趋势条件(等权指数vs MA30)单独贡献。
    事前注册口径: market_ma=1 -> rolling(1).mean()=自身 -> trend_ok恒真, 等效关闭
    趋势条件(宽度条件已证十年零单独触发, panic贡献已测-179pp)。
  - edge退潮探针: era后128笔按买入日季度分桶(n/均值/胜率/复利), 与监测器20笔
    滚动窗口(第三十九/四十五轮)互补——聚合视角回答"2026年的edge还剩多少"。

用法: python3 tools/edge_profile.py
输出: 控制台 + output/edge_profile.json
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context, run_backtest  # noqa: E402
from src.config import load_config  # noqa: E402

ERA = "20240901"
DAYS = 2500
OUT = ROOT / "output" / "edge_profile.json"


def seg_stats(trades: list, era_after: bool) -> dict:
    sel = [t for t in trades if t.get("ret_pct") is not None
           and ((t["buy_date"] >= ERA) == era_after)]
    if not sel:
        return {"n": 0, "avg": None, "wr": None, "comp": None}
    rets = [float(t["ret_pct"]) for t in sel]
    comp = 1.0
    for r in rets:
        comp *= (1 + r / 100)
    return {"n": len(sel), "avg": round(sum(rets) / len(rets), 3),
            "wr": round(100 * sum(r > 0 for r in rets) / len(rets), 1),
            "comp": round((comp - 1) * 100, 1)}


def part_a() -> dict:
    """no_trend(market_ma=1, 趋势条件等效关闭) -> 补全风控贡献矩阵。"""
    cfg0 = load_config()
    bt = json.loads((ROOT / "output" / "backtest_long.json").read_text())
    base_tr, base_mdd = bt["trades_all"], float(bt["metrics"]["最大回撤%"])
    cfg = copy.deepcopy(cfg0)
    cfg["regime"]["market_ma"] = 1   # 事前注册: rolling(1)=自身, trend_ok恒真
    t0 = time.time()
    ctx = prepare_context(cfg, DAYS)
    res = run_backtest(cfg, DAYS, ctx)
    tr = res.get("trades_all") or []
    row = {"key": "no_trend(regime.market_ma:30->1, 趋势条件等效关闭)",
           **{f"tr_{k}": v for k, v in seg_stats(tr, False).items()},
           **{f"va_{k}": v for k, v in seg_stats(tr, True).items()},
           "mdd": float(res["metrics"]["最大回撤%"]),
           "secs": round(time.time() - t0)}
    b = seg_stats(base_tr, False)
    v = seg_stats(base_tr, True)
    print(f"[BASE    ] 训练段 comp={b['comp']}% (n={b['n']}) || "
          f"验证段 comp={v['comp']}% (n={v['n']}) || 回撤{base_mdd}%", flush=True)
    print(f"[no_trend] 训练段 comp={row['tr_comp']}% (n={row['tr_n']}) || "
          f"验证段 comp={row['va_comp']}% (n={row['va_n']}) || 回撤{row['mdd']}%  "
          f"({row['secs']}s)", flush=True)
    print(f"\n  -- 风控贡献矩阵(era后, 相对BASE) --\n"
          f"    全关(45轮):      {row['va_comp']-0:+.0f}pp占位 | 实测 2366.2% (风控贡献+1183pp)\n"
          f"    仅关趋势(本轮):  3549.6% -> {row['va_comp']}% "
          f"(趋势条件贡献{v['comp']-row['va_comp']:+.1f}pp)\n"
          f"    仅关宽度(45轮):  零差异\n"
          f"    仅关panic(45轮): 3549.6% -> 3728.8% (panic代价-179pp)", flush=True)
    return row


def part_b() -> dict:
    """era后128笔按季度分桶 + 全周期滚动20笔均值的历史极值与当前位置。"""
    bt = json.loads((ROOT / "output" / "backtest_long.json").read_text())
    tr = [t for t in bt["trades_all"] if t.get("ret_pct") is not None]
    era = sorted((t for t in tr if t["buy_date"] >= ERA),
                 key=lambda t: t["buy_date"])
    buckets: dict[str, list] = {}
    for t in era:
        y, m = t["buy_date"][:4], int(t["buy_date"][4:6])
        q = f"{y}Q{(m - 1) // 3 + 1}"
        buckets.setdefault(q, []).append(float(t["ret_pct"]))
    rows = []
    print("\n-- edge季度剖面(era后, 按buy_date) --", flush=True)
    for q in sorted(buckets):
        rets = buckets[q]
        comp = 1.0
        for r in rets:
            comp *= (1 + r / 100)
        rows.append({"q": q, "n": len(rets),
                     "avg": round(sum(rets) / len(rets), 3),
                     "wr": round(100 * sum(r > 0 for r in rets) / len(rets), 1),
                     "comp": round((comp - 1) * 100, 1)})
        print(f"  {q}: n={rows[-1]['n']:3d} avg={rows[-1]['avg']:+.3f}% "
              f"wr={rows[-1]['wr']}% comp={rows[-1]['comp']:+.1f}%", flush=True)
    # 滚动20笔均值全周期轨迹(结算序): 极值与尾部位置
    srt = sorted(tr, key=lambda t: t.get("sell_date", ""))
    rets = [float(t["ret_pct"]) for t in srt]
    means = [sum(rets[i - 19:i + 1]) / 20 for i in range(19, len(rets))]
    dates = [srt[i].get("sell_date", "") for i in range(19, len(rets))]
    lo = min(range(len(means)), key=lambda i: means[i])
    hi = max(range(len(means)), key=lambda i: means[i])
    tail5 = sorted(means)[-5:]
    cur = means[-1]
    print(f"\n  滚动20笔均值(结算序{len(means)}个时点): "
          f"min={means[lo]:+.2f}%({dates[lo]}) max={means[hi]:+.2f}%({dates[hi]}) "
          f"当前(回测末笔)={cur:+.2f}%", flush=True)
    print(f"  历史最高5个时点均值: {[round(x, 2) for x in tail5]}", flush=True)
    return {"quarters": rows,
            "rolling20": {"n_points": len(means),
                          "min": round(means[lo], 2), "min_date": dates[lo],
                          "max": round(means[hi], 2), "max_date": dates[hi],
                          "current": round(cur, 2)}}


def main() -> None:
    print("== edge_profile(第四十六轮, 事前注册, 零参数变更) ==", flush=True)
    a = part_a()
    b = part_b()
    OUT.write_text(json.dumps({"meta": {"era": ERA, "days": DAYS},
                               "part_a_no_trend": a, "part_b_quarters": b},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n== 完成, 落盘 {OUT} ==", flush=True)


if __name__ == "__main__":
    main()
