# -*- coding: utf-8 -*-
"""第一百七十二轮: vwap 门槛的引擎级验证 + 基线不变性检查。

两件事
------
1. 基线不变性: 新增的 vwap_dev 列不应改变任何既有行为 —— 不启用门槛时必须与
   第 170 轮口径逐位一致(6,875.64% / -17.74% / 卡玛 5.87 / 353 笔)。
2. 门槛效果: 研究里 vwap_dev 的 Q1 最好(极差 0.52pp), 但 2024-09 之前为负。
   这里把 dip_vwap_dev_max 设成 -1.0/-1.5/-2.0 直接跑引擎, 看真实复利与笔数,
   并与"分年前后段"对照 —— 如果收益全部来自 2025+, 就是时段依赖, 不应上线。

用法: python tools/exp_vwap_gate.py [天窗口, 默认 1500]
输出: output/exp_vwap_gate.json
"""
from __future__ import annotations

import copy
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ["BT_FULL_TRADES"] = "1"

from src.backtest import prepare_context, run_backtest       # noqa: E402
from src.config import load_config                          # noqa: E402

OUT = os.path.join(ROOT, "output")
THRESHOLDS = [None, -0.5, -1.0, -1.5, -2.0]


def main(days: int = 1500) -> None:
    base_cfg = load_config()
    ctx = prepare_context(base_cfg, days)
    rows = []
    print("\n%-10s %6s %10s %10s %9s %8s %8s"
          % ("阈值", "笔数", "总收益%", "年化%", "回撤%", "卡玛", "单笔%"), flush=True)
    for thr in THRESHOLDS:
        cfg = copy.deepcopy(base_cfg)
        if thr is not None:
            cfg["selection"]["dip_vwap_dev_max"] = thr
        r = run_backtest(cfg, days, ctx)
        m = r["metrics"]
        trades = r.get("trades_all") or []
        # 分年: 用卖出日归属
        by_year: dict[str, list[float]] = {}
        for t in trades:
            y = str(t.get("sell_date", ""))[:4]
            if y:
                by_year.setdefault(y, []).append(float(t.get("ret_pct") or 0) / 100)
        ys = {y: {"n": len(v), "mean_pct": round(sum(v) / len(v) * 100, 3)}
              for y, v in sorted(by_year.items())}
        # 2024-09 前后分段
        pre = [float(t.get("ret_pct") or 0) / 100 for t in trades
               if str(t.get("sell_date", "")) < "20240930"]
        post = [float(t.get("ret_pct") or 0) / 100 for t in trades
                if str(t.get("sell_date", "")) >= "20240930"]
        seg = {"pre": {"n": len(pre), "mean_pct": round(sum(pre) / len(pre) * 100, 3) if pre else None},
               "post": {"n": len(post), "mean_pct": round(sum(post) / len(post) * 100, 3) if post else None}}
        rows.append({"thr": thr, "trades": m["交易次数"], "total_pct": m["总收益率%"],
                     "ann_pct": m["年化收益率%"], "mdd_pct": m["最大回撤%"],
                     "calmar": m["卡玛比率"], "avg_pct": m["平均单笔收益%"],
                     "win_pct": m["胜率%"], "by_year": ys, "seg_202409": seg})
        print("%-10s %6d %10.2f %10.2f %9.2f %8.2f %8.3f  前段%+.3f(n=%d) 后段%+.3f(n=%d)"
              % ("基线" if thr is None else "%.1f" % thr, m["交易次数"], m["总收益率%"],
                 m["年化收益率%"], m["最大回撤%"], m["卡玛比率"] or 0, m["平均单笔收益%"],
                 seg["pre"]["mean_pct"] or 0, seg["pre"]["n"],
                 seg["post"]["mean_pct"] or 0, seg["post"]["n"]), flush=True)

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "exp_vwap_gate.json"), "w", encoding="utf-8") as f:
        json.dump({"window_days": days, "rows": rows}, f, ensure_ascii=False, indent=1)
    print("\nsaved output/exp_vwap_gate.json")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1500)
