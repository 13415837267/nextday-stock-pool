# -*- coding: utf-8 -*-
"""副策略并发只数: 1 只(单吊) vs 2 只 vs 4 只(现行) —— 第一百七十六轮。

用户诉求: "给出评分最高的一个股票作明日计划" —— 计划层每天只推 1 只。
需要回答: 只买第 1 名(而不是 4 只分散)会付出多少代价? 若代价在噪声内,
计划就可以诚实地写"只买这一只"; 若代价很大, 必须同时披露"第 2~4 名是分散用的"。

口径: 与 tools/satellite_backtest.py 完全一致(同引擎/同成本/同止损), 只改
satellite.max_pos 与 buys_per_day。ctx 只构建一次, 三个变体共用。

用法: python3 tools/exp_plan_top1.py [--days 1500]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context          # noqa: E402
from src.config import load_config                # noqa: E402
import tools.satellite_backtest as sb             # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=1500)
    args = ap.parse_args()
    cfg0 = load_config()
    print(f"构建上下文 {args.days} 日 ...", flush=True)
    ctx = prepare_context(cfg0, args.days)
    print(f"窗口 {ctx['main_dates'][0]} ~ {ctx['main_dates'][-1]}", flush=True)

    out = {}
    for n in (1, 2, 4):
        cfg = json.loads(json.dumps(cfg0))
        cfg.setdefault("satellite", {})
        cfg["satellite"]["max_pos"] = n
        cfg["satellite"]["buys_per_day"] = n
        r = sb.run(ctx, cfg)
        m = r["metrics"]
        out[str(n)] = {k: m.get(k) for k in
                       ("总收益率%", "年化收益率%", "最大回撤%", "卡玛比率", "夏普比率",
                        "交易次数", "胜率%", "平均单笔收益%", "空仓天数")}
        print(f"max_pos={n}: 年化 {m['年化收益率%']:+.2f}% 回撤 {m['最大回撤%']:.2f}% "
              f"卡玛 {m['卡玛比率']} 笔数 {m['交易次数']} 胜率 {m['胜率%']}% "
              f"单笔 {m['平均单笔收益%']:+.3f}%", flush=True)
    p = ROOT / "output" / "exp_plan_top1.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("写出", p, flush=True)


if __name__ == "__main__":
    main()
