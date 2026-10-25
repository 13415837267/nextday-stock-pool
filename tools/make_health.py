# -*- coding: utf-8 -*-
"""生成 output/health.json —— 网页"策略体检"区数据(可重复运行刷新)。

探针口径: 复用 src.monitor.compute_monitor 对十年回测(backtest_long.json)
结算序的滚动终态 —— 与第四十六轮 edge_profile 的"滚动20笔均值"完全一致,
仅改由 compute_monitor 给出标准化的 level/note(第三十九轮注册阈值)。
静态结论块(era判定/苗头清单): 源自第四十三~四十六轮事前注册诊断, 注明轮次。

数据依赖: output/backtest_long.json(十年全量成交) + output/edge_profile.json(季度剖面)。
若缺失则降级生成仅含静态结论的版本(网页仍可渲染)。

用法: python3 tools/make_health.py
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.monitor import compute_monitor  # noqa: E402

ERA_VERDICT = ("edge依赖2024-09后市场微观结构: 2016-2023八年连亏(-91%), "
               "2024-09后128笔+3.14%/笔; 参数经12组走前反证不可调(第44轮); "
               "风控era后贡献+1183pp、趋势条件为主力(第45/46轮); "
               "监测器era前90%时间告警/era后85%正常(第45轮)")
RECHECK = [
    {"name": "苗头④ 基差5日斜率", "state": "t=1.64, 观察级"},
    {"name": "苗头③ 营收同比[-10,0)", "state": "n=55, 十年复核优势收窄"},
    {"name": "苗头② 同花顺热度冷门桶", "state": "n=8, 观察级"},
    {"name": "苗头① 龙虎榜T-5上榜", "state": "n=2, 观察级"},
]
SOURCES = ("来源: 第43轮(edge稳健性)/44(参数反证)/45(防线审计)/46(退潮探针), "
           "详见仓库 CHANGELOG.md; 均为事前注册诊断, 零参数变更")


def main() -> None:
    cfg = load_config()
    mon = cfg.get("monitor") or {"window": 20}
    probe, quarters, roll = {}, [], {}
    bt_path = ROOT / "output" / "backtest_long.json"
    ep_path = ROOT / "output" / "edge_profile.json"
    if bt_path.exists():
        bt = json.loads(bt_path.read_text())
        tr = sorted((t for t in bt["trades_all"] if t.get("ret_pct") is not None),
                    key=lambda t: t.get("sell_date", ""))
        m = compute_monitor([float(t["ret_pct"]) for t in tr], mon)   # 滚动终态
        probe = {"level": m["level"], "mean_pct": m["mean_pct"],
                 "win_pct": m["win_pct"], "n": m["n"], "window": m["window"],
                 "asof": tr[-1].get("sell_date", ""),
                 "note": ("回测口径探针(假设2016年起连续执行), 与实盘监测灯互补; "
                          "单季波动属常态, 实盘滚动均值<0%→红、<-1%→深红按注册行为处置")}
    if ep_path.exists():
        ep = json.loads(ep_path.read_text())
        quarters = ep["part_b_quarters"]["quarters"]
        roll = ep["part_b_quarters"]["rolling20"]
    if probe and roll:
        probe["peak_pct"] = roll["max"]
        probe["peak_date"] = roll["max_date"]

    health = {"updated": date.today().isoformat(), "probe": probe,
              "quarters": quarters, "era": ERA_VERDICT,
              "recheck": RECHECK, "sources": SOURCES}
    out = ROOT / "output" / "health.json"
    out.write_text(json.dumps(health, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"health.json 已生成 -> {out}")
    print(json.dumps(probe, ensure_ascii=False))


if __name__ == "__main__":
    main()
