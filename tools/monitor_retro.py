# -*- coding: utf-8 -*-
"""失效监测器回溯演示(2026-10-24 第三十九轮, 描述性用途, 不改参数)。

对两份现存成交序列逐笔滚动运行监测器, 展示"如果监测器当时在线,
各时点会亮什么灯":
  1) output/backtest.json trades_executed_recent 中的已执行成交(回测口径,
     独立字段, 仅含实际成交, 保留近200条——trades_recent混排skip密度过高装不下成交)
  2) output/history.json 实盘账本已结算记录(实盘口径)

用法: python -m tools.monitor_retro
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config          # noqa: E402
from src.monitor import compute_monitor     # noqa: E402


def retro_series(name: str, rets: list[float], labels: list[str], mon_cfg: dict) -> None:
    window = int(mon_cfg.get("window", 20))
    print(f"\n=== {name} (共{len(rets)}笔) ===")
    if not rets:
        print("无已结算成交, 跳过")
        return
    prev_level = None
    for i in range(len(rets)):
        level = compute_monitor(rets[: i + 1], mon_cfg)["level"]
        if level != prev_level:
            upto = min(i, window - 1)
            seg = rets[max(0, i - upto): i + 1]
            mean = sum(seg) / len(seg)
            print(f"  第{i + 1:>3}笔({labels[i]}): {level} | 窗口均值{mean:+.2f}%")
            prev_level = level
    m = compute_monitor(rets, mon_cfg)
    print(f"  终态: {m['level']} | 近{m['n']}笔均值{m['mean_pct']}%/胜率{m['win_pct']}%")


def main() -> None:
    cfg = load_config()
    mon_cfg = cfg.get("monitor") or {"window": 20}

    bt_path = ROOT / "output" / "backtest.json"
    if bt_path.exists():
        bt = json.loads(bt_path.read_text(encoding="utf-8"))
        tr = [t for t in bt.get("trades_executed_recent") or bt.get("trades_recent", [])
              if t.get("ret_pct") is not None and "skip" not in str(t.get("action", ""))]
        retro_series("回测成交序列(backtest.json trades_executed_recent)",
                     [float(t["ret_pct"]) for t in tr],
                     [str(t.get("sell_date", "")) for t in tr], mon_cfg)

    hist_path = ROOT / "output" / "history.json"
    if hist_path.exists():
        h = json.loads(hist_path.read_text(encoding="utf-8"))
        recs = h.get("records", []) if isinstance(h, dict) else h
        done = [r for r in recs if r.get("status") == "已结算"
                and r.get("realized") and r.get("regime_ok", True) is not False]
        # history 新记录在前, 反转为旧→新
        done = list(reversed(done))
        retro_series("实盘账本序列(history.json 已结算部分)",
                     [float(r["realized"]["avg_ret_pct"]) for r in done],
                     [str(r.get("realized", {}).get("sell_date", "")) for r in done], mon_cfg)


if __name__ == "__main__":
    main()
