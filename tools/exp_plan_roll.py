# -*- coding: utf-8 -*-
"""副策略"每日买 1 只、滚动持有"口径 —— 第一百七十六轮。

背景: 现行 satellite 是"一次买满 4 只、持 10 日"(max_pos=4/buys_per_day=4)。
用户的真实用法是"每天看一只新票、人工决定买不买", 对应的不是 max_pos=1
(那会买卖交替、大量信号被跳过), 而是 **buys_per_day=1 + max_pos=hold_days**
(每天最多开 1 个新仓, 同时最多持 10 只, 每只 1/10 仓位)。

本脚本对比: 现行(4/4) vs 每日 1 只滚动(1/10, 1/5) vs 单吊(1/1)。

用法: python3 tools/exp_plan_roll.py [--days 1500]
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

VARIANTS = [
    ("single_1", 1, 1, "单吊: 最多持 1 只"),
    ("roll_5", 5, 1, "每日 1 只, 持 10 日 -> 同时约 5 只"),
    ("roll_10", 10, 1, "每日 1 只, 持 10 日 -> 同时约 10 只"),
    ("roll_10_b2", 10, 2, "每日 2 只, 同时最多 10 只"),
    ("current_4", 4, 4, "现行: 一次买满 4 只"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=1500)
    args = ap.parse_args()
    cfg0 = load_config()
    print("构建上下文 ...", flush=True)
    ctx = prepare_context(cfg0, args.days)
    print(f"窗口 {ctx['main_dates'][0]} ~ {ctx['main_dates'][-1]}", flush=True)

    out = {}
    for name, mp, bpd, desc in VARIANTS:
        cfg = json.loads(json.dumps(cfg0))
        cfg.setdefault("satellite", {})
        cfg["satellite"]["max_pos"] = mp
        cfg["satellite"]["buys_per_day"] = bpd
        cfg["satellite"]["pool_show"] = max(4, bpd)
        r = sb.run(ctx, cfg)
        m = r["metrics"]
        out[name] = {"desc": desc, "max_pos": mp, "buys_per_day": bpd, **{
            k: m.get(k) for k in ("总收益率%", "年化收益率%", "最大回撤%", "卡玛比率",
                                  "夏普比率", "交易次数", "胜率%", "平均单笔收益%", "空仓天数")}}
        print(f"{name:12s} {desc:32s} 年化 {m['年化收益率%']:+6.2f}%  回撤 {m['最大回撤%']:7.2f}%  "
              f"卡玛 {m['卡玛比率']:5.2f}  笔数 {m['交易次数']:3d}  胜率 {m['胜率%']:.1f}%  "
              f"单笔 {m['平均单笔收益%']:+.3f}%", flush=True)
    p = ROOT / "output" / "exp_plan_roll.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("写出", p, flush=True)


if __name__ == "__main__":
    main()
