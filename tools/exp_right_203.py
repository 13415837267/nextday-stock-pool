# -*- coding: utf-8 -*-
"""第二百零三轮: 右侧追高(主策略的镜像腿)在全部历史上能不能到 100 倍。

背景
----
用户要求两条腿都用全史训练:
  · 左侧抄底 = 主策略 dip(已达标: 3664 日 3486 倍)
  · 右侧追高 = 副策略, 现行 lowvol 只做到 3000 日 23.9 倍(2286%), 离 100 倍差 4.2 倍

本脚本只做一件事: 把"右侧"的几种定义放在同一份全史数据上跑引擎, 用同一个起跑线比较。
  breakout  —— T 日放量长阳且收在 20 日高点附近(追突破)
  limit_up  —— T 日收盘封涨停(追涨停)
  bo_pullback —— 强势股缩量回踩(右侧里的"上车点", 偏左)
  lowvol    —— 现行副策略(对照)

口径: 单吊满仓、T+1 开盘买、持满 N 日收盘卖、往返成本 0.3%、大盘闸门 + 持仓层闸门。
不引入未来函数: 全部信号字段来自 T 日收盘。

用法: python tools/exp_right_203.py 3000 [变体名...]
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from src.backtest import prepare_context          # noqa: E402
from src.config import load_config                # noqa: E402
from src.satellite import daily_params, entry_ok_series  # noqa: E402
from satellite_backtest import run                # noqa: E402

OUT = ROOT / "output"


def base_sat():
    return {
        "mode": "breakout",
        "hold_days": 5,
        "max_pos": 1,
        "buys_per_day": 1,
        "leverage": 1.0,
        "cash_yield": 0.015,
        "cost_pct": 0.3,
        "amount_min_yi": 1.0,
        "min_pool": 20,
        "pool_show": 4,
        "market_gate": True,
        "gate_exit": True,
        "gate_exit_mode": "all",
        "gate_entry_buffer": 0.3,
        "gap_min": -3.0,
        "gap_max": 5.0,
        "stop_pct": 0.0,
        "trail_pct": 0.0,
        "sort_by": "mom20_ivol",
        "min_score": 0,
        "ivol_pct_max": 0.0,
        # breakout 形态
        "bo_pct_min": 3.0,
        "bo_near_high": 2.0,
        "bo_vol_ratio": 1.5,
        "bo_close_pos": 0.7,
        # limit_up
        "lu_amt_max_yi": 0.0,
        "lu_turn_min": 0.0,
        # bo_pullback
        "bp_high_min": -12.0,
        "bp_high_max": -1.0,
        "bp_vol_max": 1.0,
        "bp_mom_min": 0.0,
    }


VARIANTS = {
    # 现行副策略(对照)
    "lowvol_h10": {"mode": "lowvol", "hold_days": 10, "sort_by": "mom20_ivol"},
    # 追突破: 持有期扫描
    "bo_h3": {"mode": "breakout", "hold_days": 3},
    "bo_h5": {"mode": "breakout", "hold_days": 5},
    "bo_h10": {"mode": "breakout", "hold_days": 10},
    # 追涨停
    "lu_h3": {"mode": "limit_up", "hold_days": 3},
    "lu_h5": {"mode": "limit_up", "hold_days": 5},
    # 强势回踩(右侧的上车点)
    "bp_h5": {"mode": "bo_pullback", "hold_days": 5},
    "bp_h10": {"mode": "bo_pullback", "hold_days": 10},
    # 现行 lowvol + 回踩限价(第一百七十九轮: pb=2% 曾把全史年化从 15.1% 拉到 56.8%)
    "pb1_h10": {"mode": "lowvol", "hold_days": 10, "pullback_pct": 1.0, "pullback_days": 1},
    "pb2_h10": {"mode": "lowvol", "hold_days": 10, "pullback_pct": 2.0, "pullback_days": 1},
    "pb3_h10": {"mode": "lowvol", "hold_days": 10, "pullback_pct": 3.0, "pullback_days": 1},
    "pb2_h5": {"mode": "lowvol", "hold_days": 5, "pullback_pct": 2.0, "pullback_days": 1},
    # 突破 + 回踩限价(不在突破日追, 等次日回抽再买)
    "bo_pb2_h5": {"mode": "breakout", "hold_days": 5, "pullback_pct": 2.0, "pullback_days": 1},
    # 杠杆族: 单吊已满仓, 想再上一个数量级只能借钱(融资 5.5%/年)
    "lv15_h10": {"mode": "lowvol", "hold_days": 10, "leverage": 1.5},
    "lv20_h10": {"mode": "lowvol", "hold_days": 10, "leverage": 2.0},
    "lv25_h10": {"mode": "lowvol", "hold_days": 10, "leverage": 2.5},
    "lv20_h5": {"mode": "lowvol", "hold_days": 5, "leverage": 2.0},
    "pb2d3_h10": {"mode": "lowvol", "hold_days": 10, "pullback_pct": 2.0, "pullback_days": 3},
    "pb2d3_lv15": {"mode": "lowvol", "hold_days": 10, "pullback_pct": 2.0,
                   "pullback_days": 3, "leverage": 1.5},
    # 右侧的另一种定义: 上升趋势里的缩量回调(主策略 dip 形态搬到副腿), 持 3/5/10 日
    "dip_h3": {"mode": "dip", "hold_days": 3},
    "dip_h5": {"mode": "dip", "hold_days": 5},
    "dip_h10": {"mode": "dip", "hold_days": 10},
    # 提升型: 只动"什么时候允许来/什么时候必须走", 不动选股
    "gb0": {"mode": "lowvol", "hold_days": 10, "gate_entry_buffer": 0.0},
    "trail6": {"mode": "lowvol", "hold_days": 10, "trail_pct": 6.0},
    "trail10": {"mode": "lowvol", "hold_days": 10, "trail_pct": 10.0},
    "trail12": {"mode": "lowvol", "hold_days": 10, "trail_pct": 12.0},
    "stop8": {"mode": "lowvol", "hold_days": 10, "stop_pct": -8.0},
    "h12": {"mode": "lowvol", "hold_days": 12},
    "h15": {"mode": "lowvol", "hold_days": 15},
    "pos2": {"mode": "lowvol", "hold_days": 10, "max_pos": 2},
    "gb0_lv15": {"mode": "lowvol", "hold_days": 10, "gate_entry_buffer": 0.0, "leverage": 1.5},
}


def main():
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    picks = sys.argv[2:] or list(VARIANTS)
    cfg0 = load_config()
    t0 = time.time()
    ctx = prepare_context(cfg0, days)
    print(f"装载 {len(ctx['main_dates'])} 日 {ctx['main_dates'][0]}~{ctx['main_dates'][-1]} "
          f"用时 {time.time()-t0:.1f}s", flush=True)

    out_file = OUT / f"exp_right_203_{days}.json"
    res_all = json.loads(out_file.read_text(encoding="utf-8")) if out_file.exists() else {}

    for name in picks:
        if name not in VARIANTS:
            print("跳过未知变体", name, flush=True)
            continue
        cfg = copy.deepcopy(cfg0)
        sat = base_sat()
        sat.update(copy.deepcopy(cfg0.get("satellite") or {}))
        sat.update(VARIANTS[name])
        cfg["satellite"] = sat
        t = time.time()
        try:
            r = run(ctx, cfg, entry_ok=entry_ok_series(ctx.get("reg"), cfg))
        except Exception as e:  # noqa: BLE001
            print(f"{name} 失败: {e}", flush=True)
            continue
        m = r["metrics"]
        row = {k: m.get(k) for k in ("总收益率%", "年化收益率%", "最大回撤%", "卡玛比率",
                                     "交易次数", "胜率%", "平均单笔收益%")}
        row["params"] = {k: sat.get(k) for k in ("mode", "hold_days", "sort_by", "bo_pct_min",
                                                 "bo_near_high", "bo_vol_ratio", "bo_close_pos")}
        row["span"] = [ctx["main_dates"][0], ctx["main_dates"][-1]]
        row["secs"] = round(time.time() - t, 1)
        res_all[name] = row
        print(f"{name:12s} 总 {row['总收益率%']:>12.2f}%  年化 {row['年化收益率%']:>6.2f}%  "
              f"回撤 {row['最大回撤%']:>7.2f}%  卡玛 {row['卡玛比率']:>5.2f}  "
              f"笔数 {row['交易次数']:>4d}  单笔 {row['平均单笔收益%']:>6.3f}%  "
              f"({row['secs']}s)", flush=True)
        out_file.write_text(json.dumps(res_all, ensure_ascii=False, indent=1), encoding="utf-8")
    print("写出", out_file, flush=True)


if __name__ == "__main__":
    main()
