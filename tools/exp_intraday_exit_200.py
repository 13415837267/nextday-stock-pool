# -*- coding: utf-8 -*-
"""第200轮 · T+2 日内退出规则（分钟级）

现行: T+2 14:55 一次性卖出。
T+2 日内振幅很大（最高点平均比收盘高 2.46%、最低点低 2.58%），
所以"能不能在冲高回落时先走"是个真问题 —— 不是靠直觉，靠路径模拟。

规则族（都在 T+2 日内的 1 分钟序列上跑）:
  1. 移动止盈: 从日内 running high 回落 x% 即卖（x = 0.5~3%）
  2. 固定止盈: 相对 T+1 买入价（= T+2 开盘价附近）涨 y% 即卖
  3. 回落 + 止盈组合
基准: 14:55 收盘卖（= 回测口径）。

成交假设: 触发那一分钟的 close 成交（保守，实际滑点更差）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MINS = ROOT / "output" / "mins_200"


def load_day(code: str, d: str) -> pd.DataFrame | None:
    p = MINS / f"{code}_{d}.csv.gz"
    if not p.exists():
        return None
    x = pd.read_csv(p)
    if not len(x):
        return None
    x["t"] = x.trade_time.astype(str).str[-4:].astype(int)
    return x.sort_values("t").reset_index(drop=True)


def exits(x: pd.DataFrame, trail: float | None, take: float | None,
          take_ref: float) -> tuple[float, int]:
    """返回 (卖出价, 触发分钟序号; 未触发返回 (14:55 收盘, -1))。"""
    run_hi = float(x.open.iloc[0])
    for i, r in x.iterrows():
        c = float(r.close)
        run_hi = max(run_hi, c)
        if take is not None and c >= take_ref * (1 + take):
            return c, int(i)
        if trail is not None and c <= run_hi * (1 - trail):
            return c, int(i)
    s = x[x.t <= 1455]
    return float(s.close.iloc[-1]), -1


def main():
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    df = pd.read_csv(ROOT / "output" / f"exp_mins_200_{days}.csv")
    rows = []
    for _, r in df.iterrows():
        x = load_day(r.code, r.sell)
        if x is None:
            continue
        rows.append((x, float(r.ret), float(r.b_open)))
    print(f"{len(rows)} 笔")

    def run(trail, take):
        mult, hits = [], []
        for x, ret, bopen in rows:
            px, i = exits(x, trail, take, bopen)
            ref = float(x[x.t <= 1455].close.iloc[-1])
            mult.append((1 + ret / 100) * (px / ref))
            hits.append(i)
        m = np.array(mult)
        geo = float(np.prod(m) ** (1 / len(m)) - 1)
        return {"几何%": round(geo * 100, 3), "总倍数": round(float(np.prod(m)), 1),
                "均值%": round(float((m - 1).mean() * 100), 3),
                "触发率%": round(float(np.mean(np.array(hits) >= 0) * 100), 1),
                "首小时触发%": round(float(np.mean((np.array(hits) >= 0) & (np.array(hits) < 90)) * 100), 1)}

    base = run(None, None)
    print("基准（14:55 收盘卖）:", base)
    out = {"base": base, "trail": {}, "take": {}, "combo": {}}
    print("\n-- 移动止盈（从日内高点回落 x%）--")
    for x in (0.005, 0.008, 0.010, 0.015, 0.020, 0.025, 0.030):
        out["trail"][str(x)] = r = run(x, None)
        print(f"  回落 {x*100:>4.1f}%: {r}")
    print("\n-- 固定止盈（相对 T+1 买入价 +y%）--")
    for y in (0.02, 0.03, 0.05, 0.07, 0.10):
        out["take"][str(y)] = r = run(None, y)
        print(f"  止盈 {y*100:>4.1f}%: {r}")
    print("\n-- 组合（止盈 5% + 回落 1.5%）--")
    out["combo"]["t5_r15"] = r = run(0.015, 0.05)
    print("  ", r)

    (ROOT / "output" / f"exp_intraday_exit_200_{days}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
