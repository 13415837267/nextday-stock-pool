# -*- coding: utf-8 -*-
"""第 193 轮(续): 把"排序"和"进场价"拆开 —— 同口径下的双因素分解。

上一支实验(tools/exp_rank_193.py)的两块数字对不上:
  · A) 次日开盘买、持 10 个交易日: 低波池第 1~6 名都只有 +0.6~0.7%/笔(净);
  · 引擎: 单笔 +2.546%(净), 中位持有 6 日。
差 1.9pp。已知两个差异源: ① 进场价(信号日收盘 −1% 限价, 3 日内有效, 第 179 轮);
② 提前离场(移动止损 8% / 持满 10 日, 中位只持 6 日)。本轮把 ① 单独拎出来测,
用同一批票、同一个 10 日持有期, 只换进场规则:

  open  : T+1 开盘无条件买(缺口 −3%~+5% 之外放弃)
  limit : T+1 起挂 信号日收盘 ×0.99, 3 个交易日内 开盘已低于限价按开盘, 盘中触及按限价,
          到期未触及放弃(与引擎 satellite_backtest.py 第 179 轮实现同一套规则)

同时给出第 1/2/3 名在两种进场下各自的收益 —— 排序效应与执行效应分开记账。

用法: python3 tools/exp_entry_193.py [days]
输出: output/exp_entry_193.json
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context                      # noqa: E402
from src.config import load_config                            # noqa: E402
from src.satellite import (daily_params, entry_ok_series, rank_pool,   # noqa: E402
                           rank_qualified)

RANKS = (1, 2, 3)
HOLD = 10
COST = 0.003
PB = 1.0          # 限价 = 信号日收盘 ×(1−1%)
PB_DAYS = 3       # 3 个交易日内有效
GAP_MIN, GAP_MAX = -3.0, 5.0


def _nw_t(x: np.ndarray, lag: int = HOLD) -> float:
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 10:
        return float("nan")
    m, e = x.mean(), x - x.mean()
    var = float((e * e).sum() / n)
    for L in range(1, min(lag, n - 1) + 1):
        var += 2.0 * (1.0 - L / (lag + 1.0)) * float((e[L:] * e[:-L]).sum() / n)
    return m / math.sqrt(var / n) if var > 0 else float("nan")


def _fill(v, d: str, limit_adj: float | None):
    """单日撮合。返回 (成交价_复权, 是否已判死)。"""
    r = v.loc[d]
    if bool(r["is_paused"] == 1) or float(r["vol"]) <= 0:
        return None, False
    op = float(r["open"])
    if op >= float(r["high_limit"]) - 1e-9:
        return None, True                      # 一字涨停, 买不到
    ratio = float(r["open_adj"]) / op if op else 1.0
    o_adj, low_adj = float(r["open_adj"]), float(r["low"]) * ratio
    gap = (op / float(r["pre_close"]) - 1) * 100 if float(r["pre_close"]) else 0.0
    if gap < GAP_MIN or gap > GAP_MAX:
        return None, True                      # 缺口越界 -> 撤单
    if limit_adj is None or o_adj <= limit_adj + 1e-9:
        return o_adj, True                     # 按开盘价成交
    if low_adj <= limit_adj + 1e-9:
        return limit_adj, True                 # 盘中回踩触及限价
    return None, False                         # 当日未触及, 次日继续


def main() -> None:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    cfg = load_config()
    ctx = prepare_context(cfg, days)
    dates, by_date, per_code = ctx["main_dates"], ctx["by_date"], ctx["per_code"]
    P = daily_params(cfg)
    eok = entry_ok_series(ctx.get("reg"), cfg)
    print(f"窗口 {len(dates)} 日 {dates[0]} ~ {dates[-1]}", flush=True)

    recs = []
    t0 = time.time()
    for i in range(len(dates) - HOLD - PB_DAYS - 2):
        day = by_date[dates[i]]
        rk = rank_pool(day, cfg)
        if len(rk) < P["min_pool"]:
            continue
        q = rank_qualified(day, cfg, rk)
        if not len(q):
            continue
        tops = q.head(max(RANKS))
        gate = bool(eok.get(dates[i], False))
        for k in RANKS:
            if k > len(tops):
                break
            row = tops.iloc[k - 1]
            code = str(row["ts_code"])
            v = per_code.get(code)
            if v is None:
                continue
            lim = float(row["close"]) * (1 - PB / 100.0) * (float(row["close_adj"]) / float(row["close"])) \
                if float(row["close"]) else None
            # ---- open: T+1 开盘无条件(缺口过滤) ----
            d1 = dates[i + 1]
            if d1 in v.index:
                px, dead = _fill(v, d1, None)
                if px is not None:
                    j = i + 1 + HOLD
                    if j < len(dates) and dates[j] in v.index:
                        c = float(v.loc[dates[j]]["close_adj"])
                        if c > 0:
                            recs.append({"date": dates[i], "rank": k, "mode": "open", "gate": gate,
                                 "ret": (c / px - 1.0) - COST})
            # ---- limit: 3 日内回踩 -1% ----
            for dj in range(1, PB_DAYS + 1):
                dd = dates[i + dj]
                if dd not in v.index:
                    break
                px, dead = _fill(v, dd, lim)
                if px is not None:
                    j = i + dj + HOLD
                    if j < len(dates) and dates[j] in v.index:
                        c = float(v.loc[dates[j]]["close_adj"])
                        if c > 0:
                            recs.append({"date": dates[i], "rank": k, "mode": "limit", "gate": gate,
                                         "ret": (c / px - 1.0) - COST})
                    break
                if dead:
                    break
        if (i % 300) == 0:
            print(f"  {i}/{len(dates)} {dates[i]} {time.time() - t0:.0f}s", flush=True)

    df = pd.DataFrame(recs)
    rows = []
    for mode in ("open", "limit"):
      for gflag in (False, True):
        for k in RANKS:
            s = df[(df["mode"] == mode) & (df["rank"] == k) & (df["gate"] == gflag)]
            if not len(s):
                continue
            rows.append({"mode": mode, "gate": gflag, "rank": k, "n": int(len(s)),
                         "mean": round(float(s["ret"].mean()) * 100, 3),
                         "median": round(float(s["ret"].median()) * 100, 3),
                         "nw_t": round(_nw_t(s["ret"].to_numpy()), 2),
                         "win": round(float((s["ret"] > 0).mean()) * 100, 1)})
            print(f"  {mode:>5} gate={int(gflag)} rank{k}: n={len(s):>4} 均值 {rows[-1]['mean']:+.3f}% "
                  f"t={rows[-1]['nw_t']:+.2f} 胜率 {rows[-1]['win']:.1f}%", flush=True)

    # 事件研究里"限价单被放弃"的比例: 引擎每笔成交背后有多少天是白等的
    n_day = df[(df["mode"] == "limit") & (df["rank"] == 1)]["date"].nunique()
    n_all = len([d for d in dates[:len(dates) - HOLD - PB_DAYS - 2]])
    out = {"days": len(dates), "window": [dates[0], dates[-1]], "hold": HOLD, "cost": COST,
           "pb": PB, "pb_days": PB_DAYS, "rows": rows,
           "limit_fill_rate_rank1": round(n_day / max(1, n_all) * 100, 1)}
    p = ROOT / "output" / "exp_entry_193.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"限价单第 1 名成交日占比 {out['limit_fill_rate_rank1']}%  写出 {p}", flush=True)


if __name__ == "__main__":
    main()
