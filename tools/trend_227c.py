"""第227轮C · 趋势跟随 alpha 源（与现行隔日反转正交）

为什么做这个：现行策略 alpha 寿命只有 1 天（顺延任何天数 −66~−103pp），
一年只在场 43.7%，剩下的时间资金闲置。趋势跟随的时间尺度是 5~20 天，
与隔日反转在时间上互补，是唯一可能叠加的 alpha 源。

规则（全部可执行、T+1 合规、无权限门槛）：
  入选(T日收盘):
    1) 主板 / 非ST / 非停牌 / 价格 3~150 / 成交额 > 0.8 亿
    2) 突破: close > 前 N 日最高价(不含当日)
    3) 趋势: close > MA20 > MA60
    4) 放量: vol > 1.2 × vol20
  买入: T+1 开盘, 等权, 最多同时持 K 只
  卖出(任一触发,当日收盘价):
    a) close < MA20      b) 持有满 M 个交易日
    c) 相对成本亏 8%      d) 相对成本盈 20% 后回撤 5pp(让利润奔跑)
  成本: 佣金万3 + 印花税千1(卖) + 滑点千1

输出: 年化/回撤/卡玛/在场率/换手, 以及与现行反转策略日收益的相关系数。
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "w227c_trend.csv"

COLS = ["ts_code", "trade_date", "open", "high", "low", "close",
        "pre_close", "vol", "amount", "turnover_rate", "is_paused", "is_st"]


def load_panel(days: int) -> pd.DataFrame:
    fs = sorted(glob.glob(str(ROOT / "data/panel/*.csv.gz")))[-(days + 90):]
    parts = [pd.read_csv(f, usecols=COLS) for f in fs]
    d = pd.concat(parts, ignore_index=True)
    d["trade_date"] = d["trade_date"].astype(int)
    d = d.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    return d


def add_ind(d: pd.DataFrame) -> pd.DataFrame:
    g = d.groupby("ts_code", observed=True, sort=False)
    d["ma20"] = g["close"].transform(lambda s: s.rolling(20).mean())
    d["ma60"] = g["close"].transform(lambda s: s.rolling(60).mean())
    d["vol20"] = g["vol"].transform(lambda s: s.rolling(20).mean())
    for n in (20, 60):
        d[f"hh{n}"] = g["high"].transform(lambda s: s.rolling(n).max().shift(1))
    d["amount_yi"] = d["amount"] / 1e8
    return d


def backtest(d: pd.DataFrame, N: int, M: int, K: int,
             stop_loss: float = 0.08, trail: tuple[float, float] = (0.20, 0.05)) -> dict:
    dates = np.sort(d["trade_date"].unique())
    by_date = {t: g for t, g in d.groupby("trade_date", observed=True)}
    # 开仓信号(昨日收盘产生, 今日开盘执行)
    sig = d[(d["hh" + str(N)].notna()) & (d["close"] > d["hh" + str(N)])
            & (d["close"] > d["ma20"]) & (d["ma20"] > d["ma60"])
            & (d["vol"] > 1.2 * d["vol20"])
            & (d["amount_yi"] > 0.8) & (d["close"].between(3, 150))
            & (~d["is_st"].astype(bool)) & (~d["is_paused"].astype(bool))
            & (~d["ts_code"].str.startswith(("300", "688", "8", "4")))]
    sig_by_date = {t: g["ts_code"].tolist() for t, g in sig.groupby("trade_date", observed=True)}

    cash = 1.0
    pos: dict[str, dict] = {}   # code -> {shares, cost, entry_i, peak}
    equity = []
    holding_days = 0
    n_trades = 0

    for i, t in enumerate(dates):
        g = by_date[t].set_index("ts_code")
        # ---- 1. 开盘: 处理昨日信号 ----
        if i > 0:
            picks = sig_by_date.get(dates[i - 1], [])
            picks = [c for c in picks if c not in pos and c in g.index][: max(0, K - len(pos))]
            if picks and cash > 0.02:
                alloc = cash / len(picks)
                for c in picks:
                    px = float(g.at[c, "open"])
                    if px <= 0 or np.isnan(px):
                        continue
                    sh = alloc / (px * (1 + 0.001))     # 滑点千1
                    cash -= sh * px * (1 + 0.0003)      # 佣金万3
                    pos[c] = {"sh": sh, "cost": px, "entry_i": i, "peak": px, "last": px}
                    n_trades += 1
        # ---- 2. 收盘: 卖出判断 ----
        for c in list(pos):
            # 停牌/退市: 无当日价, 沿用上一有效价, 既不卖也不把市值算 0
            if c not in g.index:
                continue
            px = float(g.at[c, "close"])
            if np.isnan(px) or px <= 0:
                continue
            p = pos[c]
            p["last"] = px
            p["peak"] = max(p["peak"], px)
            ret = px / p["cost"] - 1
            held = i - p["entry_i"]
            ma20 = g.at[c, "ma20"]
            sell = False
            if not np.isnan(ma20) and px < float(ma20):
                sell = True
            elif held >= M:
                sell = True
            elif ret <= -stop_loss:
                sell = True
            elif p["peak"] / p["cost"] - 1 >= trail[0] and px <= p["peak"] * (1 - trail[1]):
                sell = True
            if sell:
                cash += p["sh"] * px * (1 - 0.0003 - 0.001)
                del pos[c]
        # ---- 3. 净值 ----
        mv = sum(p["sh"] * p["last"] for p in pos.values())
        equity.append(cash + mv)
        if pos:
            holding_days += 1

    eq = pd.Series(equity, index=pd.Index(dates, name="trade_date"), name="nav")
    return summarize(eq, n_trades, holding_days, len(dates)), eq


def summarize(eq: pd.Series, n_trades: int, hold_days: int, n: int) -> dict:
    yrs = max(0.3, n / 244.0)
    ann = eq.iloc[-1] ** (1 / yrs) - 1
    mdd = float((eq / eq.cummax() - 1).min())
    r = eq.pct_change().dropna()
    return {
        "年化": round(ann * 100, 2),
        "回撤": round(mdd * 100, 2),
        "卡玛": round(ann / abs(mdd) if mdd else 0, 2),
        "夏普": round(r.mean() / r.std() * np.sqrt(244), 2) if r.std() else 0,
        "笔数": n_trades,
        "在场率": round(hold_days / n * 100, 1),
        "总收益": round((eq.iloc[-1] - 1) * 100, 1),
    }


def main() -> None:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    t0 = time.time()
    d = load_panel(days)
    d = add_ind(d)
    dd = np.sort(d["trade_date"].unique())[-days:]
    d = d[d["trade_date"] >= dd[0]].reset_index(drop=True)
    print(f"面板 {len(d):,} 行 / {d.ts_code.nunique()} 只 / {len(dd)} 日  ({time.time()-t0:.0f}s)\n",
          flush=True)

    # 现行反转策略日收益, 用于相关性
    rev = None
    bt = ROOT / "output" / "backtest.json"
    if bt.exists():
        e = pd.DataFrame(json.load(open(bt))["equity"])
        e["d"] = e["d"].str.replace("-", "").astype(int)
        rev = e.set_index("d")["nav"].pct_change()

    rows = []
    for N in (20, 60):
        for M in (5, 10, 20):
            for K in (5, 10):
                t = time.time()
                m, eq = backtest(d, N, M, K)
                corr = None
                if rev is not None:
                    j = pd.concat([eq.pct_change().rename("trend"),
                                   rev.rename("rev")], axis=1, join="inner").dropna()
                    corr = round(float(j["trend"].corr(j["rev"])), 3) if len(j) > 100 else None
                rows.append({"N": N, "M": M, "K": K, **m, "与反转相关": corr})
                pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8-sig")
                print(f"N={N:<3} M={M:<3} K={K:<3} 年化{m['年化']:7.2f}% 回撤{m['回撤']:7.2f}% "
                      f"卡玛{m['卡玛']:5.2f} 在场{m['在场率']:5.1f}% {m['笔数']:5d}笔 "
                      f"相关{corr if corr is not None else '-':>6} ({time.time()-t:.0f}s)", flush=True)

    print()
    df = pd.DataFrame(rows).sort_values("卡玛", ascending=False)
    print(df.to_string(index=False))
    print(f"\n最优(按卡玛): {df.iloc[0][['N','M','K']].to_dict()} "
          f"年化{df.iloc[0]['年化']}% 回撤{df.iloc[0]['回撤']}% 卡玛{df.iloc[0]['卡玛']}")


if __name__ == "__main__":
    main()
