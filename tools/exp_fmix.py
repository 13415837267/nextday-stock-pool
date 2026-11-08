# -*- coding: utf-8 -*-
"""第一百七十二轮: 因子组合实验室 —— 在"已证 alpha 的形态池"内做因子分层筛选。

问题
----
前 171 轮把 OHLCV 单因子几乎扫遍(第 163/171 轮换池子全否), 但所有测试都停在
"因子的绝对值 + 单一周期"上。三类组合从未系统测过:
  (1) 横截面分位化: vol_ratio<=0.8 是绝对阈值, 全市场缩量日人人达标;
      改成"当日分位<=20%"会随市场状态自适应;
  (2) 二阶结构: 量能趋势(5日/20日)、振幅扩张、动量加速度 —— 都是"变化的变化";
  (3) 形态结构: 下影占比、上影占比、连续下跌天数 —— 日内路径而非幅度。

设计
----
不去全市场扫(第 171 轮证明脱离形态池全是噪声), 而是锚定已验证的 dip 形态,
**放宽到可交易规模**(pct -9~-1 / vol_ratio<=1.5), 得到 n≈数万的事件池,
在池内按每个因子分 5 档, 看 T+1 开盘买 → T+2 开盘卖(主策略口径)的分层能力。

口径
----
· 信号日 T 收盘出信号(全部因子仅用截至 T 日收盘可知字段)
· T+1 开盘买; 一字涨停/停牌/无下一交易日则剔除
· T+2 开盘卖; 一字跌停/停牌则剔除
· 复权价撮合; 往返成本 0.3%; 超额 = 单笔 - 同期市场(mkt_pct 复合)
· 基础池: 主板 / 非ST / 非停牌 / bars>=250 / 成交额>=1亿
· 分两组输出: 全样本 vs 闸门开(trade_ok) —— 闸门的影响必须并排看

用法: python tools/exp_fmix.py [天窗口, 默认 1500]
输出: output/exp_fmix.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.backtest import fetch_panel                                              # noqa: E402
from src.config import load_config                                                # noqa: E402
from src.selector import apply_regime, build_indicators, regime_base              # noqa: E402
from src.data_api import ZZClient                                                 # noqa: E402

OUT = os.path.join(ROOT, "output")
COST = 0.003
EPS = 1e-9


def prepare(cfg: dict, days: int) -> pd.DataFrame:
    """拉面板 + 指标 + 本轮新增的二阶/结构因子, 返回带前向收益的长表。"""
    bt = cfg["backtest"]
    warmup = bt["warmup_days"]
    client = ZZClient(cfg)
    all_days = client.trade_days(days=days + warmup + 5)
    dates = all_days[-(days + warmup):]
    print(f"[1/3] 面板 {len(dates)} 日 ...", flush=True)
    panel = fetch_panel(client, dates, float32=True)
    panel = build_indicators(panel)
    print(f"[2/3] 指标完成 {len(panel):,} 行, 构造新因子 ...", flush=True)

    g = panel.groupby("ts_code", sort=False)
    ca = panel["close_adj"]
    # ---- 二阶结构: 量能趋势 / 振幅扩张 / 动量加速度
    panel["vol_ma20"] = g["vol"].transform(lambda s: s.rolling(20, min_periods=15).mean())
    panel["vol_ma5"] = g["vol"].transform(lambda s: s.rolling(5, min_periods=4).mean())
    panel["amp_ma20"] = g["amp_pct"].transform(lambda s: s.rolling(20, min_periods=15).mean())
    panel["vol_trend"] = panel["vol_ma5"] / panel["vol_ma20"].replace(0, np.nan)   # 5日量 / 20日量
    panel["amp_ratio"] = panel["amp_pct"] / panel["amp_ma20"].replace(0, np.nan)   # 振幅扩张倍数
    panel["accel"] = panel["mom3"] - panel["mom20"] / 20.0 * 3.0                   # 短端相对中端提速
    # ---- 形态结构: 上影占比 / 下影占比 / 连续下跌
    body_hi = panel[["open", "close"]].max(axis=1)
    panel["upper_tail"] = (panel["high"] - body_hi) / panel["pre_close"].replace(0, np.nan) * 100
    panel["tail_share"] = panel["low_tail"] / (panel["amp_pct"] + 0.05)            # 下影占振幅
    panel["up_share"] = panel["upper_tail"] / (panel["amp_pct"] + 0.05)            # 上影占振幅
    neg = panel["pct"] < 0
    n1 = neg.groupby(panel["ts_code"]).shift(1).fillna(False).astype(bool)
    n2 = neg.groupby(panel["ts_code"]).shift(2).fillna(False).astype(bool)
    panel["down3"] = (neg & n1 & n2).astype("int8")
    # ---- 当日自身跳空(与入场缺口不同: 这是 T 日的)
    panel["gap_t"] = (panel["open"] / panel["pre_close"].replace(0, np.nan) - 1) * 100

    # ---- 前向收益: T+1 开盘买, T+2 开盘卖
    p1 = "buy_"
    for c in ("open", "open_adj", "high_limit", "is_paused", "trade_date"):
        panel[p1 + c] = g[c].shift(-1)
    p2 = "sell_"
    for c in ("open_adj", "low_limit", "is_paused"):
        panel[p2 + c] = g[c].shift(-2)
    panel["mkt1"] = g["mkt_pct"].shift(-1)
    panel["mkt2"] = g["mkt_pct"].shift(-2)

    buy = panel["buy_open_adj"]
    sell = panel["sell_open_adj"]
    # T+1 必须是紧邻的下一交易日(防停牌断档后误取更远日期)
    nxt_ok = panel["buy_trade_date"].notna() & (panel["buy_is_paused"] == 0)
    # 一字涨停买不到: **未复权** T+1 开盘价贴着 T+1 涨停价(复权价与 high_limit 不同口径)
    limit_up = panel["buy_open"] >= panel["buy_high_limit"] * 0.999
    panel["buy_ok"] = (nxt_ok.to_numpy()
                       & (panel["sell_is_paused"] == 0).to_numpy()
                       & buy.notna().to_numpy() & (buy > 0).to_numpy()
                       & sell.notna().to_numpy() & (sell > 0).to_numpy()
                       & ~limit_up.fillna(False).to_numpy())
    panel["ret"] = (sell / buy - 1 - COST).to_numpy(dtype="float64")
    panel["mkt_ret"] = ((1 + panel["mkt1"]) * (1 + panel["mkt2"]) - 1).to_numpy(dtype="float64")
    panel["excess"] = panel["ret"] - panel["mkt_ret"]
    print(f"[3/3] 前向收益完成", flush=True)
    return panel


FACTORS = [
    # 对照: 主策略已在用的
    ("vol_ratio", "量比(单日缩量)", 1),
    ("close_pos", "收盘位置", 1),
    ("amp_pct", "振幅%", 1),
    ("low_tail", "下影线%", 1),
    ("mom20", "20日动量", 1),
    ("mom3", "3日动量", 1),
    ("dist_ma5", "距5日线%", 1),
    ("dist_high20", "距20日高%", 1),
    ("ivol20", "特质波动", 1),
    ("turnover_rate", "换手率", 1),
    # 本轮新增: 二阶结构
    ("vol_trend", "量能趋势5/20", 1),
    ("amp_ratio", "振幅扩张", 1),
    ("accel", "动量加速度", 1),
    # 本轮新增: 形态结构
    ("tail_share", "下影占振幅", 1),
    ("up_share", "上影占振幅", 1),
    ("down3", "连跌3日", 1),
    ("gap_t", "当日跳空%", 1),
    # 本轮新增: 横截面分位(当日全池 rank)
    ("vr_q", "量比分位", 1),
    ("turn_q", "换手分位", 1),
    ("amp_q", "振幅分位", 1),
    ("ivol_q", "特质波动分位", 1),
    ("rs20", "20日相对强度", 1),
]


def add_cross_rank(panel: pd.DataFrame) -> pd.DataFrame:
    """当日全市场分位(0~1): 让门槛随市场状态自适应。"""
    by = panel.groupby("trade_date", sort=False)
    panel["vr_q"] = by["vol_ratio"].rank(pct=True).astype("float32")
    panel["turn_q"] = by["turnover_rate"].rank(pct=True).astype("float32")
    panel["amp_q"] = by["amp_pct"].rank(pct=True).astype("float32")
    panel["ivol_q"] = by["ivol20"].rank(pct=True).astype("float32")
    med = by["mom20"].transform("median")
    panel["rs20"] = (panel["mom20"] - med).astype("float32")
    return panel


def bucket_table(df: pd.DataFrame, col: str, nq: int = 5) -> dict:
    """按因子分档(等频)统计净收益与胜率, 同时给秩相关。"""
    s = df[col].replace([np.inf, -np.inf], np.nan)
    ok = s.notna()
    if ok.sum() < 200:
        return {"n": int(ok.sum()), "skip": True}
    sub = df.loc[ok]
    q = pd.qcut(sub[col], nq, labels=False, duplicates="drop")
    rows = []
    for k in range(int(q.max()) + 1):
        m = q == k
        if m.sum() < 30:
            continue
        r = sub.loc[m, "ret"]
        e = sub.loc[m, "excess"]
        rows.append({"q": int(k) + 1, "n": int(m.sum()),
                     "ret": round(float(r.mean()) * 100, 3),
                     "win": round(float((r > 0).mean()) * 100, 1),
                     "excess": round(float(e.mean()) * 100, 3)})
    # Spearman(秩相关): 避免 scipy 依赖
    ic = float(sub[col].rank().corr(sub["ret"].rank()))
    # 单调性: 各档收益与档序的秩相关
    if len(rows) >= 3:
        a = pd.Series([r["q"] for r in rows]).rank()
        b = pd.Series([r["ret"] for r in rows]).rank()
        mono = float(a.corr(b))
    else:
        mono = float("nan")
    top = max(rows, key=lambda x: x["ret"]) if rows else None
    bottom = min(rows, key=lambda x: x["ret"]) if rows else None
    return {"n": int(ok.sum()), "ic": round(ic, 4), "mono": round(mono, 3),
            "buckets": rows,
            "spread": round(top["ret"] - bottom["ret"], 3) if top and bottom else None}


def main(days: int = 1500) -> None:
    cfg = load_config()
    panel = prepare(cfg, days)
    panel = add_cross_rank(panel)

    # ---- 基础池 + 放宽 dip 形态
    base = ((panel["bars"] >= 250).to_numpy()
            & (panel["is_st"] == 0).to_numpy()
            & (panel["is_paused"] == 0).to_numpy()
            & (panel["amount"] >= 1e8).to_numpy()
            & panel["buy_ok"].to_numpy()
            & panel["ret"].between(-0.30, 0.30).to_numpy())     # 剔除复权断点脏值
    pool = (base
            & panel["pct"].between(-9.0, -1.0).to_numpy()
            & (panel["vol_ratio"] <= 1.5).to_numpy())
    df = panel.loc[pool].copy()
    print(f"\n事件池: {len(df):,} 笔 / {df['trade_date'].nunique()} 日 "
          f"= {len(df) / max(1, df['trade_date'].nunique()):.2f} 笔/日", flush=True)
    print(f"池内 T+1开盘买→T+2开盘卖 净收益: {df['ret'].mean() * 100:+.3f}%  "
          f"胜率 {(df['ret'] > 0).mean() * 100:.1f}%", flush=True)

    # ---- 闸门分组
    reg = apply_regime(regime_base(panel, cfg), cfg)
    ok = reg["trade_ok"].astype(bool)
    gate = df["trade_date"].map(ok).fillna(True).astype(bool)
    df_g = df.loc[gate.to_numpy()]
    print(f"闸门开: {len(df_g):,} 笔 ({len(df_g) / max(1, len(df)) * 100:.0f}%), "
          f"单笔 {df_g['ret'].mean() * 100:+.3f}%", flush=True)

    out = {"window": {"days": days, "start": str(df["trade_date"].min()),
                      "end": str(df["trade_date"].max())},
           "pool": {"n": int(len(df)), "per_day": round(len(df) / df["trade_date"].nunique(), 2),
                    "ret_pct": round(float(df["ret"].mean()) * 100, 3),
                    "win_pct": round(float((df["ret"] > 0).mean()) * 100, 1)},
           "factors": {}, "grid": {}}

    # ---- 单因子分层(两组各跑一遍)
    print("\n%-14s %-16s | %6s %7s %7s %7s | %6s %7s"
          % ("因子", "含义", "n", "IC", "Q1%", "Q5%", "极差", "单调"))
    print("-" * 88)
    for col, cn, _ in FACTORS:
        for tag, sub in (("all", df), ("gate", df_g)):
            t = bucket_table(sub, col)
            out["factors"].setdefault(col, {"cn": cn})[tag] = t
        ta = out["factors"][col]["all"]
        if ta.get("skip"):
            print("%-14s %-16s | %6d  样本不足" % (col, cn, ta["n"]))
            continue
        q1 = ta["buckets"][0]["ret"]
        q5 = ta["buckets"][-1]["ret"]
        print("%-14s %-16s | %6d %7.3f %7.3f %7.3f %7.2f | %6.2f %7s"
              % (col, cn, ta["n"], ta["ic"], q1, q5, ta["spread"], ta["mono"],
                 "★" if abs(ta["mono"]) >= 0.9 else ""))

    # ---- 交互网格: 取 |单调性| 最高的两个因子做 3x3
    cand = [c for c, _, _ in FACTORS
            if not out["factors"][c]["all"].get("skip")]
    cand.sort(key=lambda c: -abs(out["factors"][c]["all"]["mono"]))
    picks = cand[:2]
    if len(picks) == 2:
        a, b = picks
        print(f"\n=== 交互网格 {a} × {b} (3×3, 格内单笔净收益%) ===", flush=True)
        for tag, sub in (("all", df), ("gate", df_g)):
            lo_a, hi_a = sub[a].quantile([0.33, 0.67])
            lo_b, hi_b = sub[b].quantile([0.33, 0.67])
            ia = np.where(sub[a] <= lo_a, 0, np.where(sub[a] <= hi_a, 1, 2))
            ib = np.where(sub[b] <= lo_b, 0, np.where(sub[b] <= hi_b, 1, 2))
            grid = np.full((3, 3), np.nan)
            cnt = np.zeros((3, 3), int)
            for i in range(3):
                for j in range(3):
                    m = (ia == i) & (ib == j)
                    cnt[i, j] = int(m.sum())
                    if m.sum() >= 30:
                        grid[i, j] = float(sub["ret"].to_numpy()[m].mean()) * 100
            print(f"[{tag}] " + "（行=" + a + " 低→高, 列=" + b + " 低→高）")
            for i in range(3):
                print("   " + " ".join(("%8.3f" % grid[i, j]) if not np.isnan(grid[i, j])
                                       else "     n/a" for j in range(3))
                      + "   n=" + "/".join(str(int(cnt[i, j])) for j in range(3)))
            out["grid"][tag] = {"a": a, "b": b,
                                "ret": [[None if np.isnan(grid[i, j]) else round(grid[i, j], 3)
                                         for j in range(3)] for i in range(3)],
                                "n": cnt.tolist()}

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "exp_fmix.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\nsaved output/exp_fmix.json")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1500)
