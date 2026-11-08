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
    # ---- VWAP 族(本轮新信息): 当日成交均价 = 成交额/成交量。
    #   数据源 fields=all 确实返回 avg_price, 但 panel 缓存只存 16 列(_KEEP)没有它;
    #   实测 amount/vol 与 avg_price 完全一致(1812492376.93/150481965 = 12.0446),
    #   故无需重拉历史, 直接从已缓存字段反推。
    #   与 close_pos 的区别: close_pos 用当日最高最低(极值), vwap_dev 用成交量加权均价,
    #   后者才是"全体成交者的成本线" —— 收盘在均价上方=尾盘承接, 下方=尾盘出货。
    _av = panel["amount"] / panel["vol"].replace(0, np.nan)
    panel["vwap"] = _av
    panel["vwap_dev"] = (panel["close"] / _av.replace(0, np.nan) - 1) * 100          # 收盘相对均价%
    panel["vwap_gap"] = (_av / panel["pre_close"].replace(0, np.nan) - 1) * 100      # 均价相对昨收%
    panel["vwap_z"] = ((panel["close"] - _av)
                       / (panel["high"] - panel["low"]).replace(0, np.nan))          # 标准化(-1~1)
    # 时序版: 同一只票自己的历史分位(vs 横截面分位), 回答"今天是这只票少见的位置吗"
    _roll = g["vwap_dev"]
    _m60 = _roll.transform(lambda x: x.rolling(60, min_periods=40).mean())
    _s60 = _roll.transform(lambda x: x.rolling(60, min_periods=40).std())
    panel["vwap_ts"] = ((panel["vwap_dev"] - _m60) / _s60.replace(0, np.nan))

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
    # 本轮新增: VWAP 族(收盘相对成交均价)
    ("vwap_dev", "收盘相对VWAP%", 1),
    ("vwap_gap", "VWAP相对昨收%", 1),
    ("vwap_z", "VWAP标准化", 1),
    ("vwap_ts", "VWAP时序z(60日)", 1),
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


def grid_table(sub: pd.DataFrame, a: str, b: str, nq: int = 3) -> dict:
    """a × b 的等频 nq×nq 网格, 输出格内单笔净收益(%)与样本数。"""
    lo_a, hi_a = sub[a].quantile([1 / nq, 2 / nq])
    lo_b, hi_b = sub[b].quantile([1 / nq, 2 / nq])
    ia = np.where(sub[a] <= lo_a, 0, np.where(sub[a] <= hi_a, 1, 2))
    ib = np.where(sub[b] <= lo_b, 0, np.where(sub[b] <= hi_b, 1, 2))
    ret = sub["ret"].to_numpy()
    grid = np.full((nq, nq), np.nan)
    cnt = np.zeros((nq, nq), int)
    for i in range(nq):
        for j in range(nq):
            m = (ia == i) & (ib == j)
            cnt[i, j] = int(m.sum())
            if m.sum() >= 100:
                grid[i, j] = float(ret[m].mean()) * 100
    return {"lit": [[None if np.isnan(grid[i, j]) else round(grid[i, j], 3)
                     for j in range(nq)] for i in range(nq)],
            "n": cnt.tolist()}


def report(sub: pd.DataFrame, title: str, out: dict, tag: str) -> None:
    print(f"\n=== {title} (n={len(sub):,}, 单笔 {sub['ret'].mean() * 100:+.3f}%, "
          f"胜率 {(sub['ret'] > 0).mean() * 100:.1f}%) ===", flush=True)
    print("%-14s %-16s | %7s %7s %7s %7s %5s"
          % ("因子", "含义", "IC", "Q1%", "Q5%", "极差", "单调"))
    print("-" * 78)
    rows = {}
    for col, cn, _ in FACTORS:
        t = bucket_table(sub, col)
        rows[col] = t
        if t.get("skip"):
            print("%-14s %-16s | 样本不足" % (col, cn))
            continue
        q1 = t["buckets"][0]["ret"]
        q5 = t["buckets"][-1]["ret"]
        print("%-14s %-16s | %7.4f %7.3f %7.3f %7.2f %5.2f %s"
              % (col, cn, t["ic"], q1, q5, t["spread"], t["mono"],
                 "★" if abs(t["mono"]) >= 0.9 else ""))
    out["factors"][tag] = rows


def main(days: int = 1500) -> None:
    cfg = load_config()
    panel = prepare(cfg, days)
    panel = add_cross_rank(panel)

    base = ((panel["bars"] >= 250).to_numpy()
            & (panel["is_st"] == 0).to_numpy()
            & (panel["is_paused"] == 0).to_numpy()
            & (panel["amount"] >= 1e8).to_numpy()
            & panel["buy_ok"].to_numpy()
            & panel["ret"].between(-0.30, 0.30).to_numpy())
    # 宽池: 只要求"下跌 + 不显著放量", 用于测因子的纯区分力
    wide = base & panel["pct"].between(-9.0, -1.0).to_numpy() \
        & (panel["vol_ratio"] <= 1.5).to_numpy()
    # 严池: 主策略 dip 形态原样(缩量阴线 + 收在低位), 只是不叠评分与缺口过滤
    strict = base & panel["pct"].between(-6.0, -3.0).to_numpy() \
        & (panel["vol_ratio"] <= 0.8).to_numpy() \
        & (panel["close_pos"] <= 0.4).to_numpy()

    reg = apply_regime(regime_base(panel, cfg), cfg)
    ok = reg["trade_ok"].astype(bool)
    gate = panel["trade_date"].map(ok).fillna(True).astype(bool).to_numpy()

    out = {"window": {"days": days, "start": str(panel["trade_date"].min()),
                      "end": str(panel["trade_date"].max())},
           "pools": {}, "factors": {}, "grid": {}}

    for pname, m in (("wide", wide), ("strict", strict)):
        sub_all = panel.loc[m]
        sub_g = panel.loc[m & gate]
        out["pools"][pname] = {
            "n": int(len(sub_all)),
            "per_day": round(len(sub_all) / max(1, sub_all["trade_date"].nunique()), 2),
            "ret_pct": round(float(sub_all["ret"].mean()) * 100, 3),
            "win_pct": round(float((sub_all["ret"] > 0).mean()) * 100, 1),
            "gate_n": int(len(sub_g)),
            "gate_ret_pct": round(float(sub_g["ret"].mean()) * 100, 3),
        }
        report(sub_all, f"{pname} · 全样本", out, pname)
        report(sub_g, f"{pname} · 闸门开", out, pname + "_gate")
        print(f"\n[{pname}] 笔数 {len(sub_all):,} / {sub_all['trade_date'].nunique()} 日 "
              f"= {out['pools'][pname]['per_day']} 笔/日", flush=True)

    # ---- 市场状态拆分: 同一因子在趋势向上/向下时方向是否一致(状态条件化)
    trend = (reg["mkt_idx"] >= reg["mkt_ma"])
    tmap = panel["trade_date"].map(trend).fillna(True).astype(bool).to_numpy()
    rank = sorted([c for c, _, _ in FACTORS if not out["factors"]["wide"][c].get("skip")],
                  key=lambda c: -abs(out["factors"]["wide"][c]["mono"]))[:6]
    print("\n=== 市场状态 × 因子 (宽池, 不叠闸门; 只列 |单调性| 前 6) ===")
    print("%-14s | %-28s | %-28s" % ("因子", "趋势向上 Q1→Q5 / 极差 / 单调", "趋势向下 Q1→Q5 / 极差 / 单调"))
    for col in rank:
        line = []
        for lab, mm in (("up", wide & tmap), ("dn", wide & ~tmap)):
            sub = panel.loc[mm]
            t = bucket_table(sub, col)
            out.setdefault("state", {}).setdefault(col, {})[lab] = t
            if t.get("skip"):
                line.append("样本不足")
                continue
            line.append("%+.3f→%+.3f / %.2f / %+.2f  (n=%d)"
                        % (t["buckets"][0]["ret"], t["buckets"][-1]["ret"],
                           t["spread"], t["mono"], t["n"]))
        print("%-14s | %-40s | %-40s" % (col, line[0], line[1]))
    out["state_meta"] = {"trend_up_days": int(trend.sum()), "all_days": int(len(trend))}

    # ---- 显式组合网格: 有经济学含义的三对, 各自在严池与宽池上跑
    PAIRS = [("vwap_dev", "vol_ratio"), ("vwap_dev", "close_pos"), ("vwap_dev", "gap_t")]
    for pname, m in (("wide", wide), ("strict", strict)):
        sub = panel.loc[m & gate]
        if len(sub) < 500:
            print(f"\n[{pname}] 闸门开后样本 {len(sub)} 不足, 跳过网格")
            continue
        for a, b in PAIRS:
            g = grid_table(sub, a, b)
            print(f"\n=== [{pname}·闸门开] {a} × {b} (行=低→高) ===")
            for i in range(3):
                print("   " + " ".join(("%8.3f" % g["lit"][i][j]) if g["lit"][i][j] is not None
                                       else "     n/a" for j in range(3))
                      + "   n=" + "/".join(str(x) for x in g["n"][i]))
            out["grid"][f"{pname}:{a}x{b}"] = {"a": a, "b": b, **g}

    # ---- 层数前沿: 从宽池逐层 AND 收紧, 看单笔收益能爬到哪, 以及是不是噪声
    print("\n=== 层数前沿: 宽池逐层收紧(每层取五分位里最好的一档) ===", flush=True)
    cols = [c for c, _, _ in FACTORS if out["factors"]["wide"][c].get("skip") is not True]
    sub = panel.loc[wide & gate]          # 用闸门开的日子(可交易)
    ret = sub["ret"].to_numpy()
    n_all = len(sub)
    days = sub["trade_date"].nunique()
    mask = np.ones(n_all, bool)
    remaining = list(cols)
    frontier = []
    print("%3s %-16s %-8s %9s %8s %8s %7s %s"
          % ("层", "因子", "取哪档", "n", "笔/日", "单笔%", "t值", "同n随机对照 5/50/95分位"))
    rng = np.random.default_rng(42)
    for step in range(6):
        base_mu = ret[mask].mean() * 100
        best = None
        for c in remaining:
            v = sub[c].to_numpy()
            vv = v[mask]
            if np.isnan(vv).all():
                continue
            q = pd.qcut(pd.Series(vv), 5, labels=False, duplicates="drop")
            qa = q.to_numpy()
            idx = np.where(mask)[0]
            for k in np.unique(qa):
                if np.isnan(k):
                    continue
                sel = np.zeros(n_all, bool)
                sel[idx[qa == k]] = True
                if sel.sum() < 100:
                    continue
                mu = ret[sel].mean()
                if best is None or mu > best[0]:
                    best = (mu, c, int(k) + 1, int(q.max()) + 1, sel)
        if best is None:
            break
        mu, c, kk, kmax, sel = best
        if step > 0 and mu <= base_mu:
            # 仍记录这一层(证明"叠不动"), 但不替换 mask —— 后续步骤在全池层面已无增量
            pass
        # 强制前沿: 无条件在当前子集上再砍一层, 让"叠层是否还提升"由数字自己说
        improved = (mu > base_mu) or step == 0
        mask = sel
        n = int(mask.sum())
        m_ = ret[mask]
        tval = float(m_.mean() / (m_.std() / np.sqrt(n))) if m_.std() else float("nan")
        # 随机对照: 从同池随机抽 n 个
        sims = np.array([ret[rng.choice(n_all, n, replace=False)].mean() * 100
                         for _ in range(200)])
        pct = float((sims < m_.mean() * 100).mean() * 100)
        print("%3d %-16s Q%-7d %9d %8.2f %8.3f %7.2f  %.3f/%.3f/%.3f (P=%2.0f%%)"
              % (step + 1, c, kk, n, n / max(1, days), m_.mean() * 100, tval,
                 np.percentile(sims, 5), np.percentile(sims, 50), np.percentile(sims, 95), pct))
        frontier.append({"step": step + 1, "factor": c, "bucket": f"Q{kk}/{kmax}",
                         "n": n, "per_day": round(n / max(1, days), 2),
                         "ret_pct": round(float(m_.mean()) * 100, 3), "t": round(tval, 2),
                         "rand_p5": round(float(np.percentile(sims, 5)), 3),
                         "rand_p50": round(float(np.percentile(sims, 50)), 3),
                         "rand_p95": round(float(np.percentile(sims, 95)), 3),
                         "pctile": round(pct, 1)})
        remaining = [x for x in remaining if x != c]
    out["frontier"] = frontier

    # ---- 新因子能否改进主策略: 宽池 + vwap_z Q1 与"严池形态""低开入场"逐层取交集
    print("\n=== vwap_z 与主策略条件的交集(宽池·闸门开) ===", flush=True)
    sub = panel.loc[wide & gate].copy()
    q1 = sub["vwap_z"].quantile(0.2)
    strict_m = (sub["pct"].between(-6.0, -3.0).to_numpy()
                & (sub["vol_ratio"] <= 0.8).to_numpy()
                & (sub["close_pos"] <= 0.4).to_numpy())
    # 入场缺口: T+1 开盘相对 T 收盘的跳空(实盘 09:25 可知)
    gap_in = (sub["buy_open_adj"] / sub["close_adj"] - 1).to_numpy() * 100
    layers = [
        ("宽池", np.ones(len(sub), bool)),
        ("+ vwap_z Q1", (sub["vwap_z"] <= q1).to_numpy()),
        ("+ 严池形态", strict_m),
        ("+ vwap_z & 严池", strict_m & (sub["vwap_z"] <= q1).to_numpy()),
        ("+ vwap_z & 缺口<=0", (sub["vwap_z"] <= q1).to_numpy() & (gap_in <= 0)),
        ("+ 三者交集", strict_m & (sub["vwap_z"] <= q1).to_numpy() & (gap_in <= 0)),
        ("+ 三者 & 缺口<=-1", strict_m & (sub["vwap_z"] <= q1).to_numpy() & (gap_in <= -1)),
    ]
    r_ = sub["ret"].to_numpy()
    inter = []
    for name, m in layers:
        if m.sum() < 30:
            print("%-22s n<30, 跳过" % name)
            continue
        x = r_[m]
        t_ = float(x.mean() / (x.std() / np.sqrt(len(x)))) if x.std() else float("nan")
        print("%-22s n=%8d 笔/日=%7.2f 单笔=%+.3f%% 胜率=%5.1f%% t=%5.2f"
              % (name, len(x), len(x) / max(1, sub["trade_date"].nunique()),
                 x.mean() * 100, (x > 0).mean() * 100, t_), flush=True)
        inter.append({"layer": name, "n": int(len(x)),
                      "per_day": round(len(x) / max(1, sub["trade_date"].nunique()), 2),
                      "ret_pct": round(float(x.mean()) * 100, 3),
                      "win_pct": round(float((x > 0).mean()) * 100, 1), "t": round(t_, 2)})
    out["intersect"] = inter
    # 主策略锚点(第 170 轮 1500 日窗口口径), 便于横向比较
    out["anchor"] = {"n": 353, "per_day": 0.25, "ret_pct": 1.435,
                     "note": "官方主策略(评分排序+缺口过滤+池深降档), 1.25x 前口径"}

    # ---- 样本外: 前 60% 交易日训练贪心规则(阈值取自训练段分位), 后 40% 验证
    print("\n=== 样本外验证: 前 60% 训练 → 后 40% 验证 ===", flush=True)
    ds = np.array(sorted(sub["trade_date"].unique()))
    cut = ds[int(len(ds) * 0.6)]
    dts = sub["trade_date"].to_numpy()
    tr = dts <= cut
    te = ~tr
    rarr = sub["ret"].to_numpy()
    print(f"切分点 {cut}: 训练 {tr.sum():,} 笔 / 验证 {te.sum():,} 笔", flush=True)

    def fit_greedy(mask, steps=4):
        """在 mask 内贪心选因子+档位, 返回规则列表(含训练段分位阈值)。"""
        rules = []
        rem = list(cols)
        cur = mask.copy()
        for _ in range(steps):
            best = None
            for c in rem:
                v = sub[c].to_numpy()
                vv = v[cur]
                if np.isnan(vv).all() or len(vv) < 200:
                    continue
                qs = np.nanquantile(vv, [0.2, 0.4, 0.6, 0.8])
                edges = [-np.inf, qs[0], qs[1], qs[2], qs[3], np.inf]
                for k in range(5):
                    sel = cur & (v >= edges[k]) & (v <= edges[k + 1])
                    if sel.sum() < 60:
                        continue
                    mu = rarr[sel].mean()
                    if best is None or mu > best[0]:
                        best = (mu, c, edges[k], edges[k + 1])
            if best is None:
                break
            mu, c, lo, hi = best
            cur = cur & (sub[c].to_numpy() >= lo) & (sub[c].to_numpy() <= hi)
            rules.append({"factor": c, "lo": float(lo), "hi": float(hi),
                          "train_n": int(cur.sum()), "train_ret": round(float(mu) * 100, 3)})
            rem = [x for x in rem if x != c]
        return rules, cur

    rules, cur_tr = fit_greedy(tr)
    m_te = te.copy()
    step_rows = []
    print("%3s %-14s %-22s | %8s %8s | %8s %8s %7s"
          % ("层", "因子", "训练段区间", "训练n", "训练%", "验证n", "验证%", "验证t"))
    for i, r in enumerate(rules, 1):
        v = sub[r["factor"]].to_numpy()
        m_te &= (v >= r["lo"]) & (v <= r["hi"])
        # 训练段该层累计子集
        mtr = tr.copy()
        for rr in rules[:i]:
            vv = sub[rr["factor"]].to_numpy()
            mtr &= (vv >= rr["lo"]) & (vv <= rr["hi"])
        a, b = rarr[mtr], rarr[m_te]
        t_te = float(b.mean() / (b.std() / np.sqrt(len(b)))) if len(b) > 5 and b.std() else float("nan")
        span = ("%.4g~%.4g" % (r["lo"], r["hi"])) if abs(r["lo"]) < 1e9 and abs(r["hi"]) < 1e9 \
            else ("<=%.4g" % r["hi"] if abs(r["hi"]) < 1e9 else ">=%.4g" % r["lo"])
        print("%3d %-14s %-22s | %8d %8.3f | %8d %8.3f %7.2f"
              % (i, r["factor"], span, len(a), a.mean() * 100,
                 len(b), (b.mean() * 100 if len(b) else float("nan")), t_te), flush=True)
        step_rows.append({"step": i, "factor": r["factor"], "lo": r["lo"], "hi": r["hi"],
                          "train_n": int(len(a)), "train_ret": round(float(a.mean()) * 100, 3),
                          "test_n": int(len(b)),
                          "test_ret": round(float(b.mean()) * 100, 3) if len(b) else None,
                          "test_t": round(t_te, 2) if len(b) else None})
    out["oos"] = {"cut": str(cut), "rules": rules, "steps": step_rows}

    # 简单组合的样本外: vwap_z Q1 + 缺口<=0
    print("\n--- 简单组合的样本外 ---", flush=True)
    q1_tr = np.nanquantile(sub["vwap_z"].to_numpy()[tr], 0.2)
    simple = [("vwap_z<=Q1(训练段)", (sub["vwap_z"] <= q1_tr).to_numpy()),
              ("+ 缺口<=0", (sub["vwap_z"] <= q1_tr).to_numpy() & (gap_in <= 0)),
              ("+ 缺口<=-1", (sub["vwap_z"] <= q1_tr).to_numpy() & (gap_in <= -1))]
    srows = []
    for nm, m in simple:
        a, b = rarr[m & tr], rarr[m & te]
        t_te = float(b.mean() / (b.std() / np.sqrt(len(b)))) if len(b) > 5 and b.std() else float("nan")
        print("%-22s 训练 n=%6d %+.3f%% | 验证 n=%6d %+.3f%% 胜率%5.1f%% t=%5.2f"
              % (nm, len(a), a.mean() * 100 if len(a) else float("nan"),
                 len(b), b.mean() * 100 if len(b) else float("nan"),
                 (b > 0).mean() * 100 if len(b) else float("nan"), t_te), flush=True)
        srows.append({"name": nm, "train_n": int(len(a)), "train_ret": round(float(a.mean()) * 100, 3),
                      "test_n": int(len(b)), "test_ret": round(float(b.mean()) * 100, 3),
                      "test_win": round(float((b > 0).mean()) * 100, 1), "test_t": round(t_te, 2)})
    out["oos_simple"] = srows

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "exp_fmix.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\nsaved output/exp_fmix.json")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1500)
