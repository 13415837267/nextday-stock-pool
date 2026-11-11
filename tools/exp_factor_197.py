# -*- coding: utf-8 -*-
"""第197轮: 主策略逐笔收益的 alpha / beta 分解。

动机
----
196 轮归因显示"主策略有持仓的日子市场等权 +0.161%、空仓日 −0.068%" ——
策略挑日子在场, 这到底是**选股 alpha** 还是**踩对了 beta 时钟**?
不拆开, 就无法回答"110% 年化里有多少是市场给的、能不能扛住 beta 回撤"。

口径
----
· 逐笔区间 = T+1 开盘(买入) → T+2 收盘(现行 sell_open_frac=0 是全收盘卖)。
· 市场组合 = 当日全部非停牌、非 ST、成交额>0 个股的**等权**收益, 用复权价
  (open*factor / close*factor) 拼成"市场净值", 再取同区间收益。等权而非市值加权,
  因为本策略买的是中小市值, 等权才是它真正的机会集。
· 小市值代理: 每日按成交额(amount)分位取最低 30% 的等权收益 —— 无流通股本字段,
  成交额是可得的最好流动性/规模代理; 这一列只作**方向性参照**, 不作为结论。
· 回归 ret_i = a + b*mkt_i + e_i, Newey-West lag=2(相邻笔共享 1 个交易日, 必然自相关)。
· alpha 年化: 365 笔/6 年 → a × 365 / 6, 复利口径另给 (1+a)^(365/6) − 1。

输出: output/exp_factor_197_{1500,3000}.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context, run_backtest  # noqa: E402
from src.config import load_config  # noqa: E402

OUT = ROOT / "output"


def _d8(s) -> str:
    return str(s).replace("-", "")[:8]


def _market_series(ctx: dict) -> pd.DataFrame:
    """逐日等权收益(收盘/前收)与日内收益(收盘/开盘), 含低成交额组。

    注意: 不能用"日内收益连乘"当日收益 —— 那会把隔夜丢掉。买入是 T+1 开盘,
    卖出是 T+2 收盘, 区间 = (T+1 开盘→收盘) × (T+2 收盘/前收)。
    """
    dates = ctx["main_dates"]
    rows = []
    for d in dates:
        day = ctx["by_date"].get(d)
        if day is None or not len(day):
            rows.append({"d": d})
            continue
        ok = (day["is_paused"].astype(int) == 0) & (day["vol"].astype(float) > 0)
        ok &= (day["amount"].astype(float) > 0) & (day["pre_close"].astype(float) > 0)
        if "is_st" in day.columns:
            ok &= (day["is_st"].astype(int) == 0)
        sub = day.loc[ok]
        o = sub["open"].astype(float)
        c = sub["close"].astype(float)
        pc = sub["pre_close"].astype(float)
        m = (o > 0) & (c > 0) & (pc > 0)
        if int(m.sum()) < 100:
            rows.append({"d": d})
            continue
        r = (c[m] / pc[m]).values          # 收盘/前收 = 当日总收益(含隔夜)
        oc = (c[m] / o[m]).values          # 日内收益
        amt = sub["amount"].astype(float)[m].values
        q = np.quantile(amt, 0.30)
        mask = amt <= q
        rows.append({"d": d, "n": int(m.sum()),
                     "eq_r": float(r.mean()) - 1.0, "eq_oc": float(oc.mean()) - 1.0,
                     "sm_r": float(r[mask].mean()) - 1.0 if int(mask.sum()) >= 30 else np.nan,
                     "sm_oc": float(oc[mask].mean()) - 1.0 if int(mask.sum()) >= 30 else np.nan})
    return pd.DataFrame(rows).set_index("d")


def _nw_tstat(y: np.ndarray, X: np.ndarray, lag: int = 2):
    """OLS + Newey-West 标准误。X 已含常数列。"""
    n, k = X.shape
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    XtX_inv = np.linalg.inv(X.T @ X)
    S = np.zeros((k, k))
    for l in range(lag + 1):
        w = 1.0 if l == 0 else 1.0 - l / (lag + 1.0)
        for t in range(l, n):
            u = np.outer(X[t] * e[t], X[t - l] * e[t - l])
            S += w * u
    V = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(V))
    return beta, se, beta / se


def main(days: int) -> None:
    os.environ["BT_FULL_TRADES"] = "1"
    cfg = load_config()
    print(f"[factor197] 跑主引擎 {days} 日 ...", flush=True)
    ctx = prepare_context(cfg, days)
    res = run_backtest(cfg, days=days, ctx=ctx)
    trades = [t for t in res.get("trades_all", []) if t.get("ret_pct") is not None
              and t.get("buy_date") and t.get("sell_date")]
    print("成交", len(trades), flush=True)

    mkt = _market_series(ctx)
    # 每笔区间收益 = Π_{t=b+1..s}(1+日收益) × (1+买入日日内收益) − 1
    # b = buy_date(开盘买), s = sell_date(收盘卖)
    idx = {d: i for i, d in enumerate(mkt.index)}

    def _seg(i0: int, i1: int, rcol: str, ocol: str) -> float:
        """i0=buy 位置, i1=sell 位置; 返回 (open_buy → close_sell) 的净值比。"""
        w = mkt.iloc[i0:i1 + 1]
        if w[rcol].isna().any() or w[ocol].isna().any() or len(w) < 2:
            return np.nan
        rr = float(np.prod(1.0 + w[rcol].values[1:]))      # 从 b+1 到 s 的日收益
        ro = float(1.0 + w[ocol].iloc[0])                   # 买入日 open→close
        return rr * ro

    recs = []
    for t in trades:
        b, s_ = _d8(t["buy_date"]), _d8(t["sell_date"])
        if b not in idx or s_ not in idx:
            continue
        i0, i1 = idx[b], idx[s_]
        if i1 <= i0:
            continue
        ratio = _seg(i0, i1, "eq_r", "eq_oc")
        if ratio != ratio:
            continue
        smr = _seg(i0, i1, "sm_r", "sm_oc")
        recs.append({"code": t.get("code"), "buy": b, "sell": s_,
                     "ret": float(t["ret_pct"]),
                     "mkt": (ratio - 1) * 100,
                     "sm": (smr - 1) * 100 if smr == smr else np.nan,
                     "hold": i1 - i0})
    df = pd.DataFrame(recs)
    print("对齐", len(df), flush=True)

    y = df["ret"].values / 100.0
    out = {"days": days, "n": int(len(df)), "mean_ret_pct": float(df["ret"].mean()),
           "median_ret_pct": float(df["ret"].median()),
           "mean_mkt_pct": float(df["mkt"].mean()),
           "mean_sm_pct": float(df["sm"].mean()),
           "win": float((df["ret"] > 0).mean())}

    X = np.column_stack([np.ones(len(df)), df["mkt"].values / 100.0])
    beta, se, t = _nw_tstat(y, X, lag=2)
    out["mkt"] = {"alpha_daily_pct": float(beta[0] * 100), "alpha_t": float(t[0]),
                  "beta": float(beta[1]), "beta_t": float(t[1]),
                  "alpha_ann_simple_pct": float(beta[0] * 365 / 6 * 100),
                  "alpha_ann_comp_pct": float(((1 + beta[0]) ** (365 / 6) - 1) * 100),
                  "r2": float(1 - np.var(y - X @ beta) / np.var(y))}

    sub = df.dropna(subset=["sm"])
    if len(sub) > 30:
        X2 = np.column_stack([np.ones(len(sub)), sub["sm"].values / 100.0])
        b2, se2, t2 = _nw_tstat(sub["ret"].values / 100.0, X2, lag=2)
        out["small"] = {"alpha_daily_pct": float(b2[0] * 100), "alpha_t": float(t2[0]),
                        "beta": float(b2[1]), "beta_t": float(t2[1]), "n": int(len(sub))}
        # 双因子
        X3 = np.column_stack([np.ones(len(sub)), sub["mkt"].values / 100.0,
                              (sub["sm"].values - sub["mkt"].values) / 100.0])
        b3, se3, t3 = _nw_tstat(sub["ret"].values / 100.0, X3, lag=2)
        out["two"] = {"alpha_daily_pct": float(b3[0] * 100), "alpha_t": float(t3[0]),
                      "beta_mkt": float(b3[1]), "beta_sm": float(b3[2]),
                      "t_mkt": float(t3[1]), "t_sm": float(t3[2])}

    # 极端行情分层: 市场最差 20% 的笔
    q20 = df["mkt"].quantile(0.2)
    out["bad_days"] = {"n": int((df["mkt"] <= q20).sum()),
                       "ret_pct": float(df.loc[df["mkt"] <= q20, "ret"].mean()),
                       "mkt_pct": float(df.loc[df["mkt"] <= q20, "mkt"].mean()),
                       "win": float((df.loc[df["mkt"] <= q20, "ret"] > 0).mean())}
    out["good_days"] = {"n": int((df["mkt"] >= df["mkt"].quantile(0.8)).sum()),
                        "ret_pct": float(df.loc[df["mkt"] >= df["mkt"].quantile(0.8), "ret"].mean()),
                        "mkt_pct": float(df.loc[df["mkt"] >= df["mkt"].quantile(0.8), "mkt"].mean())}
    df.to_csv(OUT / f"exp_factor_197_trades_{days}.csv", index=False)
    (OUT / f"exp_factor_197_{days}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1500)
