# -*- coding: utf-8 -*-
"""第 170 轮: 复利的上限不在"选得更准", 在"下注多少" —— 融资杠杆与资金效率

背景(第 164 轮): 分仓/资金复用/卖出时点三条路全部低于纯主策略 104.04%,
结论是"复利到顶"。但那一轮只动了**资金怎么分**, 没动**总敞口多大** ——
1500 日里主策略单笔平均 +1.435%、盈亏比 2.32、日频波动约 2%, 按凯利公式
f* = μ/σ² 远大于 1, 意味着"满仓"其实低于数学最优下注额。

本轮把"下注额"这一维打开:
  A. 凯利比例: 逐笔 μ/σ², 给出理论最优杠杆与半凯利。
  B. 固定杠杆前沿 L = 1.0~2.0: 融资利息 5.5%/年按持仓日计, 现金空仓日按
     逆回购 1.8%/年计; 输出年化/回撤/卡玛/单笔最大亏损。
  C. 条件杠杆: 只在 score>=阈值 或 pool_n>=阈值 的笔上加杠杆 —— 第 169 轮
     已验证"1.0x 档 +1.849% vs 0.5x 档 +0.351%"单调, 杠杆该压在高置信笔上。
  D. 实盘约束: 单票集中度、维持担保比例 130%、利息, 给出可落地档位。

口径与主策略逐位一致(1500 日, T+1 开盘买 → T+2 收盘卖), 收益为引擎净收益。
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ["BT_FULL_TRADES"] = "1"

from src.backtest import prepare_context, run_backtest  # noqa: E402
from src.config import load_config  # noqa: E402

OUT = os.path.join(ROOT, "output")
RATE = 0.055      # 融资年利率
CASH = 0.018      # 空仓现金(国债逆回购/货基)年化
TRADING_DAYS = 252


def norm(d):
    return str(d).replace("-", "")[:8]


def daily_curve(equity, hold_days_idx, L, rate=RATE, cash=CASH):
    """按日收益放大 L 倍; 持仓日扣融资利息, 空仓日计现金收益。返回净值序列。"""
    nav = 1.0
    navs = [1.0]
    prev = equity[0]["nav"]
    for i in range(1, len(equity)):
        cur = equity[i]["nav"]
        r = cur / prev - 1.0
        prev = cur
        if hold_days_idx[i]:
            nav *= 1.0 + L * r - (L - 1.0) * rate / TRADING_DAYS
        else:
            nav *= 1.0 + cash / TRADING_DAYS
        navs.append(nav)
    return np.array(navs)


def metrics(navs):
    n = len(navs)
    total = navs[-1] / navs[0] - 1
    years = n / TRADING_DAYS
    ann = (navs[-1] / navs[0]) ** (1 / years) - 1 if years > 0 else 0
    peak = np.maximum.accumulate(navs)
    dd = (navs / peak - 1).min()
    rets = navs[1:] / navs[:-1] - 1
    sharpe = rets.mean() / (rets.std() + 1e-12) * np.sqrt(TRADING_DAYS)
    return dict(total=total * 100, ann=ann * 100, mdd=dd * 100,
                calmar=(ann / abs(dd)) if dd < 0 else np.nan, sharpe=sharpe)


def trade_leverage(trades, L_default, rate=RATE, score_th=None, pool_th=None,
                   L_hi=2.0):
    """按笔复利: 每笔投入 L 倍权益; 返回净值、逐笔杠杆、利息合计。"""
    nav = 1.0
    navs = []
    levs = []
    interest = 0.0
    for t in trades:
        ret = t["ret_pct"] / 100.0
        L = L_default
        if score_th is not None and (t.get("score") or 0) >= score_th:
            L = L_hi
        if pool_th is not None and (t.get("pool_n") or 0) >= pool_th:
            L = L_hi
        levs.append(L)
        d1, d2 = norm(t["buy_date"]), norm(t["sell_date"])
        hold = max(1, (pd.to_datetime(d2) - pd.to_datetime(d1)).days)
        cost = (L - 1.0) * rate * hold / 365.0
        interest += (L - 1.0) * rate * hold / 365.0
        nav *= 1.0 + L * ret - cost
        navs.append(nav)
    return np.array([1.0] + navs), np.array(levs), interest


def main():
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    cfg = load_config()
    ctx = prepare_context(cfg, days)
    r = run_backtest(cfg, days, ctx)
    m = r["metrics"]
    print(f"\n基线: {m['总收益率%']}% / 年化 {m['年化收益率%']}% / 回撤 {m['最大回撤%']}% "
          f"/ 卡玛 {m['卡玛比率']} / {m['交易次数']} 笔")

    trades = [t for t in r["trades_all"] if t.get("ret_pct") is not None]
    rets = np.array([t["ret_pct"] / 100.0 for t in trades])
    mu, sd = rets.mean(), rets.std(ddof=1)
    kelly = mu / (sd ** 2)
    print(f"\n=== A. 凯利: 单笔 μ={mu*100:.3f}% σ={sd*100:.3f}% "
          f"f*={kelly:.2f}x  半凯利={kelly/2:.2f}x  满仓=1.0x")

    # 持仓日索引(逐日)
    dates = [norm(e["d"]) for e in r["equity"]]
    idx = {d: i for i, d in enumerate(dates)}
    hold = np.zeros(len(dates), bool)
    for t in trades:
        i, j = idx.get(norm(t["buy_date"])), idx.get(norm(t["sell_date"]))
        if i is not None and j is not None:
            hold[i:j + 1] = True
    print(f"持仓日 {int(hold.sum())}/{len(dates)} = {hold.mean()*100:.1f}%")

    out = {"baseline": m, "kelly": {"mu": mu, "sd": sd, "f": kelly},
           "hold_frac": float(hold.mean()),
           "frontier": [], "cond": [], "trade_frontier": []}

    print("\n=== B. 固定杠杆前沿(日频再平衡, 含 5.5% 融资利息 / 空仓 1.8% 现金) ===")
    print("L      总收益%      年化%    回撤%    卡玛    夏普")
    for L in (1.0, 1.25, 1.5, 1.75, 2.0):
        navs = daily_curve(r["equity"], hold, L)
        mm = metrics(navs)
        out["frontier"].append(dict(L=L, **mm))
        print(f"{L:<6.2f} {mm['total']:>10.1f} {mm['ann']:>10.2f} {mm['mdd']:>9.2f} "
              f"{mm['calmar']:>7.2f} {mm['sharpe']:>7.2f}")

    print("\n=== C. 按笔复利(逐笔全额, 更贴近实盘不调仓) ===")
    print("L      总收益%      年化%    回撤%    单笔最大亏损%")
    for L in (1.0, 1.25, 1.5, 1.75, 2.0):
        navs, levs, itr = trade_leverage(trades, L)
        years = len(dates) / TRADING_DAYS
        total = navs[-1] / navs[0] - 1
        ann = (navs[-1] / navs[0]) ** (1 / years) - 1
        peak = np.maximum.accumulate(navs)
        dd = (navs / peak - 1).min()
        out["trade_frontier"].append(dict(L=L, total=total * 100, ann=ann * 100,
                                          mdd=dd * 100,
                                          worst=float(rets.min() * L * 100)))
        print(f"{L:<6.2f} {total*100:>10.1f} {ann*100:>10.2f} {dd*100:>9.2f} "
              f"{rets.min()*L*100:>12.2f}")

    print("\n=== D. 条件杠杆: 只在 1.5x 基础(core)上加到 2.0x ===")
    print("规则                     笔数   年化%    回撤%    相对 1.5x")
    base_nav, _, _ = trade_leverage(trades, 1.5)
    base_ann = (base_nav[-1] / base_nav[0]) ** (TRADING_DAYS / len(dates)) - 1
    for name, kw in [
        ("score>=1.2", dict(score_th=1.2)),
        ("score>=1.3", dict(score_th=1.3)),
        ("pool>=21", dict(pool_th=21)),
        ("pool>=11", dict(pool_th=11)),
    ]:
        navs, levs, itr = trade_leverage(trades, 1.5, **kw)
        years = len(dates) / TRADING_DAYS
        ann = (navs[-1] / navs[0]) ** (1 / years) - 1
        peak = np.maximum.accumulate(navs)
        dd = (navs / peak - 1).min()
        n_hi = int((levs > 1.5).sum())
        out["cond"].append(dict(rule=name, n=n_hi, ann=ann * 100, mdd=dd * 100))
        print(f"{name:<22} {n_hi:>5d} {ann*100:>8.2f} {dd*100:>8.2f} "
              f"{(ann-base_ann)*100:>+9.2f}pp")

    print("\n=== E. 单笔最大亏损(杠杆下的穿仓距离) ===")
    for L in (1.0, 1.5, 2.0):
        print(f"L={L}: 最差单笔 {rets.min()*100*L:.2f}%  2.5%分位 {np.percentile(rets,2.5)*100*L:.2f}%")

    json.dump(out, open(os.path.join(OUT, "exp_leverage.json"), "w"),
              ensure_ascii=False, indent=1)
    print("\nsaved output/exp_leverage.json")


if __name__ == "__main__":
    main()
