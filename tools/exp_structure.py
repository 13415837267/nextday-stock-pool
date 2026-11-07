"""第166轮汇总: ①N=2 时 rank1 / rank2 各自的单笔收益(拆开看稀释来源)
②买入日间隔分布(解释"现金锁"为何只触发 1 次) ③资本的时间价值(解释所有复用为何必败)
"""
import copy
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


def main():
    base = load_config()
    ctx = prepare_context(base, base["backtest"]["trading_days"])

    r1 = run_backtest(base, None, ctx)
    t1 = pd.DataFrame([t for t in r1["trades_all"] if t.get("ret_pct") is not None])

    cfg = copy.deepcopy(base)
    cfg["selection"]["exec_top_n"] = 2
    cfg["execution"]["max_buy_per_day"] = 2
    cfg["execution"]["alloc_frac_pct"] = 50.0
    r2 = run_backtest(cfg, None, ctx)
    t2 = pd.DataFrame([t for t in r2["trades_all"] if t.get("ret_pct") is not None])

    print("=== N=1 vs N=2 按名次拆单笔 ===")
    for nm, t in (("N=1", t1), ("N=2", t2)):
        g = t.groupby("rank")["ret_pct"].agg(n="count", mean="mean", win=lambda x: (x > 0).mean() * 100)
        print(nm, g.round(3).to_dict("index"))

    print("\n=== 买入日间隔(交易日) ===")
    days = sorted(ctx["by_date"].keys())
    idx = {d: i for i, d in enumerate(days)}
    bd = sorted(t1["buy_date"].unique(), key=lambda d: idx[d])
    dif = [idx[bd[i]] - idx[bd[i - 1]] for i in range(1, len(bd))]
    s = pd.Series(dif).value_counts().sort_index()
    print(s.head(8).to_string())
    print(f"间隔=1 的占比 {(np.mean(np.array(dif) == 1) * 100):.1f}%  中位 {np.median(dif):.0f} 日")

    print("\n=== 资本的时间价值 ===")
    nav = [e["nav"] for e in r1["equity"]]
    n = len(nav)
    logday = np.log(nav[-1] / nav[0]) / n
    print(f"1500 日对数年化 {logday*252*100:.1f}%  每日 {logday*100:.3f}%")
    for d in (1, 2, 5, 10):
        print(f"  占用 {d:>2} 日的机会成本 = {(np.exp(logday*d)-1)*100:.2f}%")
    print(f"  副策略单笔 1.366% / 持有 10 日 = 每日 {1.366/10:.3f}% —— 差 {logday*100 - 1.366/10*1:.3f}pp/日")

    json.dump({"rank_stats": {nm: t.groupby("rank")["ret_pct"].agg(["count", "mean"]).to_dict("index")
                              for nm, t in (("n1", t1), ("n2", t2))},
               "gap1_share": float(np.mean(np.array(dif) == 1)),
               "log_day_pct": float(logday * 100)},
              open(os.path.join(ROOT, "output", "exp_structure.json"), "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
