"""第166轮: 量化"现金锁"成本 —— 卖出在 T+2 收盘、回款次日开盘才可用,
于是隔日连发的信号会被静默吞掉。想知道吞掉的是不是好票, 先把它记录出来再算假设收益。

口径与引擎一致: 假设买入日 d 开盘买、次一交易日收盘卖(即引擎的 T+2 收盘卖), 毛收益。
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ["BT_DEBUG_CASH"] = "1"
os.environ["BT_FULL_TRADES"] = "1"

from src.backtest import prepare_context, run_backtest  # noqa: E402
from src.config import load_config  # noqa: E402


def main():
    cfg = load_config()
    ctx = prepare_context(cfg, cfg["backtest"]["trading_days"])
    r = run_backtest(cfg, None, ctx)
    tr = pd.DataFrame([t for t in r["trades_all"] if t.get("ret_pct") is not None])
    cb = pd.DataFrame(r["cash_blocked"])
    print(f"成交 {len(tr)} 笔, 现金锁吞掉 {len(cb)} 条信号 / {cb['d'].nunique()} 天")
    if cb.empty:
        return

    days = sorted(ctx["by_date"].keys())
    idx = {d: i for i, d in enumerate(days)}
    per = ctx["per_code"]

    def hyp(code, d):
        """d 开盘买, 下一交易日收盘卖(引擎口径)。"""
        i = idx.get(d)
        if i is None or i + 1 >= len(days):
            return None
        sub = per.get(code)
        if sub is None:
            return None
        d1, d2 = d, days[i + 1]
        if d1 not in sub.index or d2 not in sub.index:
            return None
        a, b = sub.loc[d1], sub.loc[d2]
        if a["is_paused"] == 1 or b["is_paused"] == 1 or a["open_adj"] <= 0:
            return None
        if float(a["open"]) >= float(a["high_limit"]) - 1e-9:
            return None      # 一字涨停买不到
        return (float(b["close_adj"]) / float(a["open_adj"]) - 1) * 100

    cb["hyp_ret"] = [hyp(c, d) for c, d in zip(cb["code"], cb["d"])]
    ok = cb.dropna(subset=["hyp_ret"])
    print(f"可回测 {len(ok)} 条")
    print("\n=== 被吞信号(假设收益) ===")
    print(ok["hyp_ret"].describe().round(3).to_string())
    print(f"均值 {ok['hyp_ret'].mean():.3f}%  中位 {ok['hyp_ret'].median():.3f}%  "
          f"胜率 {(ok['hyp_ret'] > 0).mean()*100:.1f}%")
    print(f"对照: 实际成交 均值 {tr['ret_pct'].mean():.3f}% 中位 {tr['ret_pct'].median():.3f}% "
          f"胜率 {(tr['ret_pct'] > 0).mean()*100:.1f}%")
    print("\n=== 按池 / 名次 ===")
    print(ok.groupby(["pool", "rank"])["hyp_ret"].agg(n="count", mean="mean",
                                                      win=lambda x: (x > 0).mean() * 100).round(2).to_string())
    print("\n=== 按信号日到买入日的间隔(连发程度) ===")
    ok = ok.copy()
    ok["gap_b"] = pd.cut(ok["gap"], [-99, -2, -1, 0, 0.5, 2, 99])
    print(ok.groupby("gap_b", observed=True)["hyp_ret"].agg(n="count", mean="mean",
                                                            win=lambda x: (x > 0).mean() * 100).round(2).to_string())
    print("\n=== 按评分 ===")
    ok["sb"] = pd.cut(ok["score"].astype(float), [0, 1.2, 1.6, 2.0, 99])
    print(ok.groupby("sb", observed=True)["hyp_ret"].agg(n="count", mean="mean",
                                                         win=lambda x: (x > 0).mean() * 100).round(2).to_string())
    print("\n=== 按 hit(能买到目标仓位的比例) ===")
    ok["hb"] = pd.cut(ok["hit"], [-0.01, 0.05, 0.15, 0.3, 0.6])
    print(ok.groupby("hb", observed=True)["hyp_ret"].agg(n="count", mean="mean").round(2).to_string())

    cb.to_pickle(os.path.join(ROOT, "data", "cache", "cash_blocked.pkl"))
    json.dump({"n_blocked": int(len(cb)), "n_testable": int(len(ok)),
               "blocked_mean": round(float(ok["hyp_ret"].mean()), 3),
               "blocked_win": round(float((ok["hyp_ret"] > 0).mean() * 100), 1),
               "actual_mean": round(float(tr["ret_pct"].mean()), 3),
               "actual_win": round(float((tr["ret_pct"] > 0).mean() * 100), 1)},
              open(os.path.join(ROOT, "output", "exp_cashlock.json"), "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
