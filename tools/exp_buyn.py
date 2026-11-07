"""第166轮: 主策略"一次买 N 只"网格 —— 与第163轮副策略 buys_per_day 对称的检验。

现状: max_buy_per_day=1 + alloc_frac_pct=100, 每天 100% 押一只, 持有 1 个交易日。
假设: 同一天第 2/3 名候选与第 1 名期望相近(第165轮 rank 诊断), 分散到 N 只可降低方差拖累,
      同时把成交笔数放大, 把"每天只买一只"丢掉的候选捡回来。
做法: 只改三个数(exec_top_n / max_buy_per_day / alloc_frac_pct=100/N), 官方引擎逐日重跑。
"""
import copy
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.pop("BT_TOPK", None)
os.environ["BT_FULL_TRADES"] = "1"

from src.backtest import prepare_context, run_backtest  # noqa: E402
from src.config import load_config  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")


def summarize(r, n):
    m = r["metrics"]
    tr = [t for t in r["trades_all"] if t.get("ret_pct") is not None]
    yrs = {}
    for t in tr:
        y = str(t["signal_date"])[:4]
        yrs.setdefault(y, []).append(t["ret_pct"])
    # 每日持仓只数(按买入日聚合)
    per_day = {}
    for t in tr:
        per_day[t["buy_date"]] = per_day.get(t["buy_date"], 0) + 1
    dist = {}
    for v in per_day.values():
        dist[v] = dist.get(v, 0) + 1
    return {
        "n_buy": n,
        "total_ret_pct": m["总收益率%"],
        "annual_pct": m["年化收益率%"],
        "max_dd_pct": m["最大回撤%"],
        "calmar": m["卡玛比率"],
        "sharpe": m["夏普比率"],
        "trades": m["交易次数"],
        "avg_trade": m["平均单笔收益%"],
        "win_rate": m["胜率%"],
        "flat_days": m["空仓天数"],
        "buy_days": len(per_day),
        "pos_per_buydate": dict(sorted(dist.items())),
        "years": {k: [len(v), round(sum(v) / len(v), 3)] for k, v in sorted(yrs.items())},
    }


def main():
    base = load_config()
    print("准备上下文...", flush=True)
    t0 = time.time()
    ctx = prepare_context(base, base["backtest"]["trading_days"])
    print(f"  耗时 {time.time()-t0:.0f}s", flush=True)
    rows = []
    for n in (1, 2, 3, 4):
        cfg = copy.deepcopy(base)
        cfg["selection"]["exec_top_n"] = n
        cfg["execution"]["max_buy_per_day"] = n
        cfg["execution"]["alloc_frac_pct"] = round(100.0 / n, 6)
        t = time.time()
        r = run_backtest(cfg, None, ctx)
        row = summarize(r, n)
        rows.append(row)
        print(f"N={n} 年化{row['annual_pct']}% 回撤{row['max_dd_pct']}% 卡玛{row['calmar']} "
              f"笔数{row['trades']} 单笔{row['avg_trade']}% 胜率{row['win_rate']}% "
              f"买入日{row['buy_days']} 分布{row['pos_per_buydate']} ({time.time()-t:.0f}s)",
              flush=True)
        json.dump(rows, open(os.path.join(OUT, "exp_buyn.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps(rows, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
