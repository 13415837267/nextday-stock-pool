"""第166轮: 股票池扩板 —— 现行只做主板(10% 涨跌停), 创业板(20%)从未进入选股。

动机: 主策略的瓶颈是"信号日稀缺"(1500 日只有 353 笔), 而不是资金。
扩板不放松任何选股条件, 只是把同一套 dip 形态应用到更多票上 —— 是少数
"增加次数但不稀释质量"的路径(第163轮副策略已经证明: 增加次数本身不等于增值)。
风险: 20% 涨跌停让同形态的缺口/波动尺度完全不同, 可能整族失效。用数据说话。
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


def summarize(r, tag):
    m = r["metrics"]
    tr = [t for t in r["trades_all"] if t.get("ret_pct") is not None]
    yrs = {}
    for t in tr:
        yrs.setdefault(str(t["signal_date"])[:4], []).append(t["ret_pct"])
    return {"tag": tag, "total": m["总收益率%"], "annual": m["年化收益率%"],
            "dd": m["最大回撤%"], "calmar": m["卡玛比率"], "sharpe": m["夏普比率"],
            "trades": m["交易次数"], "avg": m["平均单笔收益%"], "win": m["胜率%"],
            "years": {k: [len(v), round(sum(v) / len(v), 3)] for k, v in sorted(yrs.items())}}


def main():
    base = load_config()
    ctx_all = prepare_context(base, base["backtest"]["trading_days"])
    variants = [(["主板"], "主板(基线)"),
                (["主板", "创业板"], "主板+创业板同池"),
                (["创业板"], "仅创业板")]
    rows = []
    for boards, tag in variants:
        cfg = copy.deepcopy(base)
        cfg["universe"]["boards"] = boards
        t = time.time()
        try:
            ctx = ctx_all if boards == ["主板"] else prepare_context(cfg, cfg["backtest"]["trading_days"])
            r = run_backtest(cfg, None, ctx)
        except Exception as e:  # noqa: BLE001
            print(f"{tag} 失败: {e}")
            continue
        row = summarize(r, tag)
        rows.append(row)
        print(f"{tag}: 年化{row['annual']}% 回撤{row['dd']}% 卡玛{row['calmar']} "
              f"笔数{row['trades']} 单笔{row['avg']}% 胜率{row['win']}% ({time.time()-t:.0f}s)", flush=True)
        json.dump(rows, open(os.path.join(OUT, "exp_boards.json"), "w"), ensure_ascii=False, indent=1)
        del r, ctx          # 每个变体一份 8M 行上下文, 不释放会 2~3 份共存被 OOM 杀掉(第166轮踩过)
        ctx = None
        ctx_all = None
    for r in rows:
        print(r["tag"], r["years"])


if __name__ == "__main__":
    main()
