"""第166轮: 把"创业板 dip"当备用池 —— 不放松任何条件, 只在主板没有可买票时补位。

已证: 混合板块池(主板+创业板同池评分)会把年化从 104% 打到 76.8%, 因为创业板票进入
同一个 z 分标准化池, 改变了主板的排序 —— 是评分口径的副作用, 不是创业板本身无效
(创业板单独跑: 285 笔 / 单笔 +1.05% / 年化 33.1%)。
所以正确的接法是独立评分、主板优先、仅在主力不合格时补位(fallback)。

变体:
  V0 基线           fallback_slot=true, 三族(只补"主板无票"的 65 天)
  V1 创业板补位     fallback_slot=false, 只挂"dip@创业板"(主板出票但被评分门拦下时补)
  V2 三族+创业板     fallback_slot=false, 四族
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
GEM = {"mode": "dip", "top_n": 1, "gap_below": 0, "segment": "all", "boards": ["创业板"]}


def summarize(r, tag):
    m = r["metrics"]
    tr = [t for t in r["trades_all"] if t.get("ret_pct") is not None]
    yrs, pools = {}, {}
    for t in tr:
        yrs.setdefault(str(t["signal_date"])[:4], []).append(t["ret_pct"])
        pools.setdefault(t.get("pool", "main"), []).append(t["ret_pct"])
    return {"tag": tag, "total": m["总收益率%"], "annual": m["年化收益率%"],
            "dd": m["最大回撤%"], "calmar": m["卡玛比率"], "trades": m["交易次数"],
            "avg": m["平均单笔收益%"], "win": m["胜率%"],
            "pools": {k: [len(v), round(sum(v) / len(v), 3)] for k, v in pools.items()},
            "years": {k: [len(v), round(sum(v) / len(v), 3)] for k, v in sorted(yrs.items())}}


def main():
    base = load_config()
    variants = []
    variants.append((base, "V0 基线"))
    c1 = copy.deepcopy(base)
    c1["selection"]["fallback_slot"] = False
    c1["selection"]["fallbacks"] = [GEM]
    variants.append((c1, "V1 创业板补位"))
    c2 = copy.deepcopy(base)
    c2["selection"]["fallback_slot"] = False
    c2["selection"]["fallbacks"] = list(base["selection"]["fallbacks"]) + [GEM]
    variants.append((c2, "V2 三族+创业板"))
    rows = []
    for i, (cfg, tag) in enumerate(variants):
        t = time.time()
        ctx = prepare_context(cfg, cfg["backtest"]["trading_days"]) if i else prepare_context(base, 1500)
        r = run_backtest(cfg, None, ctx)
        row = summarize(r, tag)
        rows.append(row)
        print(f"{tag}: 年化{row['annual']}% 回撤{row['dd']}% 卡玛{row['calmar']} 笔数{row['trades']} "
              f"单笔{row['avg']}% 胜率{row['win']}% 池{row['pools']} ({time.time()-t:.0f}s)", flush=True)
        json.dump(rows, open(os.path.join(OUT, "exp_gem_fb.json"), "w"), ensure_ascii=False, indent=1)
    for r in rows:
        print(r["tag"], "years", r["years"])


if __name__ == "__main__":
    main()
