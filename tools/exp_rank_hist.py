# -*- coding: utf-8 -*-
"""同一个 alpha 的第 2 名：能不能当第二条腿（第187轮）。

背景
----
第 187 轮把"另一种形态"全部测死了：
    右侧追高 86 变体(含全板块) -> 最好 8.8 倍, 全史 -92.9%
    放量深跌反转 9 变体        -> 最好 209%, 单笔 0.45%(现役 1.44%)
    扩槽位 4 变体              -> 单笔腰斩
**在这套日线价量数据上，单笔 1%+ 的 edge 只有一种：缩量阴线 + 趋势内 + 次日低开承接。**

所以第二条腿只能从"同一个 alpha 的另一个切面"里找。候选里最干净的一种：
**同一天、同一个形态、评分第 2 名**。它与主策略（第 1 名）在物理上不重叠 ——
不同股票、不同持仓 —— 两条腿可以各自满仓复利，互不稀释。

风险是明确的：第 1 名之所以是第 1 名，可能因为评分真的有效（第 176 轮结论：
"排序键第 1 名本身就是 alpha"）。本轮直接量化第 2/3 名的衰减速度，
衰减慢（>1.0%/笔）就能独立成腿，衰减快（<0.5%）就否掉。

用法:
    python3 tools/exp_rank_hist.py --days 1500
"""
import argparse
import copy
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["BT_FULL_TRADES"] = "1"

from src.backtest import prepare_context, run_backtest  # noqa: E402
from src.config import load_config  # noqa: E402


def _row(res):
    m = res["metrics"]
    g = lambda k: float(str(m[k]).replace(",", ""))          # noqa: E731
    return {
        "总收益%": round(g("总收益率%"), 1),
        "年化%": round(g("年化收益率%"), 1),
        "回撤%": round(g("最大回撤%"), 1),
        "卡玛": round(g("卡玛比率"), 2),
        "笔数": int(g("交易次数")),
        "单笔%": round(g("平均单笔收益%"), 2),
        "胜率%": round(g("胜率%"), 1),
    }


def rk(c, off=0, lev=None, topk=None):
    c["selection"]["pick_offset"] = off
    if topk is not None:
        c["selection"]["exec_top_n"] = topk
    if lev is not None:
        c["execution"]["leverage"] = lev


PLAN = [
    ("r1", lambda c: rk(c, 0)),
    ("r2", lambda c: rk(c, 1)),
    ("r3", lambda c: rk(c, 2)),
    ("r5", lambda c: rk(c, 4)),
    ("r2_lev125", lambda c: rk(c, 1, lev=1.25)),
    ("r2_lev15", lambda c: rk(c, 1, lev=1.5)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=1500)
    ap.add_argument("--only", default="")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    cfg0 = load_config()
    ctx = prepare_context(cfg0, args.days)
    print(f"窗口 {len(ctx['main_dates'])} 日", flush=True)

    plan = PLAN
    if args.only:
        keep = {s.strip() for s in args.only.split(",")}
        plan = [x for x in PLAN if x[0] in keep]

    tag = args.tag or str(args.days)
    out = {"days": args.days, "n_days": len(ctx["main_dates"]), "variants": {}}
    print(f"{'变体':12s}{'总收益%':>14s}{'年化%':>9s}{'回撤%':>9s}{'卡玛':>7s}"
          f"{'笔数':>6s}{'单笔%':>8s}{'胜率%':>7s}", flush=True)
    for name, fn in plan:
        c = copy.deepcopy(cfg0)
        fn(c)
        try:
            r = _row(run_backtest(c, args.days, ctx))
        except Exception as e:                       # noqa: BLE001
            print(f"{name:12s} ERROR {e}", flush=True)
            continue
        out["variants"][name] = r
        print(f"{name:12s}{r['总收益%']:>14,.0f}{r['年化%']:>8.1f}%{r['回撤%']:>8.1f}%"
              f"{r['卡玛']:>7.2f}{r['笔数']:>6d}{r['单笔%']:>+8.2f}%{r['胜率%']:>6.1f}%",
              flush=True)

    p = ROOT / f"output/exp_rank_hist_{tag}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"写出 {p}", flush=True)


if __name__ == "__main__":
    main()
