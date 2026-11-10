# -*- coding: utf-8 -*-
"""右侧趋势回踩：把杠杆调到"百倍线"上（第187轮收尾）。

结论链条
--------
右侧形态本身的上限（1500 日、单吊、H1）：单笔 +0.65%、447 笔、卡玛 0.98。
    - 扩槽位降回撤 -> 单笔腰斩（0.29%），收益从 880% 掉到 95%
    - 扩板块增笔数 -> 单笔塌到 0.11%（20cm 池里 mom20/dist_ma5 评分失效）
    - 全史 3053 日   -> 92%、单笔 0.27%、回撤 -92.9%（2015-2019 动量是反转的）

所以右侧只剩一个自由度：**杠杆**。卡玛 0.98 不变的前提下，倍数是杠杆的单调函数，
代价是同比例放大的回撤。本轮把这条件量化清楚 —— 到底几倍杠杆能上百倍，回撤多大。

对照主策略卡玛 8.44：同样的 100 倍，主策略用 1.25 倍杠杆、-21.7% 回撤；
右侧要用 k 倍杠杆、-47.6%×k 回撤。这个差距来自 kama，不是参数。

用法:
    python3 tools/exp_right_lev.py --days 1500
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


def pb(c, lev=None, strict=False):
    c["selection"]["mode"] = "pullback"
    c["selection"]["fallbacks"] = []
    c["selection"]["fallback_slot"] = True
    c["selection"]["pb_pct_min"], c["selection"]["pb_pct_max"] = -3.0, 1.0
    c["selection"]["pb_mom20_min"], c["selection"]["pb_mom20_max"] = 12.0, 45.0
    c["selection"]["pb_dist_ma5_min"], c["selection"]["pb_dist_ma5_max"] = -3.0, 2.0
    c["selection"]["pb_vol_ratio_max"] = 1.3
    c["selection"]["bias20_max"] = 8.0
    c["execution"]["skip_open_gain"] = 10.0
    if strict:   # all_strict: 更严的选股(1500 日 500% / -30.3% / 卡玛 1.16, 右侧最高卡玛)
        c["selection"]["pb_pct_min"], c["selection"]["pb_pct_max"] = -5.0, -0.5
        c["selection"]["pb_mom20_min"], c["selection"]["pb_mom20_max"] = 20.0, 50.0
        c["selection"]["pb_vol_ratio_max"] = 1.0
        c["universe"]["boards"] = ["主板", "创业板", "科创板"]
        c["execution"]["skip_open_gain"] = 2.0
    if lev is not None:
        c["execution"]["leverage"] = lev


PLAN = [
    ("base_dip", lambda c: None),
    ("pb_lev10", lambda c: pb(c, lev=1.0)),
    ("pb_lev125", lambda c: pb(c, lev=1.25)),
    ("pb_lev175", lambda c: pb(c, lev=1.75)),
    ("pb_lev20", lambda c: pb(c, lev=2.0)),
    ("pb_lev25", lambda c: pb(c, lev=2.5)),
    ("st_lev125", lambda c: pb(c, lev=1.25, strict=True)),
    ("st_lev175", lambda c: pb(c, lev=1.75, strict=True)),
    ("st_lev20", lambda c: pb(c, lev=2.0, strict=True)),
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
    print(f"{'变体':12s}{'总收益%':>16s}{'年化%':>9s}{'回撤%':>9s}{'卡玛':>7s}"
          f"{'笔数':>6s}{'单笔%':>8s}", flush=True)
    for name, fn in plan:
        c = copy.deepcopy(cfg0)
        fn(c)
        try:
            r = _row(run_backtest(c, args.days, ctx))
        except Exception as e:                       # noqa: BLE001
            print(f"{name:12s} ERROR {e}", flush=True)
            continue
        out["variants"][name] = r
        print(f"{name:12s}{r['总收益%']:>16,.1f}{r['年化%']:>8.1f}%{r['回撤%']:>8.1f}%"
              f"{r['卡玛']:>7.2f}{r['笔数']:>6d}{r['单笔%']:>+8.2f}%", flush=True)

    p = ROOT / f"output/exp_right_lev_{tag}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"写出 {p}", flush=True)


if __name__ == "__main__":
    main()
