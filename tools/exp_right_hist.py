# -*- coding: utf-8 -*-
"""右侧趋势回踩：全史（3053 日）窗口验证（第187轮）。

为什么必须做这一步
------------------
第 183/186 轮所有右侧实验都只在 1500 日窗口跑过。1500 日 H1 结构最多 750 笔，
而百倍线是 r(N) = 100^(1/N) - 1：

    N=352(现役 dip)  r=1.32%   实际 1.44%  ✓  -> 498 倍
    N=447(回踩)      r=1.04%   实际 0.65%  ✗  ->  18 倍
    N=900            r=0.51%
    N=1000           r=0.46%

1500 日只有 447 笔是**窗口长度**造成的（每年约 74 笔）。全史 3053 日约 900 笔，
同样的单笔 0.65% 就是 (1.0065)^900 = 340 倍。所以"右侧永远到不了百倍"这个结论
很可能只是**窗口切短了**造成的假象 —— 本轮直接验证。

用法:
    python3 tools/exp_right_hist.py --days 3053
    python3 tools/exp_right_hist.py --days 3053 --only pb_def
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


def pb(c, pct=(-3.0, 1.0), mom=(12.0, 45.0), d5=(-3.0, 2.0),
       vr=1.3, bias=8.0, gap=10.0, lev=None):
    c["selection"]["mode"] = "pullback"
    c["selection"]["fallbacks"] = []
    c["selection"]["fallback_slot"] = True
    c["selection"]["pb_pct_min"], c["selection"]["pb_pct_max"] = pct
    c["selection"]["pb_mom20_min"], c["selection"]["pb_mom20_max"] = mom
    c["selection"]["pb_dist_ma5_min"], c["selection"]["pb_dist_ma5_max"] = d5
    c["selection"]["pb_vol_ratio_max"] = vr
    c["selection"]["bias20_max"] = bias
    c["execution"]["skip_open_gain"] = gap
    if lev is not None:
        c["execution"]["leverage"] = lev


PLAN = [
    ("base_dip", lambda c: None),
    ("pb_def", lambda c: pb(c)),
    ("pb_g2", lambda c: pb(c, gap=2.0)),
    ("pb_p6", lambda c: pb(c, pct=(-6.0, -1.0), gap=2.0)),
    ("pb_m20", lambda c: pb(c, mom=(20.0, 50.0), gap=2.0)),
    ("pb_wide", lambda c: pb(c, pct=(-5.0, 0.0), mom=(8.0, 50.0), vr=1.6, gap=2.0)),
    ("pb_lev125", lambda c: pb(c, lev=1.25)),
    ("pb_lev15", lambda c: pb(c, lev=1.5)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=3053)
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

    p = ROOT / f"output/exp_right_hist_{tag}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"写出 {p}", flush=True)


if __name__ == "__main__":
    main()
