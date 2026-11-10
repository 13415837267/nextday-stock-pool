# -*- coding: utf-8 -*-
"""右侧趋势回踩：多槽位分散能否把 -47% 回撤压下来（第187轮）。

动机
----
1500 日 pb_def：447 笔、单笔 +0.65%、回撤 -47.6%、卡玛 0.98。
收益（880%）离百倍差很远，但**笔数只有 447 是窗口长度问题**（一年约 74 笔）。
全史 3053 日约 900 笔，同样单笔就是 (1.0065)^900 ≈ 340 倍。

回撤怎么压？单吊 1 只时每笔 100% 资产，波动全暴露。分散到 k 只：
  - 期望不变：(1 + r/k)^(kN) ≈ (1+r)^N（一阶），复利几乎不损失
  - 波动降 √k：回撤按同样比例收窄
所以"多槽位"是**降回撤工具**，不是提收益工具。这一轮验证它在右侧是否同样成立。

引擎口径：execution.alloc_frac_pct = 每笔占组合总资产的比例；
          execution.max_buy_per_day = 每日最多买几只。两者解耦（不按达标数摊），
          所以 k 槽位 = alloc 100/k + max_buy_per_day k。

用法:
    python3 tools/exp_right_hist.py --days 3053 --only pb_def --tag base   # 先跑基线
    python3 tools/exp_slot_hist.py --days 3053
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
       vr=1.3, bias=8.0, gap=10.0, lev=None, slots=1):
    c["selection"]["mode"] = "pullback"
    c["selection"]["fallbacks"] = []
    c["selection"]["fallback_slot"] = True
    c["selection"]["pb_pct_min"], c["selection"]["pb_pct_max"] = pct
    c["selection"]["pb_mom20_min"], c["selection"]["pb_mom20_max"] = mom
    c["selection"]["pb_dist_ma5_min"], c["selection"]["pb_dist_ma5_max"] = d5
    c["selection"]["pb_vol_ratio_max"] = vr
    c["selection"]["bias20_max"] = bias
    c["execution"]["skip_open_gain"] = gap
    if slots > 1:
        c["selection"]["top_n"] = max(int(c["selection"].get("top_n") or 5), slots * 3)
        c["selection"]["display_top_n"] = max(int(c["selection"].get("display_top_n") or 5), slots * 3)
        c["selection"]["exec_top_n"] = slots
        c["execution"]["max_buy_per_day"] = slots
        c["execution"]["alloc_frac_pct"] = round(100.0 / slots, 4)
    if lev is not None:
        c["execution"]["leverage"] = lev


PLAN = [
    ("s1", lambda c: pb(c)),
    ("s2", lambda c: pb(c, slots=2)),
    ("s3", lambda c: pb(c, slots=3)),
    ("s4", lambda c: pb(c, slots=4)),
    ("s1_lev125", lambda c: pb(c, lev=1.25)),
    ("s2_lev125", lambda c: pb(c, slots=2, lev=1.25)),
    ("s1_lev15", lambda c: pb(c, lev=1.5)),
    ("s2_lev15", lambda c: pb(c, slots=2, lev=1.5)),
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

    p = ROOT / f"output/exp_slot_hist_{tag}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"写出 {p}", flush=True)


if __name__ == "__main__":
    main()
