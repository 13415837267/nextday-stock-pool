# -*- coding: utf-8 -*-
"""右侧趋势回踩：把股票池从"主板"扩到全板块（第187轮）。

为什么这是最后一个真正没测的维度
--------------------------------
第 183/186 轮的右侧实验 **全部只用主板池**（config.universe.boards = ["主板"]）。
主板 A 股约 3100 只，创业板+科创板约 2300 只，其中 20cm 票的日均振幅是主板的 1.6 倍。
趋势类策略的 edge 与振幅正相关（同样的趋势延续，20cm 票给两倍空间），
而笔数与池子大小正相关 —— 复利对笔数是指数的：

    全史百倍需要年化 46.8% = 每年 76 笔 × 单笔 0.51%
    主板 pb：76 笔 × 0.27% -> 年化 22.7% -> 92%
    若池子翻倍到 152 笔、单笔保住 0.27% -> 年化 50.7% -> 250 倍 ✓

所以"右侧到不了百倍"这个判断可能只是**池子太小**造成的。本轮直接验证。

同时给出 20cm 专属口径（创业板+科创板），看 edge 是否随振幅放大。

用法:
    python3 tools/exp_right_boards.py --days 1500
    python3 tools/exp_right_boards.py --days 3053
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

ALL = ["主板", "创业板", "科创板"]
GEM_STAR = ["创业板", "科创板"]


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


def pb(c, boards=None, pct=(-3.0, 1.0), mom=(12.0, 45.0), d5=(-3.0, 2.0),
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
    if boards:
        c["universe"]["boards"] = list(boards)
    if lev is not None:
        c["execution"]["leverage"] = lev


PLAN = [
    ("main_board", lambda c: pb(c)),                                   # 旧口径(主板)
    ("all_board", lambda c: pb(c, boards=ALL)),                        # 全板块
    ("gem_star", lambda c: pb(c, boards=GEM_STAR)),                    # 只要 20cm
    ("all_lev125", lambda c: pb(c, boards=ALL, lev=1.25)),
    ("all_lev15", lambda c: pb(c, boards=ALL, lev=1.5)),
    ("all_wide", lambda c: pb(c, boards=ALL, pct=(-5.0, 0.0), mom=(8.0, 50.0), vr=1.6)),
    ("all_strict", lambda c: pb(c, boards=ALL, pct=(-5.0, -0.5), mom=(20.0, 50.0), vr=1.0)),
    ("all_g2", lambda c: pb(c, boards=ALL, gap=2.0)),
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

    p = ROOT / f"output/exp_right_boards_{tag}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"写出 {p}", flush=True)


if __name__ == "__main__":
    main()
