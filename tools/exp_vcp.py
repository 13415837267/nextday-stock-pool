# -*- coding: utf-8 -*-
"""右侧最后一个没测过的形态：波动收缩 + 缩量贴新高（VCP，第187轮）。

为什么它和前面 86 个变体不同
---------------------------
前面测遍的右侧形态都要求"价格已经动起来"：
    追突破   : pct +1~+7, vol_ratio >= 1.5, 创 20 日新高   -> 买在情绪最热处
    趋势回踩 : pct -3~+1, 缩量回踩 5 日线                  -> 买在趋势内的回撤
VCP(O'Neil 波动收缩形态) 要求的是**价格不动**：贴住 20 日高点(-3% 以内)但
振幅 <= 4%、量比 <= 0.9。这是"上涨中继里的惜售" —— 浮筹被消化完、卖压枯竭，
下一步大概率向上突破。它是右侧形态里唯一"不追高也不逆势"的一种。

同样在 T+1 低开承接（同一个成交引擎、同一套成本、同一个闸门），所以对比是干净的。

判据：单笔 > 1.0% 且笔数 > 250（1500 日）才算第二条腿。

用法:
    python3 tools/exp_vcp.py --days 1500
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


def vc(c, pct=(-2.0, 3.0), dh=-3.0, amp=4.0, vr=0.9, mom=(5.0, 40.0),
       gap=10.0, lev=None):
    c["selection"]["mode"] = "vcp"
    c["selection"]["fallbacks"] = []
    c["selection"]["fallback_slot"] = True
    c["selection"]["vcp_pct_min"], c["selection"]["vcp_pct_max"] = pct
    c["selection"]["vcp_dist_high_min"] = dh
    c["selection"]["vcp_amp_max"] = amp
    c["selection"]["vcp_vol_ratio_max"] = vr
    c["selection"]["vcp_mom20_min"], c["selection"]["vcp_mom20_max"] = mom
    c["execution"]["skip_open_gain"] = gap
    if lev is not None:
        c["execution"]["leverage"] = lev


PLAN = [
    ("base_dip", lambda c: None),
    ("v1_def", lambda c: vc(c)),
    ("v2_g0", lambda c: vc(c, gap=0.0)),       # 只买低开(等回调才买)
    ("v3_g2", lambda c: vc(c, gap=2.0)),
    ("v4_tight", lambda c: vc(c, dh=-1.5, amp=3.0, vr=0.7)),
    ("v5_loose", lambda c: vc(c, dh=-5.0, amp=6.0, vr=1.2)),
    ("v6_up", lambda c: vc(c, pct=(0.0, 4.0))),  # 只要微涨的惜售
    ("v7_lev125", lambda c: vc(c, lev=1.25)),
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
    print(f"{'变体':12s}{'总收益%':>12s}{'年化%':>9s}{'回撤%':>9s}{'卡玛':>7s}"
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
        print(f"{name:12s}{r['总收益%']:>12,.0f}{r['年化%']:>8.1f}%{r['回撤%']:>8.1f}%"
              f"{r['卡玛']:>7.2f}{r['笔数']:>6d}{r['单笔%']:>+8.2f}%{r['胜率%']:>6.1f}%",
              flush=True)

    p = ROOT / f"output/exp_vcp_{tag}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"写出 {p}", flush=True)


if __name__ == "__main__":
    main()
