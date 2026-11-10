# -*- coding: utf-8 -*-
"""左侧抄底：放量深跌次日的低开承接（第187轮）。

诊断先行
--------
第 186 轮的 reversal 只有 3 笔/1500 日，原因已定位：不是"信号少"，是
`close_pos >= 0.60`（深跌还收在日内上半部）与 `low_tail >= 0.8`（长下影）
这两个条件在 A 股几乎不可能同时满足 —— 实测某日主板跌 3% 以上的 20 只票里，
close_pos >= 0.5 的有 **0 只**。深跌的票收盘都在日内低位，这是结构性事实。

所以本轮把"盘中承接"这层去掉，只保留"深跌 + 放量"，把 alpha 让给**次日低开**
（这正是主策略 dip 赚钱的同一个微观结构）。两种形态物理互斥：

    dip      : pct -6~-3, vol_ratio <= 4（缩量阴跌），趋势内（mom20 5~25）
    reversal : pct -9~-3, vol_ratio >= 1.5（放量宣泄），趋势外（mom20 <= 20）

一个是"没人卖"，一个是"抢着卖"。都在 T+1 低开接，但接的是两种完全不同的钱。

用法:
    python3 tools/exp_rev_grid.py --days 1500
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


def rv(c, pct=(-9.0, -3.0), vr=1.5, cp=0.0, lt=0.0, mom=20.0, dl=60.0,
       gap=10.0, lev=None):
    c["selection"]["mode"] = "reversal"
    c["selection"]["fallbacks"] = []
    c["selection"]["fallback_slot"] = True
    c["selection"]["rev_pct_min"], c["selection"]["rev_pct_max"] = pct
    c["selection"]["rev_vol_ratio_min"] = vr
    c["selection"]["rev_close_pos_min"] = cp
    c["selection"]["rev_low_tail_min"] = lt
    c["selection"]["rev_mom20_max"] = mom
    c["selection"]["rev_dist_low20_max"] = dl
    c["execution"]["skip_open_gain"] = gap
    if lev is not None:
        c["execution"]["leverage"] = lev


PLAN = [
    ("base_dip", lambda c: None),
    ("r1_all", lambda c: rv(c)),                                    # 全买(缺口<10%都买)
    ("r2_vr2", lambda c: rv(c, vr=2.0)),                            # 更放量
    ("r3_g0", lambda c: rv(c, gap=0.0)),                            # 只买低开
    ("r4_g2", lambda c: rv(c, gap=2.0)),                            # 低开或平开
    ("r5_shallow", lambda c: rv(c, pct=(-6.0, -3.0))),              # 浅一点
    ("r6_cp2", lambda c: rv(c, cp=0.2, lt=0.1)),                    # 略要求承接
    ("r7_lev125", lambda c: rv(c, lev=1.25)),
    ("r8_lev15", lambda c: rv(c, lev=1.5)),
    ("r9_g0_lev125", lambda c: rv(c, gap=0.0, lev=1.25)),
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
    print(f"{'变体':14s}{'总收益%':>12s}{'年化%':>9s}{'回撤%':>9s}{'卡玛':>7s}"
          f"{'笔数':>6s}{'单笔%':>8s}{'胜率%':>7s}", flush=True)
    for name, fn in plan:
        c = copy.deepcopy(cfg0)
        fn(c)
        try:
            r = _row(run_backtest(c, args.days, ctx))
        except Exception as e:                       # noqa: BLE001
            print(f"{name:14s} ERROR {e}", flush=True)
            continue
        out["variants"][name] = r
        print(f"{name:14s}{r['总收益%']:>12,.0f}{r['年化%']:>8.1f}%{r['回撤%']:>8.1f}%"
              f"{r['卡玛']:>7.2f}{r['笔数']:>6d}{r['单笔%']:>+8.2f}%{r['胜率%']:>6.1f}%",
              flush=True)

    p = ROOT / f"output/exp_rev_grid_{tag}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"写出 {p}", flush=True)


if __name__ == "__main__":
    main()
