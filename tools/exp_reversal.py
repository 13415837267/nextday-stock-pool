# -*- coding: utf-8 -*-
"""第二个独立策略的第二条候选：放量恐慌反转（reversal）在主策略同口径引擎下（第186轮）。

为什么测它
----------
"右侧"被 64 个变体否决后，能在 H1 结构下独立达到百倍的第二条路只剩「另一种错杀」。
主策略吃的是**缩量阴跌**的错杀（无量下跌 → 次日低开是流动性冲击 → 修复）；
reversal 吃的是**放量恐慌**的宣泄（放量深跌 + 长下影 + 收盘拉回 → 恐慌盘出清 → 修复）。

两者的信号在物理上互斥（缩量 vs 放量、趋势内 vs 趋势外、浅跌 vs 深跌），
成交都在 T+1 深低开，是**两种不同的钱**，可以作为两个独立策略并行。

用法: python3 tools/exp_reversal.py --days 1500
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=1500)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cfg0 = load_config()
    ctx = prepare_context(cfg0, args.days)
    print(f"窗口 {len(ctx['main_dates'])} 日", flush=True)

    def rv(c, pct=(-9.5, -4.0), vr=1.2, cp=0.60, lt=0.8, mom=5.0, dl=15.0,
           floor=None, gap=None, lev=None):
        c["selection"]["mode"] = "reversal"
        c["selection"]["fallbacks"] = []
        c["selection"]["fallback_slot"] = True
        c["selection"]["rev_pct_min"], c["selection"]["rev_pct_max"] = pct
        c["selection"]["rev_vol_ratio_min"] = vr
        c["selection"]["rev_close_pos_min"] = cp
        c["selection"]["rev_low_tail_min"] = lt
        c["selection"]["rev_mom20_max"] = mom
        c["selection"]["rev_dist_low20_max"] = dl
        if floor is not None:
            c["execution"]["buy_gap_below"] = floor
        if gap is not None:
            c["execution"]["skip_open_gain"] = gap
        if lev is not None:
            c["execution"]["leverage"] = lev

    plan = [
        ("base_dip", lambda c: None),
        ("rv_def", lambda c: rv(c)),
        ("rv_f0", lambda c: rv(c, floor=0.0)),
        ("rv_fm1", lambda c: rv(c, floor=-1.0)),
        # 更严的恐慌: 更大放量
        ("rv_vr2", lambda c: rv(c, vr=2.0)),
        ("rv_vr3", lambda c: rv(c, vr=3.0)),
        # 收盘拉回更强
        ("rv_cp08", lambda c: rv(c, cp=0.8)),
        # 长下影更长
        ("rv_lt2", lambda c: rv(c, lt=2.0)),
        # 更深的跌
        ("rv_deep", lambda c: rv(c, pct=(-11.0, -6.0), vr=2.0, cp=0.7)),
        # 组合: 放量+拉回+长下影
        ("rv_combo", lambda c: rv(c, pct=(-11.0, -5.0), vr=2.0, cp=0.75, lt=1.5)),
        ("rv_combo_g0", lambda c: rv(c, pct=(-11.0, -5.0), vr=2.0, cp=0.75, lt=1.5, gap=0.0)),
        # 去闸门
        ("rv_nogate", lambda c: (rv(c, pct=(-11.0, -5.0), vr=2.0, cp=0.75, lt=1.5),
                                 c["regime"].__setitem__("enabled", False))),
        # 杠杆
        ("rv_lev15", lambda c: rv(c, pct=(-11.0, -5.0), vr=2.0, cp=0.75, lt=1.5, lev=1.5)),
        ("rv_lev20", lambda c: rv(c, pct=(-11.0, -5.0), vr=2.0, cp=0.75, lt=1.5, lev=2.0)),
    ]

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

    p = Path(args.out or f"output/exp_reversal_{args.days}.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"写出 {p}", flush=True)


if __name__ == "__main__":
    main()
