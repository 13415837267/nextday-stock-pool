# -*- coding: utf-8 -*-
"""创业板 dip 的精调与杠杆（第186轮续）。

首轮扫描发现 gem_amt（创业板 + pct -9~-4 + 成交额≥0.5亿）在**杠杆 1.0** 下
就有 5414% / 卡玛 2.77 / 单笔 1.87% / 242 笔 —— 单笔赔率高于主策略的 1.44%，
差的只是杠杆（主策略用 1.25）与笔数（352 vs 242）。

本轮把它调到百倍，并做全史验证。

用法: python3 tools/exp_gem_tune.py --days 1500
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

    def g(c, pct=(-9.0, -4.0), amt=0.5, lev=None, mom=None, vr=None, cp=None):
        c["universe"]["boards"] = ["创业板"]
        c["universe"]["amount_min_yi"] = amt
        c["selection"]["dip_pct_min"], c["selection"]["dip_pct_max"] = pct
        if lev is not None:
            c["execution"]["leverage"] = lev
        if mom:
            c["selection"]["dip_mom20_min"], c["selection"]["dip_mom20_max"] = mom
        if vr is not None:
            c["selection"]["dip_vol_ratio_max"] = vr
        if cp is not None:
            c["selection"]["dip_close_pos_max"] = cp

    plan = [
        ("base_main", lambda c: None),
        ("g_lv10", lambda c: g(c, lev=1.0)),
        ("g_lv125", lambda c: g(c, lev=1.25)),
        ("g_lv15", lambda c: g(c, lev=1.5)),
        ("g_lv18", lambda c: g(c, lev=1.8)),
        # 参数微调
        ("g_p10", lambda c: g(c, pct=(-10.0, -4.0), lev=1.25)),
        ("g_p8", lambda c: g(c, pct=(-8.0, -3.5), lev=1.25)),
        ("g_amt3", lambda c: g(c, amt=0.3, lev=1.25)),
        ("g_vr3", lambda c: g(c, vr=3.0, lev=1.25)),
        ("g_cp4", lambda c: g(c, cp=0.4, lev=1.25)),
        ("g_m", lambda c: g(c, mom=(8.0, 40.0), lev=1.25)),
        # 组合微调
        ("g_best", lambda c: g(c, pct=(-10.0, -4.0), amt=0.3, vr=3.0, lev=1.25)),
    ]

    out = {"days": args.days, "variants": {}}
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

    p = Path(args.out or f"output/exp_gem_tune_{args.days}.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"写出 {p}", flush=True)


if __name__ == "__main__":
    main()
