# -*- coding: utf-8 -*-
"""创业板 dip 第二轮：把回撤压到 -32% 以内并做到百倍（第186轮续二）。

首轮最好 g_best（pct -10~-4 / 成交额≥0.3亿 / 量比≤3 / lev1.25）
= 5113% / -27.1% / 卡玛 3.49 / 252 笔 / 单笔 1.73%。
但它只有 51 倍 —— 差一半。百倍需要 lev1.5，而 lev1.5 的回撤会顶到 -35%~-40%。

本轮只做一件事：在不牺牲太多收益的前提下把回撤压下去，让 lev1.5 可用。
可用旋钮只有三个（主引擎无个股止损）：
  ① 大盘闸门更严（breadth_min / vol_max）
  ② 回调区间更宽（笔数分散 → 单笔波动摊薄）
  ③ 换手/成交额更严（只做流动性好的）

用法: python3 tools/exp_gem_tune2.py --days 1500
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

    def g(c, pct=(-10.0, -4.0), amt=0.3, vr=3.0, lev=1.25,
          breadth=None, volmax=None, turnmax=None, mom=None):
        c["universe"]["boards"] = ["创业板"]
        c["universe"]["amount_min_yi"] = amt
        c["selection"]["dip_pct_min"], c["selection"]["dip_pct_max"] = pct
        c["selection"]["dip_vol_ratio_max"] = vr
        c["execution"]["leverage"] = lev
        if breadth is not None:
            c["regime"]["breadth_min"] = breadth
        if volmax is not None:
            c["regime"]["vol_max"] = volmax
        if turnmax is not None:
            c["universe"]["turnover_max"] = turnmax
        if mom:
            c["selection"]["dip_mom20_min"], c["selection"]["dip_mom20_max"] = mom

    plan = [
        ("base_main", lambda c: None),
        ("gb_lv15", lambda c: g(c, lev=1.5)),
        ("gb_lv15_b35", lambda c: g(c, lev=1.5, breadth=0.35)),
        ("gb_lv15_b45", lambda c: g(c, lev=1.5, breadth=0.45)),
        ("gb_lv15_v18", lambda c: g(c, lev=1.5, volmax=1.8)),
        ("gb_lv15_p12", lambda c: g(c, pct=(-12.0, -4.0), lev=1.5)),
        ("gb_lv15_t8", lambda c: g(c, turnmax=8.0, lev=1.5)),
        ("gb_lv15_amt5", lambda c: g(c, amt=0.5, lev=1.5)),
        ("gb_lv15_b35v18", lambda c: g(c, lev=1.5, breadth=0.35, volmax=1.8)),
        ("gb_lv18_b35", lambda c: g(c, lev=1.8, breadth=0.35)),
        ("gb_lv20_b35v18", lambda c: g(c, lev=2.0, breadth=0.35, volmax=1.8)),
    ]

    out = {"days": args.days, "variants": {}}
    print(f"{'变体':16s}{'总收益%':>12s}{'年化%':>9s}{'回撤%':>9s}{'卡玛':>7s}"
          f"{'笔数':>6s}{'单笔%':>8s}{'胜率%':>7s}", flush=True)
    for name, fn in plan:
        c = copy.deepcopy(cfg0)
        fn(c)
        try:
            r = _row(run_backtest(c, args.days, ctx))
        except Exception as e:                       # noqa: BLE001
            print(f"{name:16s} ERROR {e}", flush=True)
            continue
        out["variants"][name] = r
        print(f"{name:16s}{r['总收益%']:>12,.0f}{r['年化%']:>8.1f}%{r['回撤%']:>8.1f}%"
              f"{r['卡玛']:>7.2f}{r['笔数']:>6d}{r['单笔%']:>+8.2f}%{r['胜率%']:>6.1f}%",
              flush=True)

    p = Path(args.out or f"output/exp_gem_tune2_{args.days}.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"写出 {p}", flush=True)


if __name__ == "__main__":
    main()
