# -*- coding: utf-8 -*-
"""右侧之第三/四种形态：涨停次日承接、放量突破次日承接（第186轮）。

为什么这两个还有戏
------------------
前两轮已经把「纯右侧」的两条路走死：
  · 追突破（trend）    : 11 个变体，最好单笔 +0.14%，需要 3300 笔才百倍
  · 趋势回踩（pullback）: 15 个变体，最好单笔 +0.65%，880%，卡玛 0.98

两者的共同点是**次日平开/高开成交** —— 买在别人也要买的位置，没有折价。
而主策略 dip 的单笔是 +1.44%，它赚的是"隔夜恐慌抛售造成的折价"，成交在**深低开**。

本脚本测两个"右侧确认 + 左侧成交"的混合形态，它们是**不同的钱**，可以当独立策略：
  A. limitup : T 日封死涨停（极强势，右侧确认），T+1 恐慌低开时承接
  B. spike   : T 日放量中阳收在当日高位（突破，右侧确认），T+1 回抽深低开承接

如果连这两个都不行，那"右侧在 A 股日频没有 alpha"就是硬结论。

用法: python3 tools/exp_right_hybrid.py --days 1500
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

    def m(c, mode, floor=None, gap=None, extra=None):
        c["selection"]["mode"] = mode
        c["selection"]["fallbacks"] = []
        c["selection"]["fallback_slot"] = True
        if floor is not None:
            c["execution"]["buy_gap_below"] = floor
        if gap is not None:
            c["execution"]["skip_open_gain"] = gap
        for k, v in (extra or {}).items():
            c["selection"][k] = v

    plan = [
        ("base_dip", lambda c: None),
        # --- A. 涨停次日承接 ---
        ("lu_def", lambda c: m(c, "limitup")),
        ("lu_f0", lambda c: m(c, "limitup", floor=0.0)),            # 不要求深低开
        ("lu_fm2", lambda c: m(c, "limitup", floor=-2.0)),
        ("lu_fm3", lambda c: m(c, "limitup", floor=-3.0)),
        ("lu_close", lambda c: m(c, "limitup", floor=-2.0,
                                 extra={"lu_pct_min": 9.5})),       # 只要真涨停(9.5%+)
        ("lu_low", lambda c: m(c, "limitup", floor=-2.0,
                               extra={"lu_mom20_max": 20.0})),      # 排除高位连板
        # --- B. 放量突破次日承接 ---
        ("sp_def", lambda c: m(c, "spike")),
        ("sp_f0", lambda c: m(c, "spike", floor=0.0)),
        ("sp_fm2", lambda c: m(c, "spike", floor=-2.0)),
        ("sp_fm3", lambda c: m(c, "spike", floor=-3.0)),
        # --- C. 强趋势票的深回调(仍叫 dip, 但选股更偏右侧) ---
        ("dip_m20", lambda c: m(c, "dip", extra={"dip_mom20_min": 20.0, "dip_mom20_max": 45.0})),
        ("dip_m25", lambda c: m(c, "dip", extra={"dip_mom20_min": 25.0, "dip_mom20_max": 60.0})),
        ("dip_deep", lambda c: m(c, "dip", extra={"dip_pct_min": -9.0, "dip_pct_max": -4.0})),
    ]

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

    p = Path(args.out or f"output/exp_right_hybrid_{args.days}.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"写出 {p}", flush=True)


if __name__ == "__main__":
    main()
