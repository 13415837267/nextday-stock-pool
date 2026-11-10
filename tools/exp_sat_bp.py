# -*- coding: utf-8 -*-
"""右侧第五种形态：卫星框架下"趋势回调"的多持有期（第186轮）。

为什么还要测
------------
satellite 框架与 backtest 框架的差别不是选股，是**持有期与资金周转**：
  backtest : T+1 买、T+2 卖（H1），同一笔钱两天周转一次；
  satellite: 买后持 N 天，单吊时资金占用率低，但单笔赔率可以更大。

trend/pullback 在 H1 下失败，可能不是"信号没 alpha"，而是"H1 太短，
趋势延续需要时间"。第 133 轮测过 trend 的纯信号持有期（全部为负），
但 bo_pullback（强势股回调，与突破不同）从未在 H3/H5/H10 上测过，
也从未放开低波池限制（第 183 轮只在低波前 10% 里测）。

用法: python3 tools/exp_sat_bp.py --days 1500
"""
import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context  # noqa: E402
from src.config import load_config  # noqa: E402
from src.satellite import entry_ok_series  # noqa: E402
import tools.satellite_backtest as sb  # noqa: E402

BASE = {"mode": "bo_pullback", "max_pos": 1, "buys_per_day": 1, "hold_days": 5,
        "ivol_pct_max": 1.0, "sort_by": "ivol", "min_score": 0,
        "pullback_pct": 1.0, "pullback_days": 3, "pullback_mode": "low",
        "market_gate": True, "gate_entry_buffer": 0.3,
        "gate_exit": True, "gate_exit_mode": "all",
        "trail_pct": 8.0, "stop_pct": -12.0, "min_pool": 20,
        "exit_limit_pct": 0.0, "leverage": 1.0, "margin_rate": 0.055}


def run_one(ctx, cfg, sat):
    c = copy.deepcopy(cfg)
    c.setdefault("satellite", {}).update(sat)
    e = entry_ok_series(ctx.get("reg"), c)
    res = sb.run(dict(ctx), c, entry_ok=e)
    m = res["metrics"]
    return {"倍数": round(1.0 + m["总收益率%"] / 100.0, 2), "年化%": m["年化收益率%"],
            "回撤%": m["最大回撤%"], "卡玛": m["卡玛比率"], "笔数": m["交易次数"],
            "胜率%": m["胜率%"], "单笔%": m["平均单笔收益%"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=1500)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    cfg = load_config()
    ctx = prepare_context(cfg, args.days)
    print(f"窗口 {len(ctx['main_dates'])} 日", flush=True)

    plan = [
        # 现役副策略对照
        ("lowvol_h10", {**BASE, "mode": "lowvol", "ivol_pct_max": 0.1,
                        "sort_by": "mom20_ivol", "hold_days": 10}),
        # 右侧趋势回调: 持有期扫描(全市场)
        ("bp_h3", {**BASE, "hold_days": 3}),
        ("bp_h5", {**BASE, "hold_days": 5}),
        ("bp_h10", {**BASE, "hold_days": 10}),
        # 趋势更强
        ("bp_h5_m10", {**BASE, "hold_days": 5, "bp_mom_min": 10.0}),
        ("bp_h5_m20", {**BASE, "hold_days": 5, "bp_mom_min": 20.0}),
        # 回调更深
        ("bp_h5_d8", {**BASE, "hold_days": 5, "bp_high_min": -8.0, "bp_high_max": -2.0}),
        # 低波池内(与第183轮口径一致, 用于对照)
        ("bp_h5_iv10", {**BASE, "hold_days": 5, "ivol_pct_max": 0.1}),
        ("bp_h10_iv10", {**BASE, "hold_days": 10, "ivol_pct_max": 0.1}),
        # 追突破(satellite 版, 对照)
        ("bo_h3", {**BASE, "mode": "breakout", "hold_days": 3}),
        ("bo_h5", {**BASE, "mode": "breakout", "hold_days": 5}),
    ]

    out = {"days": args.days, "variants": {}}
    print(f"{'变体':12s}{'倍数':>8s}{'年化%':>9s}{'回撤%':>9s}{'卡玛':>7s}"
          f"{'笔数':>6s}{'单笔%':>8s}{'胜率%':>7s}", flush=True)
    for name, sat in plan:
        try:
            r = run_one(ctx, cfg, sat)
        except Exception as e:                       # noqa: BLE001
            print(f"{name:12s} ERROR {e}", flush=True)
            continue
        out["variants"][name] = r
        print(f"{name:12s}{r['倍数']:>8.2f}{r['年化%']:>8.1f}%{r['回撤%']:>8.1f}%"
              f"{r['卡玛']:>7.2f}{r['笔数']:>6d}{r['单笔%']:>+8.2f}%{r['胜率%']:>6.1f}%",
              flush=True)

    p = Path(args.out or f"output/exp_sat_bp_{args.days}.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"写出 {p}", flush=True)


if __name__ == "__main__":
    main()
