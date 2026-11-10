# -*- coding: utf-8 -*-
"""右侧之第二种形态：强趋势缩量回踩（pullback）在完整引擎下的参数网格（第186轮）。

为什么还要测这个
----------------
"追突破"已在 exp_right_hunt.py 里被彻底否决（11 个变体，最好单笔 +0.14%，
需要 3300 笔才百倍，H1 上限 750 笔 —— 数学上不可能）。

但"右侧交易"不等于"追突破"。右侧的定义是**趋势已确认**，买入时机可以是：
  ① 突破新高（追高）—— 已否决
  ② 趋势中回踩均线（低吸）—— 本轮测

②与主策略 dip 的机制差别：
  dip      : pct -6~-3 的**深阴线** + 缩量，吃"恐慌抛售后的修复"，成交在次日深低开；
  pullback : pct -3~+1 的**浅回调** + 更严的缩量，趋势更强（mom20 12~45），成交在次日平开。
前者赚"错杀修复"，后者赚"趋势延续"，是两条不同的钱。

用法: python3 tools/exp_right_pull.py --days 1500
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

    plan = [
        ("base_dip", lambda c: None),
        ("pb_def", lambda c: pb(c)),
        ("pb_g2", lambda c: pb(c, gap=2.0)),
        ("pb_g05", lambda c: pb(c, gap=0.5)),
        ("pb_g0", lambda c: pb(c, gap=0.0)),
        # 趋势更强
        ("pb_m20", lambda c: pb(c, mom=(20.0, 50.0), gap=2.0)),
        ("pb_m30", lambda c: pb(c, mom=(30.0, 60.0), gap=2.0)),
        # 回调更深
        ("pb_p5", lambda c: pb(c, pct=(-5.0, -0.5), gap=2.0)),
        ("pb_p6", lambda c: pb(c, pct=(-6.0, -1.0), gap=2.0)),
        # 更严缩量
        ("pb_vr1", lambda c: pb(c, vr=1.0, gap=2.0)),
        ("pb_vr08", lambda c: pb(c, vr=0.8, gap=2.0)),
        # 贴线更紧
        ("pb_d5a", lambda c: pb(c, d5=(-1.5, 1.0), gap=2.0)),
        # 组合: 强趋势 + 深回调 + 严缩量
        ("pb_combo", lambda c: pb(c, pct=(-5.0, -0.5), mom=(20.0, 50.0), vr=1.0, gap=2.0)),
        ("pb_combo2", lambda c: pb(c, pct=(-5.0, -0.5), mom=(20.0, 50.0), vr=1.0, d5=(-1.5, 1.5), gap=0.5)),
        # 去掉大盘闸门, 看闸门贡献
        ("pb_nogate", lambda c: (pb(c, pct=(-5.0, -0.5), mom=(20.0, 50.0), vr=1.0, gap=2.0),
                                 c["regime"].__setitem__("enabled", False))),
        # 杠杆
        ("pb_lev15", lambda c: pb(c, pct=(-5.0, -0.5), mom=(20.0, 50.0), vr=1.0, gap=2.0, lev=1.5)),
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

    p = Path(args.out or f"output/exp_right_pull_{args.days}.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"写出 {p}", flush=True)


if __name__ == "__main__":
    main()
