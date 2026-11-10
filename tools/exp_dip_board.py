# -*- coding: utf-8 -*-
"""第二个独立策略：按板块切分的 dip（第186轮）。

为什么换这条思路
----------------
右侧（追突破/趋势回踩/涨停次日/放量突破/放量恐慌）累计 76 个变体全部失败，
最好单笔 +0.65%，而百倍需要 352 笔 × 1.44%——差 2.2 倍，是结构性的。

但"两个独立策略"不必是"左+右"。可以是一条 alpha 在两个**互不重叠的股票池**上：
主策略做主板（10% 涨跌幅），第二策略做创业板/科创板（20% 涨跌幅）。
两者的信号物理不重叠、持仓可完全分开、资金可分两本账，是真正的两个独立策略。

创业板 20cm 的日内波动是主板的 2 倍，"缩量阴线 → 次日深低开修复"这个机制
在 20cm 上应该有更大的幅员（同样是 -5% 的回调，在 20cm 票里只是"小波动"，
抛压性质不同），因此 pct 区间需要独立标定。

用法: python3 tools/exp_dip_board.py --days 1500
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

    def bd(c, boards, pct=None, mom=None, lev=None, amt=None):
        c["universe"]["boards"] = boards
        if pct:
            c["selection"]["dip_pct_min"], c["selection"]["dip_pct_max"] = pct
        if mom:
            c["selection"]["dip_mom20_min"], c["selection"]["dip_mom20_max"] = mom
        if lev is not None:
            c["execution"]["leverage"] = lev
        if amt is not None:
            c["universe"]["amount_min_yi"] = amt

    plan = [
        ("base_main", lambda c: None),
        ("gem_def", lambda c: bd(c, ["创业板"])),
        ("gem_w", lambda c: bd(c, ["创业板"], pct=(-9.0, -4.0))),
        ("gem_w2", lambda c: bd(c, ["创业板"], pct=(-12.0, -5.0))),
        ("gem_w3", lambda c: bd(c, ["创业板"], pct=(-10.0, -3.0))),
        ("gem_m", lambda c: bd(c, ["创业板"], pct=(-9.0, -4.0), mom=(10.0, 40.0))),
        ("gem_amt", lambda c: bd(c, ["创业板"], pct=(-9.0, -4.0), amt=0.5)),
        ("star_def", lambda c: bd(c, ["科创板"])),
        ("star_w", lambda c: bd(c, ["科创板"], pct=(-9.0, -4.0))),
        ("mg", lambda c: bd(c, ["主板", "创业板"])),
        ("mg_lev", lambda c: bd(c, ["主板", "创业板"], lev=1.5)),
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

    p = Path(args.out or f"output/exp_dip_board_{args.days}.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"写出 {p}", flush=True)


if __name__ == "__main__":
    main()
