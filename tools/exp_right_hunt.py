# -*- coding: utf-8 -*-
"""右侧追高：trend 模式在**完整引擎**下的参数网格（第186轮）。

背景
----
用户要求「一个左侧抄底、一个右侧追高，两个都要独立达到百倍」。
现役主策略 = dip（缩量阴线回调）= 左侧；副策略 = lowvol+mom20_ivol = 低波防御，
**不是右侧追高**。第133~139 轮的 trend 测试全部在"备用池补位"或"纯信号"口径下做，
从未在**与主策略完全同口径**（T+1 开盘买、T+2 收盘卖、单吊、MA20+0.3 闸门、杠杆1.25）跑过。

本脚本只问一件事：trend 单独当主力，在完整引擎里，最好能做到多少。
核心新变量 = **买入缺口上限**（skip_open_gain）。右侧追高最可能的死因是
"T+1 高开接盘"——突破次日常高开 1~3%，买在最高点的概率极大。把上限从 10% 一路压到 0%，
等于从「无条件追」到「只在平开/低开时追」，可分离"追强"与"接盘"。

用法: python3 tools/exp_right_hunt.py --days 1500
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

STRICT = {
    "tr_pct_min": 2.5, "tr_pct_max": 6.5,
    "tr_close_pos_min": 0.70,
    "tr_mom20_min": 8.0, "tr_mom20_max": 30.0,
    "tr_dist_high20_min": -0.5,
    "tr_vol_ratio_min": 1.2, "tr_vol_ratio_max": 2.5,
    "tr_bias20_max": 12.0,
}


def _row(res):
    m = res["metrics"]
    return {
        "总收益%": round(float(str(m["总收益率%"]).replace(",", "")), 1),
        "年化%": round(float(str(m["年化收益率%"]).replace(",", "")), 1),
        "回撤%": round(float(str(m["最大回撤%"]).replace(",", "")), 1),
        "卡玛": round(float(str(m["卡玛比率"]).replace(",", "")), 2),
        "笔数": int(str(m["交易次数"]).replace(",", "")),
        "单笔%": round(float(str(m["平均单笔收益%"]).replace(",", "")), 2),
        "胜率%": round(float(str(m["胜率%"]).replace(",", "")), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=1500)
    ap.add_argument("--out", default="")
    ap.add_argument("--tag", default="a")
    args = ap.parse_args()

    cfg0 = load_config()
    ctx = prepare_context(cfg0, args.days)
    print(f"窗口 {len(ctx['main_dates'])} 日", flush=True)

    plan = []

    def add(name, fn):
        plan.append((name, fn))

    add("base_dip", lambda c: None)                      # 现役主策略对照

    def _tr(c, gap=None, strict=False, lev=None):
        c["selection"]["mode"] = "trend"
        c["selection"]["fallbacks"] = []
        c["selection"]["fallback_slot"] = True
        for k, v in (STRICT if strict else {}).items():
            c["selection"][k] = v
        if gap is not None:
            c["execution"]["skip_open_gain"] = gap
        if lev is not None:
            c["execution"]["leverage"] = lev

    add("tr_def", lambda c: _tr(c))
    add("tr_g3", lambda c: _tr(c, gap=3.0))
    add("tr_g2", lambda c: _tr(c, gap=2.0))
    add("tr_g1", lambda c: _tr(c, gap=1.0))
    add("tr_g05", lambda c: _tr(c, gap=0.5))
    add("tr_g0", lambda c: _tr(c, gap=0.0))
    add("tr_gm1", lambda c: _tr(c, gap=-0.5))            # 只买低开 >0.5% 的
    add("trs_g2", lambda c: _tr(c, gap=2.0, strict=True))
    add("trs_g05", lambda c: _tr(c, gap=0.5, strict=True))
    add("trs_g10", lambda c: _tr(c, gap=10.0, strict=True))

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

    p = Path(args.out or f"output/exp_right_hunt_{args.tag}_{args.days}.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"写出 {p}", flush=True)


if __name__ == "__main__":
    main()
