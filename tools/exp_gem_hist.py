# -*- coding: utf-8 -*-
"""创业板 dip：全史验证 + 参数平台检验（第186轮续三）。

1500 日最好 gb_lv15_amt5 = 10332% / -33.6% / 卡玛 3.52 / 245 笔 / 单笔 1.81% ——
百倍达成。但这只是"在 22 个变体里挑出来的最好一个"，必须过两关：

  关1 全史（3053 日，含 2015 熔断 / 2018 熊市 / 2024 微盘踩踏）：
      若全史年化跌到 30% 以下或回撤破 -60%，说明是近六年的产物。
  关2 参数平台：把 pct_min / 成交额门槛各扫 5 档，看是连续平台还是单点尖峰。
      尖峰 = 过拟合，平台 = 真结构。

用法: python3 tools/exp_gem_hist.py
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


def gem(c, pct_min=-10.0, amt=0.5, lev=1.5):
    c["universe"]["boards"] = ["创业板"]
    c["universe"]["amount_min_yi"] = amt
    c["selection"]["dip_pct_min"] = pct_min
    c["selection"]["dip_pct_max"] = -4.0
    c["selection"]["dip_vol_ratio_max"] = 3.0
    c["execution"]["leverage"] = lev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="output/exp_gem_hist.json")
    args = ap.parse_args()
    cfg0 = load_config()

    out = {"hist": {}, "plateau": {}}

    # ---- 关1: 全史 ----
    days = 3053
    ctx = prepare_context(cfg0, days)
    print(f"=== 全史 {len(ctx['main_dates'])} 日 ===", flush=True)
    print(f"{'变体':16s}{'总收益%':>14s}{'年化%':>9s}{'回撤%':>9s}{'卡玛':>7s}"
          f"{'笔数':>6s}{'单笔%':>8s}{'胜率%':>7s}", flush=True)
    for name, pmin, amt, lev in (("hist_main", None, None, None),
                                 ("hist_g_lv125", -10.0, 0.5, 1.25),
                                 ("hist_g_lv15", -10.0, 0.5, 1.5),
                                 ("hist_g_lv15_amt3", -10.0, 0.3, 1.5)):
        c = copy.deepcopy(cfg0)
        if pmin is not None:
            gem(c, pct_min=pmin, amt=amt, lev=lev)
        r = _row(run_backtest(c, days, ctx))
        out["hist"][name] = r
        print(f"{name:16s}{r['总收益%']:>14,.0f}{r['年化%']:>8.1f}%{r['回撤%']:>8.1f}%"
              f"{r['卡玛']:>7.2f}{r['笔数']:>6d}{r['单笔%']:>+8.2f}%{r['胜率%']:>6.1f}%",
              flush=True)

    # ---- 关2: 参数平台（1500 日） ----
    days = 1500
    ctx = prepare_context(cfg0, days)
    print(f"\n=== 参数平台 {len(ctx['main_dates'])} 日 ===", flush=True)
    print(f"{'变体':16s}{'总收益%':>14s}{'年化%':>9s}{'回撤%':>9s}{'卡玛':>7s}"
          f"{'笔数':>6s}{'单笔%':>8s}{'胜率%':>7s}", flush=True)
    for pmin in (-8.0, -9.0, -10.0, -11.0, -12.0):
        c = copy.deepcopy(cfg0)
        gem(c, pct_min=pmin, amt=0.5, lev=1.5)
        r = _row(run_backtest(c, days, ctx))
        out["plateau"][f"pct{pmin:+.0f}"] = r
        print(f"{'pct'+str(pmin):16s}{r['总收益%']:>14,.0f}{r['年化%']:>8.1f}%{r['回撤%']:>8.1f}%"
              f"{r['卡玛']:>7.2f}{r['笔数']:>6d}{r['单笔%']:>+8.2f}%{r['胜率%']:>6.1f}%",
              flush=True)
    for amt in (0.2, 0.3, 0.5, 0.8, 1.2):
        c = copy.deepcopy(cfg0)
        gem(c, pct_min=-10.0, amt=amt, lev=1.5)
        r = _row(run_backtest(c, days, ctx))
        out["plateau"][f"amt{amt}"] = r
        print(f"{'amt'+str(amt):16s}{r['总收益%']:>14,.0f}{r['年化%']:>8.1f}%{r['回撤%']:>8.1f}%"
              f"{r['卡玛']:>7.2f}{r['笔数']:>6d}{r['单笔%']:>+8.2f}%{r['胜率%']:>6.1f}%",
              flush=True)

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"写出 {p}", flush=True)


if __name__ == "__main__":
    main()
