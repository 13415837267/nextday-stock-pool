# -*- coding: utf-8 -*-
"""创业板 dip 全史抢救：更严的大盘闸门能不能把 -75% 压到 -40% 以内（第186轮续四）。

全史（3052 日）实测：
  创业板 dip lev1.25 : 59976% / -67.0% / 卡玛 1.04
  创业板 dip lev1.5  : 156698% / -74.9% / 卡玛 1.12
  主板 dip lev1.25   : 3083513% / -27.0% / 卡玛 5.00

1500 日窗口（2020-08 起）创业板 103 倍看起来很美，但它避开了 2015 熔断与 2018 熊市。
全史 -75% 说明：20cm 品种的回撤在主升浪之外无法控制。

本轮只问一句：更严的闸门（宽度 / 波动上限 / 入场缓冲）能不能救回来。
若不能，创业板 dip 判死刑 —— "两个独立策略都百倍"这件事在本数据上不成立。

用法: python3 tools/exp_gem_save.py
"""
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
    return {"总收益%": round(g("总收益率%"), 1), "年化%": round(g("年化收益率%"), 1),
            "回撤%": round(g("最大回撤%"), 1), "卡玛": round(g("卡玛比率"), 2),
            "笔数": int(g("交易次数")), "单笔%": round(g("平均单笔收益%"), 2),
            "胜率%": round(g("胜率%"), 1)}


def main():
    cfg0 = load_config()
    days = 3052
    ctx = prepare_context(cfg0, days)
    print(f"全史 {len(ctx['main_dates'])} 日", flush=True)

    plan = [
        ("main_ref", None),
        ("g_b45v18", dict(breadth=0.45, volmax=1.8, lev=1.25)),
        ("g_b45v15", dict(breadth=0.45, volmax=1.5, lev=1.25)),
        ("g_b55v18", dict(breadth=0.55, volmax=1.8, lev=1.25)),
        ("g_b45v18_p9", dict(breadth=0.45, volmax=1.8, lev=1.25, pmin=-9.0)),
        ("g_b45v18_amt8", dict(breadth=0.45, volmax=1.8, lev=1.25, amt=0.8)),
    ]
    out = {}
    print(f"{'变体':16s}{'总收益%':>13s}{'年化%':>9s}{'回撤%':>9s}{'卡玛':>7s}"
          f"{'笔数':>6s}{'单笔%':>8s}{'胜率%':>7s}", flush=True)
    for name, kw in plan:
        c = copy.deepcopy(cfg0)
        if kw:
            c["universe"]["boards"] = ["创业板"]
            c["universe"]["amount_min_yi"] = kw.get("amt", 0.5)
            c["selection"]["dip_pct_min"] = kw.get("pmin", -10.0)
            c["selection"]["dip_pct_max"] = -4.0
            c["selection"]["dip_vol_ratio_max"] = 3.0
            c["execution"]["leverage"] = kw.get("lev", 1.25)
            c["regime"]["breadth_min"] = kw["breadth"]
            c["regime"]["vol_max"] = kw["volmax"]
        r = _row(run_backtest(c, days, ctx))
        out[name] = r
        print(f"{name:16s}{r['总收益%']:>13,.0f}{r['年化%']:>8.1f}%{r['回撤%']:>8.1f}%"
              f"{r['卡玛']:>7.2f}{r['笔数']:>6d}{r['单笔%']:>+8.2f}%{r['胜率%']:>6.1f}%",
              flush=True)

    p = Path("output/exp_gem_save.json")
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"写出 {p}", flush=True)


if __name__ == "__main__":
    main()
