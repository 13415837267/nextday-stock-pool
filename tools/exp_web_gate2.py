# -*- coding: utf-8 -*-
"""第184轮·G: 入口的另一半 —— 两融能否"替代"MA20 闸门(而不是叠加)

第四组实验(F)证明: 在 MA20 闸门之上**再叠**一层两融闸门, 放行率只剩 10~40%,
笔数腰斩, 收益从 7.3 倍掉到 1.7~3.2 倍 —— 叠法是错的。

但两融闸门本身有质量: rzcalm(杠杆不过热) 单笔 +3.27%、回撤 -17.4%(基线 +2.55% / -21.6%)。
于是换一个问题: **去掉 MA20 闸门, 只留两融**, 会不会既增加笔数、又保住质量?
(第 181 轮测过"完全无闸门"很差, 但那是纯裸奔; 两融是慢变量, 可能补上 MA20 缺的那部分)

变体: 关掉 market_gate + gate_exit, 只留两融 mask。
  裸奔            —— 对照(应最差, 用于确认闸门本身的价值)
  rz5>0 / rz20>0 / rzcalm / rz_5>-1%

用法: python tools/exp_web_gate2.py 1500
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context  # noqa: E402
from src.config import load_config  # noqa: E402
from src.satellite import entry_ok_series  # noqa: E402
import tools.satellite_backtest as sb  # noqa: E402
from tools.exp_web_margin import load_margin, masks  # noqa: E402


def yearly(res: dict) -> dict:
    out = {}
    for pt in res["equity"]:
        y = pt["d"][:4]
        out.setdefault(y, [pt["nav"], pt["nav"]])
        out[y][1] = pt["nav"]
    return {y: round((v[1] / v[0] - 1) * 100, 1) for y, v in out.items()}


def run_one(ctx, cfg, sat: dict, extra: pd.Series | None = None):
    c = copy.deepcopy(cfg)
    c.setdefault("satellite", {}).update(sat)
    e = entry_ok_series(ctx.get("reg"), c)
    if extra is not None:
        ex = extra.copy()
        ex.index = ex.index.astype(str)
        e = e.copy()
        idx = e.index.astype(str)
        e.index = idx
        e = e & ex.reindex(idx).fillna(False)
    res = sb.run(dict(ctx), c, entry_ok=e)
    m = res["metrics"]
    return {"总收益%": m["总收益率%"], "年化%": m["年化收益率%"], "回撤%": m["最大回撤%"],
            "卡玛": m["卡玛比率"], "笔数": m["交易次数"], "胜率%": m["胜率%"],
            "单笔%": m["平均单笔收益%"], "空仓天": m["空仓天数"],
            "放行率%": round(100.0 * float(e.mean()), 1), "年度": yearly(res)}


FULL = {"mode": "lowvol", "max_pos": 1, "buys_per_day": 1, "hold_days": 10,
        "ivol_pct_max": 0.1, "sort_by": "mom20_ivol", "min_score": 0,
        "pullback_pct": 1.0, "pullback_days": 3, "pullback_mode": "low",
        "market_gate": True, "gate_entry_buffer": 0.3,
        "gate_exit": True, "gate_exit_mode": "all",
        "trail_pct": 8.0, "stop_pct": -12.0, "min_pool": 20,
        "exit_limit_pct": 0.0}
OFF = {"market_gate": False, "gate_exit": False}


def main():
    cfg = load_config()
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    ctx = prepare_context(cfg, days)
    print(f"实际 {len(ctx['main_dates'])} 日 {ctx['main_dates'][0]} ~ {ctx['main_dates'][-1]}",
          flush=True)
    m = load_margin()
    by_date = ctx["by_date"]
    mkt = pd.Series({str(d): float(v["pct"].mean()) for d, v in by_date.items()}).sort_index()
    mk = masks(m, mkt)
    rz = m["rzye"].astype(float)
    mk["rz5_gt-1"] = rz.pct_change(5) > -0.01

    out = {"days": len(ctx["main_dates"]), "variants": {}}
    plan = [("base(MA20闸门)", FULL, None),
            ("no_gate", {**FULL, **OFF}, None)]
    plan += [(f"no_gate+{k}", {**FULL, **OFF}, mk[k]) for k in ("rz5_up", "rz20_up", "rzcalm", "rz5_gt-1")]
    for nm, sat, ex in plan:
        r = run_one(ctx, cfg, sat, ex)
        out["variants"][nm] = r
        print(f"{nm:18s} 总{r['总收益%'] or 0:>10.1f}% 年化{r['年化%'] or 0:7.1f}% "
              f"回撤{r['回撤%'] or 0:6.1f}% 卡玛{r['卡玛'] or 0:5.2f} 笔{r['笔数'] or 0:4d} "
              f"胜{r['胜率%'] or 0:5.1f}% 单笔{r['单笔%'] or 0:+.2f}% 放行{r['放行率%']:.0f}%",
              flush=True)
    p = ROOT / f"output/exp_web_gate2_{days}.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"写出 {p}", flush=True)


if __name__ == "__main__":
    main()
