# -*- coding: utf-8 -*-
"""执行摩擦审计(第一百六十二轮) —— 回测的成交价, 实盘能不能拿到。

两个被审出来的系统性乐观偏差:
  1. **跌停日卖不出**: 卖出按收盘价结算, 从不检查当天是否封死跌停。副策略 284 笔里 5 笔命中,
     且**全部落在移动止损触发日**(跌停 -10% 必然击穿 -8% 阈值)。顺延到次日开盘, 合计 -20.57pp。
  2. **印花税算错时代**: config 的 stamp_tax=0.0005 是 2023-08-28 降税后口径, 此前是 0.001,
     每笔少算 0.05%。主策略 1500 日 7348.85% -> 6875.64%(年化 -2.27pp)。

用法:
  python tools/exp_friction.py            # 跑全部四组 + 审计
  python tools/exp_friction.py --audit    # 只做卖出日跌停审计
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from src.backtest import prepare_context          # noqa: E402
from src.config import load_config                # noqa: E402
import satellite_backtest as SB                   # noqa: E402

PANEL = ROOT / "data" / "panel"
EP = 1e-9


def audit_exits(trades: list[dict]) -> pd.DataFrame:
    """逐笔查卖出日的跌停状态。L1=一字跌停 L2=收盘封跌停 L3=触及跌停后拉回。"""
    rows = {}
    for d in sorted({t["exit_date"] for t in trades}):
        f = PANEL / f"{d}.csv.gz"
        if not f.exists():
            continue
        df = pd.read_csv(f, usecols=["ts_code", "high", "low", "close",
                                     "low_limit", "vol", "is_paused"])
        rows[d] = df.set_index("ts_code")
    out = []
    for t in trades:
        dd = rows.get(t["exit_date"])
        if dd is None or t["code"] not in dd.index:
            out.append({**t, "exit_state": "no_row"})
            continue
        x = dd.loc[t["code"]]
        hi, lo, cl, LL = (float(x["high"]), float(x["low"]),
                          float(x["close"]), float(x["low_limit"]))
        st = ("L1_一字跌停" if hi <= LL + EP else
              "L2_收盘跌停" if cl <= LL + EP else
              "L3_触及跌停" if lo <= LL + EP else "ok")
        out.append({**t, "exit_state": st, "e_close": cl, "e_ll": LL})
    return pd.DataFrame(out)


def dd_stats(nav: pd.Series) -> dict:
    c = nav / nav.cummax()
    dd = 1 - c
    return {"meanDD": round(float(dd.mean()) * 100, 2),
            "p90DD": round(float(np.percentile(dd, 90)) * 100, 2),
            "maxDD": round(float(dd.max()) * 100, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--out", default="output/exp_friction.json")
    args = ap.parse_args()

    cfg = load_config()
    days = int((cfg.get("backtest") or {}).get("trading_days", 1500))
    ctx = prepare_context(cfg, days)
    print(f"窗口 {len(ctx['main_dates'])} 日 {ctx['main_dates'][0]} ~ {ctx['main_dates'][-1]}")

    base = SB.run(ctx, cfg, defer_on_limit_down=False,
                  stamp_2023=False)               # 旧口径, 用于审计与对照
    if args.audit:
        aud = audit_exits(base["trades"])
        print(aud["exit_state"].value_counts().to_string())
        bad = aud[aud["exit_state"].str.startswith("L")]
        if len(bad):
            print(bad[["code", "name", "entry_date", "exit_date", "ret_pct",
                       "note", "exit_state"]].to_string(index=False))
        return

    rows = {}
    for nm, defer, stamp in [("BASE", False, False), ("DEFER", True, False),
                             ("DEFER+STAMP", True, True)]:
        r = SB.run(ctx, cfg, defer_on_limit_down=defer, stamp_2023=stamp)
        m = r["metrics"]
        nav = pd.Series({x["d"]: x["nav"] for x in r["equity"]}, dtype=float)
        rows[nm] = {**m, **dd_stats(nav)}
        print(f"{nm:14s} 年化 {m['年化收益率%']:6.2f}%  回撤 {m['最大回撤%']:7.2f}%  "
              f"卡玛 {m['卡玛比率']:5.2f}  平均水下 {rows[nm]['meanDD']:5.2f}%")

    # trail 存废: 净口径下再跑一次关闭
    c = load_config()
    c["satellite"]["trail_pct"] = 0
    r = SB.run(ctx, c)
    m = r["metrics"]
    nav = pd.Series({x["d"]: x["nav"] for x in r["equity"]}, dtype=float)
    rows["DEFER+STAMP_NOTRAIL"] = {**m, **dd_stats(nav)}
    print(f"{'DEFER+STAMP_NOTRAIL':14s} 年化 {m['年化收益率%']:6.2f}%  "
          f"回撤 {m['最大回撤%']:7.2f}%  卡玛 {m['卡玛比率']:5.2f}  "
          f"平均水下 {rows['DEFER+STAMP_NOTRAIL']['meanDD']:5.2f}%")

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1, default=float),
                   encoding="utf-8")
    print(f"写出 {out}")


if __name__ == "__main__":
    main()
