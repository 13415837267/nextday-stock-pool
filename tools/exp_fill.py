"""第一百七十七轮: 主策略"出票但接不到"的日子, 副策略能否补位。

背景(第一百七十七轮诊断):
  1500 日内主策略出票 1236 天(82.4%), 实际成交只有 352 天 —— 884 天是
  "形态有票、但 T+1 竞价缺口没到承接线 -2.5%" 而跳过。第 91 轮的 fallback_slot
  让这些日子连备用池都不发牌, 于是页面长期显示"今日无计划"。

  第 175 轮已证: 主策略那 702 笔"缺口未达"的票硬买是 -0.606%/t=-4.12 —— 不能补。
  但**换一批票**(低波池追高)在这 884 天里买, 还没测过。

口径: 全部统一 T+1 开盘买 / T+2 收盘卖(与主策略同口径), 便于逐笔对比。
      不看卫星自己的 10 日持有规则, 也不做组合资金约束 —— 这是纯截面统计。
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

from src.config import load_config          # noqa: E402
from src.backtest import prepare_context    # noqa: E402
from src.satellite import (                 # noqa: E402
    daily_params, entry_gate, pick_list,
)


def _px(day_df: pd.DataFrame, code: str):
    m = day_df[day_df["ts_code"] == code]
    return None if m.empty else m.iloc[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=1500)
    ap.add_argument("--out", default="output/exp_fill.json")
    a = ap.parse_args()

    cfg = load_config()
    ctx = prepare_context(cfg, a.days)
    by_date, dates = ctx["by_date"], ctx["main_dates"]
    pos = {d: i for i, d in enumerate(dates)}
    P = daily_params(cfg)

    bt = json.load(open(ROOT / "output" / "backtest.json"))
    rows = bt.get("trades_all") or []
    filled = {r["signal_date"] for r in rows if r.get("ret_pct") is not None}
    skipped = {r["signal_date"] for r in rows if r.get("ret_pct") is None}
    no_sig = [d for d in dates if d not in filled and d not in skipped]
    print(f"日子: 总{len(dates)} 成交{len(filled)} 出票跳过{len(skipped)} 完全无票{len(no_sig)}")

    def one(d: str, code: str):
        """T+1 开盘买 / T+2 收盘卖, 返回 (ret_pct, gap_pct, note)。"""
        i = pos.get(d)
        if i is None or i + 2 >= len(dates):
            return None
        bd, sd = dates[i + 1], dates[i + 2]
        rb = _px(by_date[bd], code)
        rs = _px(by_date[sd], code)
        r0 = _px(by_date[d], code)
        if rb is None or rs is None or r0 is None:
            return None
        o, c = float(rb["open_adj"]), float(rs["close_adj"])
        pc = float(rb["prev_close_adj"])
        if o <= 0 or c <= 0 or pc <= 0:
            return None
        o_raw = float(rb["open"])
        if o_raw >= float(rb["high_limit"]) - 1e-9:   # 一字涨停买不到
            return None
        if o_raw <= float(rb["low_limit"]) + 1e-9:    # 一字跌停买不到
            return None
        gap = (o / pc - 1.0) * 100.0
        # 成本: 佣金万2.5 x2 + 印花税万5 + 滑点千1 x2
        cost = (2.5 * 2 + 5 + 10) / 1e4
        ret = (c / o - 1.0 - cost) * 100.0
        return ret, gap

    def scan(tag: str, day_list):
        recs = []
        for d in day_list:
            cs = pick_list(by_date[d], cfg, (), 1)
            if not cs:
                continue
            c = cs[0]
            r = one(d, c["code"])
            if r is None:
                continue
            ret, gap = r
            ok, why = entry_gate(gap, cfg)
            recs.append({"signal_date": d, "code": c["code"], "score": c["score"],
                         "rank_pct": c["rank_pct"], "ret_pct": round(ret, 3),
                         "gap_pct": round(gap, 3), "gate_ok": bool(ok), "why": why,
                         "grp": tag})
        return recs

    out = []
    out += scan("skip", sorted(skipped))      # 主策略有票但跳过
    out += scan("nosig", no_sig)              # 主策略完全无票
    out += scan("filled", sorted(filled))     # 主策略成交日(基准对照)

    df = pd.DataFrame(out)
    print(f"\n候选覆盖: skip {sum(df.grp=='skip')} / nosig {sum(df.grp=='nosig')} "
          f"/ filled {sum(df.grp=='filled')}  (各自日子数 "
          f"{len(skipped)}/{len(no_sig)}/{len(filled)})")

    def stat(sub):
        if not len(sub):
            return dict(n=0)
        r = sub["ret_pct"].to_numpy()
        t = r.mean() / (r.std(ddof=1) / np.sqrt(len(r))) if len(r) > 1 else 0.0
        return dict(n=int(len(r)), mean=round(float(r.mean()), 3),
                    med=round(float(np.median(r)), 3),
                    win=round(float((r > 0).mean() * 100), 1),
                    t=round(float(t), 2),
                    gap=round(float(sub["gap_pct"].mean()), 2))

    print("\n=== 分组(同口径 T+1开→T+2收) ===")
    for g in ("skip", "nosig", "filled"):
        print(f"{g:7s}", stat(df[df.grp == g]))

    print("\n=== skip 组按卫星自身缺口闸门 ===")
    for ok in (True, False):
        print(f"gate_ok={ok}", stat(df[(df.grp == "skip") & (df.gate_ok == ok)]))

    print("\n=== skip 组按缺口分档 ===")
    s = df[df.grp == "skip"]
    edges = [-99, -2.0, -1.0, 0.0, 1.0, 2.0, 99]
    for lo, hi in zip(edges[:-1], edges[1:]):
        print(f"[{lo:>5},{hi:>4})", stat(s[(s.gap_pct >= lo) & (s.gap_pct < hi)]))

    print("\n=== skip 组按年份 ===")
    s = s.assign(y=s.signal_date.str[:4])
    for y, g in s.groupby("y"):
        print(" ", y, stat(g))

    print("\n=== skip 组按主干当天缺口(若主票可成交会是多少) ===")
    # 主策略跳过日的真实缺口: 从 note 里取
    main_gap = {}
    for r in rows:
        if r.get("ret_pct") is None and r.get("pool") == "main":
            n = r.get("note") or ""
            if "缺口" in n:
                try:
                    main_gap[r["signal_date"]] = float(n.split("缺口")[1].split("%")[0])
                except Exception:
                    pass
    s2 = s[s.signal_date.isin(main_gap)].copy()
    s2["mg"] = s2.signal_date.map(main_gap)
    edges = [-99, -2.0, -1.0, 0.0, 1.0, 2.0, 99]
    for lo, hi in zip(edges[:-1], edges[1:]):
        sub = s2[(s2.mg >= lo) & (s2.mg < hi)]
        print(f"主缺口[{lo:>5},{hi:>4})", stat(sub))

    Path(ROOT / a.out).write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n写出 {a.out} ({len(out)} 条)")


if __name__ == "__main__":
    main()
