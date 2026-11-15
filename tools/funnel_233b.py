"""第233轮 · 漏斗：2500 个交易日里，钱到底卡在哪一层

前 232 轮一直在问"能不能从已成交的 559 笔里挤出更多"。
本轮先问一个更基本的问题：2500 天里只有 559 笔，剩下 1941 天为什么没做？
按层拆开（交易日 → 风控放行 → 非持仓占用 → 候选可用 → 实际成交），
才能知道"笔数"这个乘数还有没有空间。
"""
from __future__ import annotations

import os
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context, run_backtest  # noqa: E402
from src.config import load_config  # noqa: E402


def cat(note: str) -> str:
    if not note:
        return "成交"
    if note.startswith("top1评分"):
        return "评分门槛"
    if note.startswith("top1领先"):
        return "领先门槛"
    if note in ("停牌/无成交",):
        return "停牌"
    if note == "开盘一字涨停":
        return "一字涨停"
    if note.startswith("开盘高开"):
        return "开盘高开"
    if note.startswith("开盘低开"):
        return "开盘低开"
    if "承接线" in note or "崩塌线" in note or "下限" in note:
        return "缺口分档"
    if note == "资金不足一手":
        return "资金不足"
    return re.sub(r"[-\d.]+%?", "", note)[:12] or note


def main() -> None:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 2500
    os.environ["BT_FULL_TRADES"] = "1"
    cfg = load_config()
    ctx = prepare_context(cfg, days)
    res = run_backtest(cfg, None, ctx)
    md = ctx["main_dates"]
    reg = ctx["reg"]

    ok_days = [d for d in md if bool(reg.loc[d, "trade_ok"])] if len(reg) else md
    ts = res["trades_all"]
    filled = [t for t in ts if t.get("ret_pct") is not None]
    skip = [t for t in ts if t.get("ret_pct") is None]

    # 持仓占用：buy -> sell 之间（不含 sell 当天，因为收盘才卖，当天可下新单？引擎按 T+1 买,T+2 卖）
    busy: set[str] = set()
    for t in filled:
        b, s = t.get("buy_date"), t.get("sell_date")
        if b and s:
            busy.add(b)          # 买入当天及次日都占资金
    print(f"\n交易日 {len(md)} 日 | 风控放行 {len(ok_days)} 日 ({len(ok_days)/len(md):.1%}) "
          f"| 实际成交 {len(filled)} 笔")

    # 按日归并：每个交易日的状态
    by_day: dict[str, dict] = {d: {"状态": "空仓无信号"} for d in md}
    for d in md:
        if d not in ok_days:
            by_day[d]["状态"] = "风控空仓"
    for t in filled:
        by_day.setdefault(t["buy_date"], {})["状态"] = "成交"
    cnt = Counter(v["状态"] for v in by_day.values())
    print("\n== 每日状态分布 ==")
    for k, v in cnt.most_common():
        print(f"  {k:8s} {v:5d} 日  {v/len(md):6.1%}")

    print("\n== 放弃原因（trades 明细内 skip 条数） ==")
    cc = Counter(cat(t.get("note", "")) for t in skip)
    for k, v in cc.most_common():
        print(f"  {k:10s} {v:5d}")
    print(f"  合计 skip {len(skip)} 条（同一信号日可能多条）")

    # 每日是否被"持仓占用"：放行日 - 成交日 - 当日有skip的日
    skip_days = {t.get("buy_date") for t in skip if t.get("buy_date")}
    used = set(by_day.keys())
    idle_ok = [d for d in ok_days if by_day[d]["状态"] != "成交" and d not in skip_days]
    print(f"\n放行但既未成交也无 skip 记录: {len(idle_ok)} 日（通常=当日无候选/持仓中）")

    # 月度成交数
    f = pd.DataFrame([{"d": t["buy_date"]} for t in filled])
    if len(f):
        f["ym"] = f["d"].str[:6]
        m = f.groupby("ym").size()
        print(f"\n月成交笔数: 均值 {m.mean():.1f} 中位 {m.median():.0f} 最大 {m.max()} "
              f"零成交月份 {int((m.reindex(pd.unique(f['ym'])).isna()).sum())}")
        print(f"  全年最高 {m.max()} 笔/月；按月分布 25/50/75 分位 "
              f"{m.quantile([.25,.5,.75]).round(1).to_dict()}")

    # 两融（市场级，2012-09 起）：与 559 笔收益的关系
    mf = sorted((ROOT / "data" / "margin").glob("*.csv.gz"))
    if mf:
        mm = pd.concat([pd.read_csv(p) for p in mf], ignore_index=True)
        mm["trade_date"] = mm["trade_date"].str.replace("-", "", regex=False)
        mm = mm.sort_values("trade_date").reset_index(drop=True)
        mm["rz_yi"] = mm["rzye"].astype(float)
        mm["rz_chg5"] = mm["rz_yi"].pct_change(5) * 100
        mm["rz_chg20"] = mm["rz_yi"].pct_change(20) * 100
        mm["rz_ratio"] = mm["rzbuy"].astype(float) / mm["rz_yi"] * 100
        mm.to_csv(ROOT / "output" / "w233_margin.csv", index=False, encoding="utf-8-sig")
        print(f"\n两融序列 {len(mm)} 日 {mm['trade_date'].min()}~{mm['trade_date'].max()}")
        tr = pd.read_csv(ROOT / "output" / "w232_trades.csv", dtype={"signal": str})
        j = tr.merge(mm[["trade_date", "rz_yi", "rz_chg5", "rz_chg20", "rz_ratio"]],
                     left_on="signal", right_on="trade_date", how="left")
        print(f"  559 笔中匹配 {j['rz_yi'].notna().sum()} 笔")
        for c in ("rz_chg5", "rz_chg20", "rz_ratio"):
            v = j[c].astype(float); y = j["ret"].astype(float)
            ok = v.notna()
            import numpy as np
            pr = float(np.corrcoef(v[ok], y[ok])[0, 1])
            t = pr * (ok.sum() - 2) ** 0.5 / max(1e-9, 1 - pr ** 2) ** 0.5
            bins = pd.qcut(v[ok], 5, labels=False, duplicates="drop")
            means = [round(float(y[ok][bins == i].mean()), 2) for i in range(bins.max() + 1)]
            print(f"  {c:9s} r={pr:+.3f} t={t:+.2f} 五档 {means}")


if __name__ == "__main__":
    main()
