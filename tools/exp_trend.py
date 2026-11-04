# -*- coding: utf-8 -*-
"""上升趋势策略 + 双策略并联 —— 结构与时机验证(2026-11-05, 第一百三十二轮)。

用户诉求
--------
「现在策略都是选出超跌深低开反弹的, 能否加一个选出上升趋势的, 架构是每日两个策略都推荐
  top1, 如果跌的不到标准成交, 就看上升趋势的能否成交」

翻译成可验证的问题, 有三层:
  Q1 上升趋势本身有没有 alpha? —— mode=trend 单独当主力策略跑全窗口。
  Q2 它和 dip 是否正交? —— 同日信号重合率、补位笔的缺口/收益分布。
  Q3 "dip 未成交 -> trend 顶上" 这个时机是否比 "dip 无信号才顶上" 更好?
     第七十八~九十一轮的历史结论说"主力出票但缺口不达标的日子补位"很糟(逆向选择最强),
     但那是 buy_gap_below=-0.8(主力也挑单)时代; 现行 buy_gap_below=0, 主力几乎总成交,
     补位只发生在**风控日**(dip 要求缺口<=-2.5)与停牌/一字涨停日 —— 场景已经变了, 必须重测。

四个变体(同一 ctx, 只换配置)
---------------------------
  base  现行: dip + momentum/oversold/pullback 三池, fallback_slot=true(主力有票就不补位)
  T1    dip + trend, fallback_slot=false  ← 用户描述的架构: 主力未成交则 trend 顶上
  T2    trend 单独当主力(无备用池)         ← Q1
  T3    dip + trend, fallback_slot=true    ← 对照: 主力无信号才补位

trend 池的执行口径(与 dip 的关键差异)
  gap_own: true   不继承风控日的 block_gap_below=-2.5 —— 深低开对趋势票是"破坏"不是"折价"
  gap_above: -1.5 低开超过 1.5% 视为趋势破坏, 放弃
  gap_below: 3.0  高开超过 3% 不追
  仓位仍按风控 blocked_frac=0.40 / 池深分档执行(风控管仓位, 不管缺口口径)

用法
----
  python3 tools/exp_trend.py               # 1500 日
  python3 tools/exp_trend.py --days 3378   # 全史
"""
import argparse
import copy
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["BT_FULL_TRADES"] = "1"

from src.backtest import prepare_context, run_backtest  # noqa: E402
from src.config import load_config  # noqa: E402

FIELDS = ["总收益率%", "年化收益率%", "最大回撤%", "卡玛比率", "夏普比率",
          "交易次数", "胜率%", "平均单笔收益%", "风控空仓天数"]

TREND_FB = {
    "mode": "trend", "top_n": 1, "segment": "all",
    "gap_own": True, "gap_above": -1.5, "gap_below": 3.0,
}


def _run(cfg, days, ctx):
    t = time.time()
    res = run_backtest(cfg, days, ctx)
    res["_sec"] = round(time.time() - t)
    return res


def _exec(trades):
    return [t for t in trades if t.get("ret_pct") is not None]


def _stat(trades, label):
    """按池统计: 笔数 / 均值 / 胜率 / 加权收益合计。"""
    if not trades:
        return {"label": label, "n": 0}
    w = sum(t["ret_pct"] * t.get("pos_frac", 1) for t in trades)
    return {
        "label": label, "n": len(trades),
        "avg": round(sum(t["ret_pct"] for t in trades) / len(trades), 3),
        "win": round(100 * sum(1 for t in trades if t["ret_pct"] > 0) / len(trades), 1),
        "wsum": round(w, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=1500)
    ap.add_argument("--out", default="output/exp_trend.json")
    args = ap.parse_args()

    cfg0 = load_config()
    print(f"[1/5] 构建上下文 {args.days} 日 ...", flush=True)
    t0 = time.time()
    ctx = prepare_context(cfg0, args.days)
    print(f"      ctx ready {time.time() - t0:.0f}s", flush=True)

    variants = {}

    print("[2/5] base(现行 dip + 三池) ...", flush=True)
    variants["base"] = _run(copy.deepcopy(cfg0), args.days, ctx)

    print("[3/5] T1(dip + trend, 主力未成交即补位) ...", flush=True)
    c1 = copy.deepcopy(cfg0)
    c1["selection"]["fallback_slot"] = False
    c1["selection"]["fallbacks"] = [dict(TREND_FB)]
    variants["T1"] = _run(c1, args.days, ctx)

    print("[4/5] T2(trend 单独当主力) ...", flush=True)
    c2 = copy.deepcopy(cfg0)
    c2["selection"]["mode"] = "trend"
    c2["selection"]["fallbacks"] = []
    c2["selection"]["fallback_slot"] = True
    variants["T2"] = _run(c2, args.days, ctx)

    print("[5/5] T3(dip + trend, 仅主力无信号才补位) ...", flush=True)
    c3 = copy.deepcopy(cfg0)
    c3["selection"]["fallback_slot"] = True
    c3["selection"]["fallbacks"] = [dict(TREND_FB)]
    variants["T3"] = _run(c3, args.days, ctx)

    # ---- 指标表 ----
    print(f"\n=== 指标对比({args.days} 日) ===", flush=True)
    hdr = f"  {'指标':14s}" + "".join(f"{k:>15s}" for k in variants)
    print(hdr, flush=True)
    for f in FIELDS:
        line = f"  {f:14s}"
        for k, r in variants.items():
            v = r["metrics"].get(f)
            line += f"{str(v):>15s}"
        print(line, flush=True)

    # ---- 按池拆成交 ----
    print("\n=== 成交构成 ===", flush=True)
    detail = {}
    for k, r in variants.items():
        ex = _exec(r.get("trades_all", []))
        pools = Counter(t.get("pool", "main") for t in ex)
        rows = []
        for pool in sorted(pools):
            rows.append(_stat([t for t in ex if t.get("pool", "main") == pool], pool))
        allst = _stat(ex, "ALL")
        detail[k] = {"pools": rows, "all": allst, "n_exec": len(ex)}
        desc = " ".join(f"{p['label']}:n={p['n']}/{p.get('avg', 0):+.2f}%/{p.get('win', 0):.0f}%"
                        for p in rows)
        print(f"  {k:4s} {len(ex):4d} 笔 | {desc}", flush=True)

    # ---- T1 补位笔的画像: 发生在什么日子 ----
    print("\n=== T1 补位笔(trend 顶上)明细 ===", flush=True)
    t1_fb = [t for t in _exec(variants["T1"].get("trades_all", []))
             if t.get("pool", "main") != "main"]
    base_by_day = {t["buy_date"] for t in _exec(variants["base"].get("trades_all", []))}
    t1_by_day = {t["buy_date"] for t in _exec(variants["T1"].get("trades_all", []))}
    only_t1 = sorted(t1_by_day - base_by_day)
    print(f"  T1 补位 {len(t1_fb)} 笔 / base 没有成交的买入日新增 {len(only_t1)} 天", flush=True)
    if t1_fb:
        print(f"  {'买入日':10s} {'代码':11s} {'名称':9s} {'缺口%':>7s} {'仓':>5s} {'收益%':>7s}", flush=True)
        for t in t1_fb[:25]:
            print(f"  {t['buy_date']:10s} {t['code']:11s} {str(t['name'])[:9]:9s} "
                  f"{t.get('gap_pct', 0):7.2f} {t.get('pos_frac', 0):5.2f} {t['ret_pct']:7.2f}",
                  flush=True)
        if len(t1_fb) > 25:
            print(f"  ... 其余 {len(t1_fb) - 25} 笔略", flush=True)
        gaps = [t.get("gap_pct", 0) for t in t1_fb]
        print(f"  缺口分布: min {min(gaps):+.2f}% / 中位 {sorted(gaps)[len(gaps)//2]:+.2f}% / "
              f"max {max(gaps):+.2f}%", flush=True)

    # ---- 信号重合度: dip 与 trend 是否真的选不同的票 ----
    print("\n=== dip vs trend 信号重合(Q2) ===", flush=True)
    from src.selector import select_stocks  # noqa: E402
    ctx_days = ctx.get("by_date") or {}
    sample = sorted(ctx_days)[-250:]
    both = dip_n = tr_n = 0
    for d in sample:
        day = ctx_days[d]
        pd_, _ = select_stocks(d, day, cfg0, None, top_n=1, name_filter=False)
        ct = copy.deepcopy(cfg0)
        ct["selection"]["mode"] = "trend"
        ct["selection"]["fallbacks"] = []
        pt, _ = select_stocks(d, day, ct, None, top_n=1, name_filter=False)
        if pd_:
            dip_n += 1
        if pt:
            tr_n += 1
        if pd_ and pt and pd_[0]["code"] == pt[0]["code"]:
            both += 1
    print(f"  近 {len(sample)} 日: dip 有票 {dip_n} 天 / trend 有票 {tr_n} 天 / 同日且同票 {both} 天",
          flush=True)

    out = {
        "days": args.days,
        "variants": {k: {"metrics": r["metrics"], "sec": r["_sec"]} for k, r in variants.items()},
        "detail": detail,
        "t1_fill": t1_fb,
        "t1_new_days": only_t1,
        "overlap": {"days": len(sample), "dip": dip_n, "trend": tr_n, "same": both},
    }
    op = Path(args.out)
    if not op.is_absolute():
        op = ROOT / op
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"\n结果已写 {op} (总耗时 {time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
