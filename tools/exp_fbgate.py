# -*- coding: utf-8 -*-
"""备用池承接线 vs 风控日承接线 —— 口径分叉审计(2026-11-03)。

研究问题
--------
实盘台账 2026-11 三笔全空仓, 回测月度表却给出 +1.57%。追下去发现:
  11-03 同一只 600285(羚锐制药, 备用池 fb1, 缺口 -0.18%), 实盘按风控日
  block_gap_below=-2.5 判定"缺口不够深"空仓, 回测却按开盘价成交(+3.42%, 40% 仓)。

根因(代码级)
------------
src/backtest.py 买入闸:
    _fl = p.get("_gap_below")
    _fl = floor if _fl is None else _fl
备用池票在挑选阶段被写入 p["_gap_below"] = fallbacks[i].gap_below, 而配置里
三个备用池都是 gap_below: 0(第九十二轮"与主力同口径"), 于是 _fl=0 直接覆盖了
当日的 floor(= 风控日的 -2.5), `if _fl and ...` 又因 0 为假直接放行。
实盘 src/daily_run.py 走的是 floor_now = block_gap_below, 对任何池都生效 —— 两边分叉。

本工具回答
----------
把"备用池承接线与当日 floor 取更严者"重跑一遍, 量化这条缝漏掉多少笔、多少钱:
  · base: 现行配置(fallbacks.gap_below=0)
  · fixed: fallbacks.gap_below=None → _fl 回落到当日 floor(风控日 -2.5, 平日 0)

用法
----
  python3 tools/exp_fbgate.py            # 1500 日
  python3 tools/exp_fbgate.py --days 3378
"""
import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["BT_FULL_TRADES"] = "1"   # 需要全量成交序列做逐笔比对

from src.backtest import prepare_context, run_backtest  # noqa: E402
from src.config import load_config  # noqa: E402

FIELDS = ["总收益率%", "年化收益率%", "最大回撤%", "卡玛比率", "夏普比率",
          "交易次数", "胜率%", "平均单笔收益%", "风控空仓天数"]


def _run(cfg, days, ctx):
    t = time.time()
    res = run_backtest(cfg, days, ctx)
    res["_sec"] = round(time.time() - t)
    return res


def _exec_key(t):
    return (t.get("buy_date"), t.get("code"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=1500)
    ap.add_argument("--out", default="output/exp_fbgate.json")
    args = ap.parse_args()

    cfg0 = load_config()
    print(f"[1/3] 构建上下文 {args.days} 日 ...", flush=True)
    t0 = time.time()
    ctx = prepare_context(cfg0, args.days)
    print(f"      ctx ready {time.time() - t0:.0f}s", flush=True)

    print("[2/3] base(现行配置) ...", flush=True)
    base = _run(copy.deepcopy(cfg0), args.days, ctx)

    print("[3/3] fixed(备用池回落当日 floor) ...", flush=True)
    cfg1 = copy.deepcopy(cfg0)
    for fb in cfg1["select"]["fallbacks"]:
        fb["gap_below"] = None
    fixed = _run(cfg1, args.days, ctx)

    mb, mf = base["metrics"], fixed["metrics"]
    rows = []
    for k in FIELDS:
        rows.append({"field": k, "base": mb.get(k), "fixed": mf.get(k)})

    print(f"\n=== 指标对比({args.days} 日) ===", flush=True)
    print(f"  {'指标':16s} {'base':>14s} {'fixed':>14s} {'差':>12s}", flush=True)
    for r in rows:
        b, f = r["base"], r["fixed"]
        try:
            d = f"{float(f) - float(b):+.2f}"
        except Exception:  # noqa: BLE001
            d = "-"
        print(f"  {r['field']:16s} {str(b):>14s} {str(f):>14s} {d:>12s}", flush=True)

    # 逐笔比对: fixed 之外多出来的成交(即被风控承接线拦掉的那些)
    eb = {_exec_key(t): t for t in base.get("trades_all", []) if t.get("ret_pct") is not None}
    ef = {_exec_key(t): t for t in fixed.get("trades_all", []) if t.get("ret_pct") is not None}
    dropped = [eb[k] for k in eb if k not in ef]
    added = [ef[k] for k in ef if k not in eb]
    dropped.sort(key=lambda t: t["buy_date"])

    n_fb = sum(1 for t in eb.values() if t.get("pool") != "main")
    print(f"\n=== 逐笔差异 ===", flush=True)
    print(f"  base 成交 {len(eb)} 笔(其中备用池 {n_fb} 笔) / fixed 成交 {len(ef)} 笔", flush=True)
    print(f"  消失 {len(dropped)} 笔 / 新增 {len(added)} 笔", flush=True)
    if dropped:
        avg = sum(t["ret_pct"] * t["pos_frac"] for t in dropped) / len(dropped)
        pos = sum(t["ret_pct"] * t["pos_frac"] for t in dropped)
        print(f"  消失笔的加权收益合计 {pos:+.2f}pp, 均值 {avg:+.3f}%/笔", flush=True)
        print(f"  {'买入日':10s} {'代码':11s} {'名称':8s} {'池':4s} {'缺口%':>7s} {'仓':>5s} {'收益%':>7s}", flush=True)
        for t in dropped[:15]:
            print(f"  {t['buy_date']:10s} {t['code']:11s} {str(t['name'])[:8]:8s} {str(t['pool']):4s} "
                  f"{t['gap_pct']:7.2f} {t['pos_frac']:5.2f} {t['ret_pct']:7.2f}", flush=True)
        if len(dropped) > 15:
            print(f"  ... 其余 {len(dropped) - 15} 笔略", flush=True)

    # 月度差异(近 6 月)
    mbase = {m["month"]: m["ret_pct"] for m in base.get("monthly", [])}
    mfix = {m["month"]: m["ret_pct"] for m in fixed.get("monthly", [])}
    months = [m for m in mbase if m in mfix][-6:]
    print(f"\n=== 月度收益(近 6 月) ===", flush=True)
    for m in months:
        print(f"  {m}  base {mbase[m]:7.2f}%   fixed {mfix[m]:7.2f}%   差 {mfix[m] - mbase[m]:+6.2f}pp", flush=True)

    out = {
        "days": args.days,
        "base": {"metrics": mb, "n_exec": len(eb), "n_fb": n_fb, "sec": base["_sec"]},
        "fixed": {"metrics": mf, "n_exec": len(ef), "sec": fixed["_sec"]},
        "rows": rows,
        "dropped": dropped,
        "added": added,
        "monthly_base": {m: mbase[m] for m in months},
        "monthly_fixed": {m: mfix[m] for m in months},
    }
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"\n结果已写 {out_path} (总耗时 {time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
