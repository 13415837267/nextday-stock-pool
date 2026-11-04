# -*- coding: utf-8 -*-
"""信号漂移诊断: 趋势票/回调票在"裸信号"口径下的持有期收益曲线(2026-11-05)。

为什么还要跑这一步
------------------
tools/exp_trend.py 已经给出三个负面结论:
  · T2 trend 单独当主力 1500 日只有 +1.36%(单笔 -0.083%, 胜率 43.9%, 回撤 -43%);
  · T1 "dip 未成交 -> trend 顶上" 91 笔补位, 均值 -0.35%/胜率 43%, 把总收益从 7349% 打到 4942%;
  · 信号完全正交(近 250 日同日同票 0 天), 所以不是"重复选票"的问题。
但在下结论"趋势策略没用"之前, 必须先排除一个混淆: **它可能是被 T+2 的持有期憋死的**。
趋势跟随类策略的经典持有期是数周到数月, 而本系统强制 T+2 收盘清仓 ——
如果趋势信号在 k=10/20 日上转正, 那结论就不是"没有 alpha", 而是"这里的出口太早"。

本脚本剥离所有执行约束(不模拟资金/风控/仓位), 只回答一件事:
  T日收盘选出的 top1, **T+1 开盘买入**, 持有 k 个交易日后收盘卖出, 单笔净收益是多少。

口径
----
  · 两个策略各取 top1: dip(现行主力) vs trend(第一百三十二轮新增);
  · 买入价 = T+1 的 open_adj; 卖出价 = T+k+1 的 close_adj(k 个完整交易日);
  · 成本 = 单边滑点千1 × 2 + 佣金万2.5 × 2 + 印花税万5 = 0.30%/笔, 两边同扣;
  · 剔除 T+1 停牌/开盘一字涨停(买不到), 与回测同口径;
  · 按 signal_date 的风控状态分层(regime.trade_ok), 因为"补位"恰好只发生在风控日。
无未来函数: 选票只用 T 日收盘数据; 卖出价是仿真必需的事后价格。

用法
----
  python3 tools/exp_trend_hold.py               # 1500 日
  python3 tools/exp_trend_hold.py --days 3378   # 全史
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
os.environ["BT_FULL_TRADES"] = "1"

from src.backtest import prepare_context  # noqa: E402
from src.config import load_config  # noqa: E402
from src.selector import select_stocks  # noqa: E402

KS = [1, 2, 3, 5, 10, 20]
COST = 0.0030          # 往返成本 0.30%


def _px(sub, d):
    try:
        if d in sub.index:
            return sub.loc[d]
    except Exception:  # noqa: BLE001
        pass
    return None


def _agg(rows, k):
    v = [r["r%d" % k] for r in rows if r.get("r%d" % k) is not None]
    if not v:
        return None
    n = len(v)
    return {"n": n, "avg": round(sum(v) / n, 3),
            "win": round(100 * sum(1 for x in v if x > 0) / n, 1),
            "med": round(sorted(v)[n // 2], 3)}


def _f3(a):
    return ("%+.2f%%/%.0f%%/%d" % (a["avg"], a["win"], a["n"])) if a else "-"


def _f2(a):
    return ("%+.2f%%/%.0f%%" % (a["avg"], a["win"])) if a else "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=1500)
    ap.add_argument("--out", default="output/exp_trend_hold.json")
    args = ap.parse_args()

    cfg0 = load_config()
    print("[1/2] 构建上下文 %d 日 ..." % args.days, flush=True)
    t0 = time.time()
    ctx = prepare_context(cfg0, args.days)
    print("      ctx ready %.0fs" % (time.time() - t0), flush=True)

    by_date, per_code, reg = ctx["by_date"], ctx["per_code"], ctx["reg"]
    dates = sorted(by_date)
    pos = {d: i for i, d in enumerate(dates)}
    main_dates = ctx["main_dates"]

    cfg_dip = copy.deepcopy(cfg0)
    cfg_dip["selection"]["fallbacks"] = []
    cfg_tr = copy.deepcopy(cfg0)
    cfg_tr["selection"]["mode"] = "trend"
    cfg_tr["selection"]["fallbacks"] = []

    print("[2/2] 逐日取 top1 并展开持有期 ...", flush=True)
    recs = []
    for d in main_dates:
        day = by_date[d]
        for tag, cfg in (("dip", cfg_dip), ("trend", cfg_tr)):
            picks, _ = select_stocks(d, day, cfg, None, top_n=1, name_filter=False)
            if not picks:
                continue
            code = picks[0]["code"]
            i = pos[d]
            if i + 1 >= len(dates):
                continue
            d1 = dates[i + 1]
            sub = per_code.get(code)
            if sub is None:
                continue
            r1 = _px(sub, d1)
            if r1 is None or r1["is_paused"] == 1 or r1["vol"] <= 0:
                continue
            if r1["open"] >= r1["high_limit"] - 1e-9:
                continue
            buy = float(r1["open_adj"])
            if buy <= 0:
                continue
            sig = day.loc[day["ts_code"] == code, "close_adj"]
            gap = round((buy / float(sig.iloc[0]) - 1) * 100, 2) if len(sig) else None
            row = {"signal_date": d, "code": code, "strategy": tag,
                   "regime_ok": bool(reg.loc[d, "trade_ok"]) if d in reg.index else True,
                   "gap_pct": gap, "score": picks[0]["score"]}
            for k in KS:
                j = i + 1 + k
                rk = _px(sub, dates[j]) if j < len(dates) else None
                row["r%d" % k] = (round((float(rk["close_adj"]) / buy - 1) * 100 - COST * 100, 3)
                                  if rk is not None else None)
            recs.append(row)

    print("\n=== 裸信号持有期曲线(%d 日窗口, 已扣 %.2f%% 往返成本) ===" % (args.days, COST * 100),
          flush=True)
    print("  %-6s %-10s " % ("策略", "分层") + "".join("%18s" % ("k=%d" % k) for k in KS), flush=True)
    print("  %-6s %-10s " % ("", "") + "".join("%18s" % "均值/胜率/n" for _ in KS), flush=True)
    summary = {}
    layers = [("all", lambda r: True),
              ("风控日", lambda r: not r["regime_ok"]),
              ("非风控", lambda r: r["regime_ok"])]
    for tag in ("dip", "trend"):
        for layer, sel in layers:
            rows = [r for r in recs if r["strategy"] == tag and sel(r)]
            line = "  %-6s %-10s " % (tag, layer)
            summary["%s|%s" % (tag, layer)] = {}
            for k in KS:
                a = _agg(rows, k)
                summary["%s|%s" % (tag, layer)]["k%d" % k] = a
                line += "%18s" % _f3(a)
            print(line, flush=True)

    print("\n=== trend 按评分中位分层(检验门槛是否放得太松) ===", flush=True)
    tr = [r for r in recs if r["strategy"] == "trend" and r.get("r2") is not None]
    if tr:
        scores = sorted(r["score"] for r in tr)
        med = scores[len(scores) // 2]
        for label, sel in (("高分半", lambda r: r["score"] >= med),
                           ("低分半", lambda r: r["score"] < med)):
            rows = [r for r in tr if sel(r)]
            line = "  %-8s n=%4d " % (label, len(rows))
            for k in (1, 2, 3, 5):
                line += " k%d:%14s" % (k, _f2(_agg(rows, k)))
            print(line, flush=True)
        print("  评分中位 %.2f" % med, flush=True)

    out = {"days": args.days, "cost_pct": COST * 100, "summary": summary, "n_recs": len(recs)}
    op = Path(args.out)
    if not op.is_absolute():
        op = ROOT / op
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print("\n结果已写 %s (总耗时 %.0fs)" % (op, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
