# -*- coding: utf-8 -*-
"""第 194 轮: 主策略的"买价"旋钮 —— 把副策略第 179 轮的限价回踩搬到 dip 上。

背景
----
第 193 轮把副策略的收益拆开: 106 笔单笔 +2.546%, 其中**限价回踩贡献 +0.34~0.54pp**、
大盘闸门 +0.44~0.48pp、排序 ≈0。也就是说副策略唯一被证实的"钱"在执行价, 不在选股。

主策略(dip) 至今仍是"T+1 开盘无条件按开盘价成交"(buy_gap_below=0)。它的持有期只有
1 个交易日(T+1 开盘买 / T+2 开盘卖), 买价差 1% 直接吃掉单笔 1.439% 的七成 —— 与副策略
第 179 轮之前"19% 天花板"的处境同构。所以问题是: **主策略能不能也挂限价?**

与 buy_gap_below 的区别(必须说清, 否则会被当成重复实验)
------------------------------------------------------
  buy_gap_below = -1.0  →  "缺口不够深就不买"(挑单), 但成交价仍是开盘价,
                             若开盘 -3% 就吃到 -3% 的坑;
  限价 收盘×(1-1%)        →  "买价钉死在 -1%": 开盘更低按开盘(更优), 盘中触及按限价,
                             没触及就当日空仓。既是挑单、也是压价。

本脚本只改**一个变量**: 买价。卖出日/卖出价/仓位系数/风控全部沿用引擎逐笔结果,
所以差额可以 100% 归因到买价。

用法: python3 -u tools/exp_bid_194.py [days]
"""
from __future__ import annotations

import json
import os
import statistics as st
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest import prepare_context, run_backtest   # noqa: E402
from src.config import ROOT, load_config                # noqa: E402

PB = [0.0, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]
LEV = 1.25


def main() -> None:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    os.environ["BT_FULL_TRADES"] = "1"
    cfg = load_config()
    print(f"== 第194轮 主策略买价实验: {days} 日 ==", flush=True)
    ctx = prepare_context(cfg, days)
    _from = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--from=")), None)
    if _from:   # 复用已跑好的回测(省一次引擎), trades_all 必须来自同参数
        res = json.loads(Path(_from).read_text(encoding="utf-8"))
        print(f"复用 {_from}: {res['meta']['days']} 日 {res['meta']['start']}~{res['meta']['end']}", flush=True)
    else:
        res = run_backtest(cfg, days, ctx)
    trades = [t for t in res["trades_all"]
              if t.get("buy_px") and t.get("ret_pct") is not None]
    print(f"引擎成交 {len(trades)} 笔, 单笔 {res['metrics']['平均单笔收益%']:+.3f}%", flush=True)

    per_code = ctx["per_code"]
    by_date = ctx["by_date"]

    # 信号日收盘(复权) —— 限价基准只能用信号日已知价, 不许读 T+1
    def sig_close(code: str, sd: str):
        sub = per_code.get(code)
        if sub is None or sd not in sub.index:
            return None
        return float(sub.loc[sd]["close_adj"])

    dates = ctx["main_dates"]
    didx = {dd: i for i, dd in enumerate(dates)}
    recs = []
    for t in trades:
        code, sd, bd = t["code"], t["signal_date"], t["buy_date"]
        sc = sig_close(code, sd)
        sub = per_code.get(code)
        if sc is None or sub is None or bd not in sub.index:
            continue
        r = sub.loc[bd]
        # per_code 只存 open_adj/close_adj, high/low 是未复权原价; 复权因子 = close_adj/close
        _f = float(r["close_adj"]) / float(r["close"]) if float(r["close"]) else 1.0
        o = float(r["open_adj"])
        hi, lo = float(r["high"]) * _f, float(r["low"]) * _f
        # ★坑: 引擎 trades 里的 buy_px / sell_px 是**未复权**价(backend.py:928 用 _lp/_f 还原成
        #   展示价), 直接拿它和复权 OHLC 相比会系统性偏低 7%(前复权基准差), 单笔虚增 +6.9pp。
        #   这里按买卖日各自的复权因子一并还原成复权价, 全流程同源。
        buy_px, sell_px, pos = float(t["buy_px"]), float(t["sell_px"]), float(t.get("pos_frac") or 1.0)
        buy_px *= _f
        _sd = t.get("sell_date")
        if _sd and sub is not None and _sd in sub.index:
            _r2 = sub.loc[_sd]
            sell_px *= float(_r2["close_adj"]) / float(_r2["close"]) if float(_r2["close"]) else 1.0
        else:
            sell_px = float("nan")
        # 由引擎实际结果反推成本(双边佣金+印花+滑点), 供新买价复用
        cost_pp = (sell_px / buy_px - 1.0) * 100.0 - float(t["ret_pct"])
        if not (sell_px == sell_px):
            continue
        recs.append({"sd": sd, "bd": bd, "code": code, "sc": sc, "o": o, "hi": hi, "lo": lo,
                     "buy": buy_px, "sell": sell_px, "pos": pos, "cost": cost_pp})

    print(f"可模拟 {len(recs)} 笔", flush=True)
    _chk = [(x["sell"] / x["buy"] - 1.0) * 100.0 - x["cost"] for x in recs]
    _tr = {(t["code"], t["buy_date"]): float(t["ret_pct"]) for t in trades}
    _d = [abs(a - _tr[(x["code"], x["bd"])]) for a, x in zip(_chk, recs)]
    print(f"复权还原自洽校验: 与引擎 ret_pct 差中位 {np.median(_d):.4f}pp, "
          f"P99 {np.percentile(_d, 99):.3f}pp, 最大 {max(_d):.3f}pp", flush=True)

    def _walk(pb: float, wait: int):
        """按限价规则重走一遍。wait=限价单挂单天数(1=只在 T+1 挂)。

        返回 (每笔收益列表, 每笔仓位, 成交形态列表)。成交日 > T+1 时, 卖出价改用
        "成交日次日开盘"(引擎原规则只覆盖 T+1 买入 -> T+2 卖出, 这里是对照的上界口径)。
        """
        out = []
        for x in recs:
            lim = x["sc"] * (1.0 - pb / 100.0)
            px = None
            how = "miss"
            for w in range(wait):
                i = didx.get(x["bd"])
                if i is None or i + w >= len(dates):
                    break
                dd = dates[i + w]
                sub = per_code.get(x["code"])
                if sub is None or dd not in sub.index:
                    continue
                r = sub.loc[dd]
                f = float(r["close_adj"]) / float(r["close"]) if float(r["close"]) else 1.0
                o, lo = float(r["open_adj"]), float(r["low"]) * f
                if o <= lim:
                    px, how = o, ("open" if w == 0 else f"open+{w}")
                    i_hit = i + w
                    break
                if lo <= lim:
                    px, how = lim, ("limit" if w == 0 else f"limit+{w}")
                    i_hit = i + w
                    break
            else:
                continue
            if px is None:
                continue
            if how in ("open", "limit"):
                sell = x["sell"]                      # T+1 成交: 沿用引擎实际卖出价
            else:
                if i_hit + 1 >= len(dates):
                    continue
                r2 = per_code[x["code"]]
                sd2 = dates[i_hit + 1]
                if sd2 not in r2.index:
                    continue
                sell = float(r2.loc[sd2]["open_adj"])
            gross = (sell / px - 1.0) * 100.0
            out.append((gross - x["cost"], x["pos"], how))
        return out

    def _engine_base():
        """pb=0 的真基线: 引擎 352 笔原值(不限价, 开盘无条件买)。"""
        nav = 1.0
        rets = []
        for x in recs:
            net = (x["sell"] / x["buy"] - 1.0) * 100.0 - x["cost"]
            rets.append(net)
            nav *= (1.0 + net / 100.0 * x["pos"] * LEV)
        return {"pb": 0.0, "wait": 1, "n": len(rets), "fill": 1.0, "mean": st.mean(rets),
                "median": st.median(rets), "win": 100.0 * sum(1 for r in rets if r > 0) / len(rets),
                "sd": st.pstdev(rets), "nav": nav, "gross_mean": st.mean(rets) + 0.35}

    def _walk_loc(pb: float):
        """限价优先 + 未触及则按当日收盘买(兜底): 笔数不减少, 只把成交价压低。
        实盘可执行: 9:25 挂限价, 14:55 未成交则以收盘价市价买入。"""
        out = []
        for x in recs:
            lim = x["sc"] * (1.0 - pb / 100.0)
            sub = per_code.get(x["code"])
            r = sub.loc[x["bd"]] if sub is not None and x["bd"] in sub.index else None
            if r is None:
                continue
            f = float(r["close_adj"]) / float(r["close"]) if float(r["close"]) else 1.0
            o, lo, c = float(r["open_adj"]), float(r["low"]) * f, float(r["close_adj"])
            if o <= lim:
                px, how = o, "open"
            elif lo <= lim:
                px, how = lim, "limit"
            else:
                px, how = c, "close"
            out.append(((x["sell"] / px - 1.0) * 100.0 - x["cost"], x["pos"], how))
        return out

    def simulate_loc(pb: float) -> dict:
        res_ = _walk_loc(pb)
        nav, rets = 1.0, []
        for net, pos_, how in res_:
            rets.append(net)
            nav *= (1.0 + net / 100.0 * pos_ * LEV)
        return {"pb": pb, "n": len(rets), "fill": 1.0, "mean": st.mean(rets),
                "median": st.median(rets), "win": 100.0 * sum(1 for r in rets if r > 0) / len(rets),
                "sd": st.pstdev(rets), "nav": nav, "gross_mean": st.mean(rets) + 0.35,
                "mix": {k: sum(1 for _, _, h in res_ if h == k) for k in ("open", "limit", "close")}}

    def simulate(pb: float, wait: int = 1) -> dict:
        if pb <= 0:
            r0 = _engine_base()
            return {**r0, "wait": wait}
        res_ = _walk(pb, wait)
        if not res_:
            return {"pb": pb, "wait": wait, "n": 0, "fill": 0.0, "mean": float("nan"),
                    "median": float("nan"), "win": 0.0, "sd": float("nan"), "nav": 1.0,
                    "gross_mean": float("nan")}
        nav, n = 1.0, 0
        rets, grosses = [], []
        for net, pos_, how in res_:
            rets.append(net)
            grosses.append(net + 0.35)
            nav *= (1.0 + net / 100.0 * pos_ * LEV)
            n += 1
        return {"pb": pb, "wait": wait, "n": n, "fill": n / max(1, len(recs)),
                "mean": st.mean(rets), "median": st.median(rets),
                "win": 100.0 * sum(1 for r in rets if r > 0) / len(rets),
                "sd": st.pstdev(rets), "nav": nav, "gross_mean": st.mean(grosses)}

    rows = [simulate(p, 1) for p in PB]
    rows3 = [simulate(p, 3) for p in (0.0, 0.5, 1.0, 1.5, 2.0)]
    base = rows[0]
    print("\n[挂 1 天: 只在 T+1 挂, 未触及当日空仓]")
    print("  pb% |   笔数 | 成交率 | 单笔均值 |  中位 |  胜率 |    sd | 终值(L=1.25) | 相对base")
    for r in rows:
        print(f"  {r['pb']:4.1f} | {r['n']:6d} | {r['fill']*100:5.1f}% | {r['mean']:+7.3f}% | "
              f"{r['median']:+6.2f}% | {r['win']:5.1f}% | {r['sd']:5.2f}% | "
              f"{r['nav']:11.1f}x | {r['nav']/base['nav']:6.2f}x", flush=True)

    print("\n[挂 3 天: 3 个交易日内触及即买, 卖出自成交日次日开盘]")
    print("  pb% |   笔数 | 成交率 | 单笔均值 |  中位 |  胜率 |    sd | 终值(L=1.25)")
    for r in rows3:
        print(f"  {r['pb']:4.1f} | {r['n']:6d} | {r['fill']*100:5.1f}% | {r['mean']:+7.3f}% | "
              f"{r['median']:+6.2f}% | {r['win']:5.1f}% | {r['sd']:5.2f}% | {r['nav']:11.1f}x", flush=True)

    print("\n[限价优先 + 未成交按收盘买兜底: 笔数不减]")
    print("  pb% |   笔数 | 单笔均值 |  中位 |  胜率 | 终值(L=1.25) | 相对base | open/limit/close")
    rows_loc = [simulate_loc(p) for p in PB if p > 0]
    for r in rows_loc:
        print(f"  {r['pb']:4.1f} | {r['n']:6d} | {r['mean']:+7.3f}% | {r['median']:+6.2f}% | "
              f"{r['win']:5.1f}% | {r['nav']:11.1f}x | {r['nav']/base['nav']:6.2f}x | "
              f"{r['mix']['open']}/{r['mix']['limit']}/{r['mix']['close']}", flush=True)

    # 逐笔明细落盘(诊断用: 找出贡献最大的笔, 确认不是复权/口径错误)
    dump = []
    for x in recs:
        lim = x["sc"] * (1.0 - 1.0 / 100.0)
        if x["o"] <= lim:
            px, how = x["o"], "open"
        elif x["lo"] <= lim:
            px, how = lim, "limit"
        else:
            px, how = None, "miss"
        dump.append({**{k: round(v, 4) if isinstance(v, float) else v for k, v in x.items()},
                     "lim": round(lim, 4), "px": round(px, 4) if px else None, "how": how,
                     "ret_old": round((x["sell"] / x["buy"] - 1) * 100 - x["cost"], 3),
                     "ret_new": round((x["sell"] / px - 1) * 100 - x["cost"], 3) if px else None})

    # 分年(只对两个候选做, 避免刷屏)
    by_year = {}
    for pb in (0.0, 1.0, 1.5):
        d = {}
        for x in recs:
            lim = x["sc"] * (1.0 - pb / 100.0)
            px = x["buy"] if pb <= 0 else (x["o"] if x["o"] <= lim else (lim if x["lo"] <= lim else None))
            if px is None:
                continue
            y = x["bd"][:4]
            d.setdefault(y, []).append((x["sell"] / px - 1.0) * 100.0 - x["cost"])
        by_year[pb] = {y: (len(v), st.mean(v)) for y, v in sorted(d.items())}

    out = {"days": days, "base_metrics": res["metrics"], "rows": rows,
           "rows_wait3": rows3, "rows_loc": rows_loc, "dump": dump,
           "by_year": {str(k): v for k, v in by_year.items()},
           "n_trades": len(recs), "lev": LEV}
    p = ROOT / "output" / f"exp_bid_194_{days}.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n写入 {p}")

    print("\n分年单笔均值(pb=0 / 1.0 / 1.5):")
    ys = sorted(set(by_year[0.0]) | set(by_year[1.0]))
    for y in ys:
        a = by_year[0.0].get(y, (0, float("nan")))
        b = by_year[1.0].get(y, (0, float("nan")))
        c = by_year[1.5].get(y, (0, float("nan")))
        print(f"  {y}: base {a[1]:+6.2f}%({a[0]:3d}) | pb1.0 {b[1]:+6.2f}%({b[0]:3d}) | pb1.5 {c[1]:+6.2f}%({c[0]:3d})")


if __name__ == "__main__":
    main()
