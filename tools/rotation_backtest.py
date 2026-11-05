# -*- coding: utf-8 -*-
"""副策略 v4(低波组合轮动)回测 —— 输出结构与 backtest.json / satellite_bt.json 同构。

页面用同一套卡片/曲线/月度渲染器, 只有数字来源不同。
口径与 src/rotation.py 逐条对齐(同一池子、同一排序、同一成本), 不另写一套选股。

用法: python tools/rotation_backtest.py [--days 1500] [--out output/rotation_bt.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context  # noqa: E402
from src.config import load_config  # noqa: E402
from src.rotation import STRATEGY_CN, params, pick_basket  # noqa: E402

HALF = 0.0015


def _row(day, code):
    r = day.loc[day["ts_code"] == code]
    return r.iloc[0] if len(r) else None


def simulate(ctx, cfg, days=None):
    by_date, dates, reg = ctx["by_date"], ctx["main_dates"], ctx["reg"]
    if reg is not None and len(reg) and not isinstance(reg.index[0], str):
        reg = reg.copy(); reg.index = reg.index.astype(str)
    P = params(cfg)
    n, hold = P["n_hold"], P["hold_days"]
    names = ctx.get("names") or {}
    init = 100000.0

    cash, holds, last_px = init, {}, {}
    entry = {}          # code -> (买入日, 复权买入价), 期末未平仓表要写"买入日/买入价/浮盈"
    pending = None
    navs, daily, rounds, open_rows = [], [], [], []
    round_start_nav = init
    cur_target = []

    for j, d in enumerate(dates):
        day = by_date.get(d)
        if day is None:
            navs.append((d, cash + sum(s * last_px.get(c, 0) for c, s in holds.items())))
            daily.append({"date": d, "holding_n": len(holds), "nav": None, "note": "面板缺该日行情"})
            continue

        # T+1 开盘执行调仓
        if pending and pending[1] == j:
            target = pending[0]
            prev_codes = set(holds)      # 调仓前的持仓, 用来算"留"几只(买入后再算恒等于 N)
            px = {}
            for c in set(list(holds) + list(target)):
                r = _row(day, c)
                if r is not None and bool(r["is_paused"] == 0) and float(r["vol"]) > 0:
                    px[c] = (float(r["open_adj"]), float(r["open"]), float(r["high_limit"]))
            sold = bought = 0.0
            sold_list, buy_list = [], []
            for c in list(holds):
                if c not in target and c in px:
                    amt = holds[c] * px[c][0]
                    cash += amt * (1 - HALF)
                    sold += amt
                    sold_list.append({"code": c, "name": names.get(c, ""),
                                      "px": round(px[c][1], 3)})
                    del holds[c]
            equity = cash + sum(s * px.get(c, (last_px.get(c, 0),))[0] for c, s in holds.items())
            need = [c for c in target if c not in holds and c in px
                    and px[c][1] < px[c][2] - 1e-9]
            if need:
                each = equity / max(len(target), 1)
                for c in need:
                    budget = min(each, cash)
                    if budget <= 0:
                        break
                    sh = budget / (px[c][0] * (1 + HALF))
                    cash -= budget
                    holds[c] = sh
                    entry[c] = (d, px[c][0])
                    bought += budget
                    buy_list.append({"code": c, "name": names.get(c, ""), "px": round(px[c][1], 3)})
            turn = (sold + bought) / equity if equity else 0.0
            mv_now = sum(s * px.get(c, (last_px.get(c, 0),))[0] for c, s in holds.items())
            rnd_ret = (cash + mv_now) / round_start_nav - 1
            rounds.append({"date": d, "sold": sold_list, "bought": buy_list,
                           "keep": len([c for c in target if c in prev_codes]),
                           "turn_pct": round(turn * 100, 1),
                           "ret_pct": round(rnd_ret * 100, 2),
                           "hold_n": len(holds)})
            round_start_nav = cash + mv_now
            cur_target = target
            pending = None

        # 收盘估值
        mv = 0.0
        for c, s in holds.items():
            r = _row(day, c)
            if r is not None and bool(r["is_paused"] == 0) and float(r["vol"]) > 0:
                last_px[c] = float(r["close_adj"])
            mv += s * last_px.get(c, 0)
        nav_now = cash + mv
        navs.append((d, nav_now))
        daily.append({"date": d, "holding_n": len(holds), "nav": round(nav_now / init, 4),
                      "note": ""})

        # 收盘生成下一次调仓清单
        if j % hold == 0 and j + 1 < len(dates):
            basket = pick_basket(day, cfg, names)
            if basket:
                pending = ([b["code"] for b in basket], j + 1, basket)
            elif holds:
                pending = ([], j + 1, [])

    nav = pd.Series({d: v for d, v in navs}, dtype=float)
    ret = nav.pct_change().dropna()
    years = len(nav) / 250.0
    total = nav.iloc[-1] / init - 1
    cagr = (1 + total) ** (1 / years) - 1 if years > 0 and (1 + total) > 0 else -1.0
    mdd = float((nav / nav.cummax() - 1).min())
    sharpe = float(ret.mean() / ret.std() * np.sqrt(252)) if ret.std() else 0.0
    rs = [r["ret_pct"] for r in rounds]
    wins = [x for x in rs if x > 0]

    # 月度
    monthly = []
    s = nav.copy()
    s.index = pd.to_datetime(s.index, format="%Y%m%d")
    mo = s.resample("ME").last()
    for i in range(1, len(mo)):
        monthly.append({"month": mo.index[i].strftime("%Y-%m"),
                        "ret_pct": round((mo.iloc[i] / mo.iloc[i - 1] - 1) * 100, 2)})

    bench = reg.loc[reg.index.isin(dates), "mkt_idx"] if reg is not None else pd.Series(dtype=float)
    bench_norm = bench / bench.iloc[0] * init if len(bench) else bench

    # 期末持仓
    if holds and dates:
        day = by_date.get(dates[-1])
        for c, sh in holds.items():
            r = _row(day, c) if day is not None else None
            lp = float(r["close_adj"]) if r is not None else last_px.get(c, 0)
            # 字段名对齐页面共用渲染器(btBlock): buy_date/buy_px/cost_pct/unrealized_pct,
            # 否则三列全是 "-", 看着像数据缺失。weight_pct/last_px 保留给别的读者。
            ed, epx = entry.get(c, (None, None))
            open_rows.append({"code": c, "name": names.get(c, ""), "shares": round(sh, 0),
                              "last_px": round(lp, 3),
                              "weight_pct": round(sh * lp / nav.iloc[-1] * 100, 2),
                              "buy_date": ed, "buy_px": round(epx, 3) if epx else None,
                              "cost_pct": round(sh * (epx or lp) / nav.iloc[-1] * 100, 2),
                              "unrealized_pct": round((lp / epx - 1) * 100, 2) if epx else None})

    return {
        "meta": {"run_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                 "start": dates[0], "end": dates[-1], "days": len(dates),
                 "initial_capital": init, "strategy": STRATEGY_CN,
                 "rules": "低波 %d 只等权 · 每 %d 个交易日调仓 · T+1 开盘 · 单边成本 0.15%%"
                          % (n, hold)},
        "metrics": {"总收益率%": round(total * 100, 2), "年化收益率%": round(cagr * 100, 2),
                    "最大回撤%": round(mdd * 100, 2),
                    "夏普比率": round(sharpe, 2),
                    "卡玛比率": round(cagr / abs(mdd), 2) if mdd else 0.0,
                    "胜率%": round(len(wins) / len(rs) * 100, 1) if rs else 0.0,
                    "交易次数": len(rs),
                    "平均单笔收益%": round(float(np.mean(rs)), 2) if rs else 0.0,
                    "调仓换手%": round(float(np.mean([r["turn_pct"] for r in rounds])), 1) if rounds else 0.0},
        "equity": [{"d": d, "nav": round(v, 2)} for d, v in navs],
        "benchmark": [{"d": d, "nav": round(float(v), 2)} for d, v in bench_norm.items()] if len(bench) else [],
        "monthly": monthly[-24:],
        "rounds": rounds,
        "open_positions": open_rows,
        "replay": {
            "meta": {"start": dates[0], "end": dates[-1], "days": len(dates), "init_capital": init},
            "summary": {"total_pct": round(total * 100, 2), "mdd_pct": round(mdd * 100, 2),
                        "trades": len(rs), "win_rate": round(len(wins) / len(rs) * 100, 1) if rs else 0.0,
                        "span": "%s ~ %s" % (dates[0], dates[-1])},
            "curve": [{"d": d, "nav": round(v / init, 4)} for d, v in navs],
            "monthly": monthly[-13:],
            # 回放页按"轮次"展示: 一次调仓一行(买N/卖N/留N/换手), 不是单票逐笔
            "trades": [{"signal_date": r["date"], "entry_date": r["date"], "exit_date": r["date"],
                        "code": "", "name": "%d买/%d卖" % (len(r["bought"]), len(r["sold"])),
                        "buy_n": len(r["bought"]), "sell_n": len(r["sold"]),
                        "keep_n": r["keep"], "turn_pct": r["turn_pct"],
                        "entry_px": None, "exit_px": None, "ret_pct": r["ret_pct"],
                        "note": "换手 %.1f%% · 期末持仓 %d 只" % (r["turn_pct"], r["hold_n"])}
                       for r in rounds],
            "daily": daily[-260:],
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=1500)
    ap.add_argument("--out", default="output/rotation_bt.json")
    a = ap.parse_args()
    cfg = load_config()
    print(f"[ctx] {a.days} 日...", flush=True)
    ctx = prepare_context(cfg, a.days)
    print(f"[ctx] {len(ctx['main_dates'])} 日", flush=True)
    out = simulate(ctx, cfg, a.days)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    m = out["metrics"]
    print("总 %.2f%% 年化 %.2f%% 回撤 %.2f%% 卡玛 %.2f 夏普 %.2f 调仓 %d 次 胜率 %.1f%% 换手 %.1f%%"
          % (m["总收益率%"], m["年化收益率%"], m["最大回撤%"], m["卡玛比率"], m["夏普比率"],
             m["交易次数"], m["胜率%"], m["调仓换手%"]))
    print("写出", a.out)


if __name__ == "__main__":
    main()
