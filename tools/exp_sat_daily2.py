# -*- coding: utf-8 -*-
"""副策略"每日一只"的三种落地方式 vs 月度基线(2026-11-05, 第一百三十八轮之二)。

第一版(exp_satellite_daily.py)已证明:
  · 每天换 1 只(H=1): 年化 -50.8%, 回撤 -98.3% —— 成本 0.3%/日 = 75%/年, 必死;
  · 每天新增 1 只、持 20 日: -1.5%; 持 120 日: +4.7%。
但那一版允许**同一只票连续多日重复建仓**, 低波票持续性极强, 结果实际是单票满仓。
真实落地必须去重。本脚本比较四种不重复的做法:

  1. monthly_buf  月度 20 只 + 3 倍缓冲(第一百三十七轮基线, 同窗口对照)
  2. dedupe_H     每天在 ivol 前 50 名里找**第一个未持有**的票, 建 1/H 仓, 持 H 日
                  (= "每日推一只不同的票", 账户稳态 H 只)
  3. replace_N    持仓上限 N=20 等权; 每天 top1 若不在持仓且比持仓最差者更优(严格更优)则替换
  4. event_N      同 3, 但要求新票 ivol 比最差持仓低 10% 才换(事件驱动, 换手更低)

口径: 金额制(现金+持仓市值), T 日收盘建仓/结算, 单次往返 0.30%, 股票池同主策略。
用法:
  python3 tools/exp_sat_daily2.py --days 300     # 冒烟
  python3 tools/exp_sat_daily2.py --days 1500
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context  # noqa: E402
from src.config import board_of, load_config  # noqa: E402

COST = 0.0030
AMT_MIN = 1e8
MIN_POOL = 200
MIN_BARS = 250
HOLDS = [10, 20, 30, 40, 60, 120]
NS = [20]
MARGINS = [0.0, 0.10, 0.25]


def _stats(rets):
    r = np.asarray(rets, dtype=float)
    if len(r) == 0:
        return {}
    nav = np.cumprod(1 + r)
    years = len(r) / 250.0
    ann = float(nav[-1] ** (1 / years) - 1) if nav[-1] > 0 else -1.0
    peak = np.maximum.accumulate(nav)
    mdd = float((nav / peak - 1).min())
    sharpe = float(np.mean(r) / (np.std(r) + 1e-12) * np.sqrt(250))
    return {"ann": round(ann * 100, 2), "mdd": round(mdd * 100, 2),
            "calmar": round(ann / abs(mdd), 2) if mdd < 0 else None,
            "sharpe": round(sharpe, 2), "days": len(r), "nav": round(float(nav[-1]), 4),
            "win": round(100 * float((r > 0).mean()), 1)}


def _universe(day, cfg):
    boards = set(cfg["universe"]["boards"])
    m = ((day["is_paused"] == 0) & (day["is_st"] == 0) & (day["bars"] >= MIN_BARS)
         & (day["amount"] >= AMT_MIN))
    df = day.loc[m]
    return df.loc[df["ts_code"].map(lambda c: board_of(c) in boards)]


class Book:
    """金额制账本: 现金 + 持仓市值, 逐日按收盘价重估。"""

    def __init__(self):
        self.cash = 1.0
        self.hold = {}          # code -> {"v": 市值, "exp": 到期 j}
        self.turn = 0.0
        self.ntrade = 0

    def value(self):
        return self.cash + sum(h["v"] for h in self.hold.values())

    def mark(self, j, ret_m, cpos):
        """按第 j 日收益(前收->今收)重估持仓。"""
        for c, h in list(self.hold.items()):
            i = cpos.get(c)
            if i is None:
                continue
            r = ret_m[i, j - 1]
            if np.isfinite(r):
                h["v"] *= (1.0 + float(r))

    def expire(self, j):
        for c in [c for c, h in self.hold.items() if h["exp"] <= j]:
            v = self.hold.pop(c)["v"]
            self.cash += v * (1 - COST / 2)
            self.turn += v

    def buy(self, c, amount, exp):
        amount = min(amount, self.cash)
        if amount <= 1e-9:
            return False
        self.cash -= amount
        self.hold[c] = {"v": amount * (1 - COST / 2), "exp": exp}
        self.turn += amount
        self.ntrade += 1
        return True

    def sell(self, c):
        if c not in self.hold:
            return 0.0
        v = self.hold.pop(c)["v"]
        self.cash += v * (1 - COST / 2)
        self.turn += v
        return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=1500)
    ap.add_argument("--out", default="output/exp_sat_daily2.json")
    args = ap.parse_args()

    cfg = load_config()
    print("[1/4] 构建上下文 %d 日 ..." % args.days, flush=True)
    t0 = time.time()
    ctx = prepare_context(cfg, args.days)
    by_date, main_dates = ctx["by_date"], ctx["main_dates"]
    print("      ctx ready %.0fs, %d 日" % (time.time() - t0, len(main_dates)), flush=True)

    dates = sorted(by_date)
    dpos = {d: j for j, d in enumerate(dates)}
    codes = sorted({c for d in dates for c in by_date[d]["ts_code"]})
    cpos = {c: i for i, c in enumerate(codes)}

    close_m = np.full((len(codes), len(dates)), np.nan, dtype=np.float32)
    for j, d in enumerate(dates):
        df = by_date[d]
        close_m[df["ts_code"].map(cpos).to_numpy(), j] = df["close_adj"].to_numpy()
    ret_m = close_m[:, 1:] / close_m[:, :-1] - 1.0
    ret_m = np.where(np.isfinite(ret_m), ret_m, np.nan).astype(np.float32)

    # ---- 逐日排名(ivol 升序), 只保留前 50 ----
    print("[2/4] 逐日 ivol 排名 ...", flush=True)
    t1 = time.time()
    rank = {}                  # date -> [code...] 前 50
    ivol_of = {}               # (date, code) -> ivol
    for k, d in enumerate(main_dates):
        j = dpos[d]
        if j < 1 or j >= len(dates) - 1:
            continue
        day = _universe(by_date[d], cfg)
        if len(day) < MIN_POOL:
            continue
        v = pd.to_numeric(day["ivol20"], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(v) & (v > 0)
        if ok.sum() < 60:
            continue
        cc = np.asarray(day["ts_code"], dtype=object)[ok]
        vv = v[ok]
        order = np.argsort(vv, kind="mergesort")[:50]
        lst = [str(cc[i]) for i in order]
        rank[d] = lst
        for c, x in zip(lst, vv[order]):
            ivol_of[(d, c)] = float(x)
        if (k + 1) % 300 == 0:
            print("      %d 日, %.0fs" % (k + 1, time.time() - t1), flush=True)
    sig = [d for d in main_dates if d in rank]
    print("      有效 %d 日" % len(sig), flush=True)

    res = {"days": args.days, "cost_pct": COST * 100, "holds": HOLDS, "ns": NS,
           "margins": MARGINS, "strategies": {}, "series": {}}
    bench = []

    def run(name, day_hook):
        bk = Book()
        rets = []
        prev_nav = 1.0
        for d in sig:
            j = dpos[d]
            bk.mark(j, ret_m, cpos)
            bk.expire(j)
            day_hook(bk, d, j)
            nav = bk.value()
            rets.append(nav / prev_nav - 1.0)
            prev_nav = nav
        st = _stats(rets)
        st["turnover_yr"] = round(bk.turn / 2.0 / max(1e-9, len(sig) / 250.0), 2)
        st["trades"] = bk.ntrade
        res["strategies"][name] = st
        res["series"][name] = [round(x * 100, 4) for x in rets]
        return st

    # ---- 基准 ----
    for d in sig:
        j = dpos[d]
        day = _universe(by_date[d], cfg)
        ci = day["ts_code"].map(cpos).to_numpy()
        r = ret_m[ci, j - 1]
        r = r[np.isfinite(r)]
        bench.append(float(np.mean(r)) if len(r) else 0.0)
    res["benchmark"] = _stats(bench)

    # ---- 1) 月度 20 只 + 3 倍缓冲 ----
    month_sig = {}
    for i in range(len(sig) - 1):
        if sig[i][:6] != sig[i + 1][:6]:
            month_sig[sig[i + 1]] = sig[i]      # 次月首个交易日执行, 信号=上月最后一日
    state = {"prev": []}

    def monthly(bk, d, j):
        if d not in month_sig:
            return
        sd = month_sig[d]
        lst = rank.get(sd) or []
        if len(lst) < 60:
            return
        top20 = lst[:20]
        if state["prev"]:
            keep = [c for c in state["prev"] if c in set(lst[:60])]
            add = [c for c in top20 if c not in set(keep)]
            codes_n = (keep + add)[:20]
        else:
            codes_n = top20
        # 全部卖出再等权买入(月度基线按换手计成本)
        for c in list(bk.hold):
            bk.sell(c)
        target = bk.value() / max(1, len(codes_n))
        for c in codes_n:
            bk.buy(c, target, j + 10 ** 6)
        state["prev"] = codes_n

    print("[3/4] 回放 ...", flush=True)
    st = run("monthly_buf", monthly)
    print("      monthly_buf %+.1f%%/年, 换手 %.1f 倍/年" % (st["ann"], st["turnover_yr"]), flush=True)

    # ---- 2) dedupe_H: 每天找第一个未持有的票, 建 1/H 仓 ----
    for H in HOLDS:
        def dedupe(bk, d, j, H=H):
            lst = rank.get(d) or []
            for c in lst:
                if c in bk.hold:
                    continue
                amt = bk.value() / H
                if bk.buy(c, amt, j + H):
                    break
        st = run("dedupe_H%d" % H, dedupe)
        print("      dedupe_H%-3d %+.1f%%/年 回撤 %+.1f%% 换手 %.1f" % (
            H, st["ann"], st["mdd"], st["turnover_yr"]), flush=True)

    # ---- 3) replace_N: 持仓 N=20, top1 严格更优则替换最差 ----
    for N in NS:
        for mg in MARGINS:
            def replace(bk, d, j, N=N, mg=mg):
                lst = rank.get(d) or []
                if not lst:
                    return
                if len(bk.hold) < N:
                    # 建仓期: 每天买一只未持有的最优票, 直到满 N 只
                    for c in lst:
                        if c not in bk.hold:
                            bk.buy(c, bk.value() / N, j + 10 ** 6)
                            break
                    return
                cand = lst[0]
                if cand in bk.hold:
                    return
                # 最差持仓: 当日 ivol 最大者
                worst, wv = None, -1.0
                for c in bk.hold:
                    v = ivol_of.get((d, c))
                    if v is None:
                        # 不在前 50 的持仓视为最差
                        worst, wv = c, 9e9
                        break
                    if v > wv:
                        worst, wv = c, v
                cv = ivol_of.get((d, cand), 9e9)
                if worst and cv < wv * (1.0 - mg):
                    got = bk.sell(worst)
                    bk.buy(cand, got, j + 10 ** 6)
            st = run("replace_N%d_m%02d" % (N, int(mg * 100)), replace)
            print("      replace_N%d_m%-2d %+.1f%%/年 回撤 %+.1f%% 换手 %.1f" % (
                N, int(mg * 100), st["ann"], st["mdd"], st["turnover_yr"]), flush=True)

    # ---- 4) event_N 已在 replace 的 margin 里覆盖 ----

    print("\n=== 汇总 (%d 日) ===" % len(sig), flush=True)
    print("  %-18s %8s %8s %8s %8s %8s" % ("组合", "年化", "回撤", "卡玛", "换手/年", "交易数"), flush=True)
    b = res["benchmark"]
    print("  %-18s %+7.1f%% %+7.1f%% %8.2f %8s %8s" % (
        "[基准]", b["ann"], b["mdd"], b["calmar"] or 0, "-", "-"), flush=True)
    for k, v in res["strategies"].items():
        print("  %-18s %+7.1f%% %+7.1f%% %8.2f %8.1f %8d" % (
            k, v["ann"], v["mdd"], v["calmar"] or 0, v["turnover_yr"], v["trades"]), flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print("\n结果已写 %s" % out, flush=True)


if __name__ == "__main__":
    main()
