# -*- coding: utf-8 -*-
"""副策略换赛道: 低波组合轮动(第一百四十五轮)。

为什么换结构而不是换因子
------------------------
因子扫描(exp_factor_scan)结论: 7 个单因子 + 3 个合成因子里, 低波(ivol20)仍是最强的
(IC -0.131, 五组完全单调, Q1-Q5 多空 2.26%/20日); 低波+小额只把 Q5 从 +1.23% 拉到
+1.23%(单低波 Q1 +1.10%), 提升 0.13pp —— 没有"更好的赛道"可换。

但扫描同时暴露了真问题: 那个 +1.10% 是**一组约 100 只等权**的均值, 而副策略是
单吊 1 只。低波因子的收益来自"避开高波组"(Q5 -1.16%), 是排除法, 不是精选法;
从 100 只里挑 1 只, 等于把 15% 的年化期望丢进一个方差极大的抽样。

所以这一轮换的是**结构**: 单吊 → 小组合等权轮动, 因子仍是低波(可叠加小额)。

口径
----
· T 日收盘按因子排序取前 N 只 → T+1 开盘调仓(卖不在榜的、买新进榜的, 等权)
· 每 hold 个交易日调一次; 持有期内不动
· 单边成本 0.15%, 按实际换手扣(不是每次都全换)
· 停牌/一字涨停买不到就跳过, 资金留存; 卖不出(停牌)继续持有
· 无未来函数: 信号只用 T 日收盘列, 成交用 T+1 开盘

用法: python tools/exp_rot_lowvol.py [--days 1500] [--json output/exp_rot_lowvol.json]
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

from src.backtest import prepare_context  # noqa: E402
from src.config import board_of, load_config  # noqa: E402
from src.satellite import MIN_BARS, daily_params  # noqa: E402

HALF = 0.0015


def _pool(day, cfg):
    P = daily_params(cfg)
    uni = cfg.get("universe") or {}
    boards = set(uni.get("boards") or ["main", "gem"])
    m = ((day["is_paused"] == 0) & (day["is_st"] == 0) & (day["bars"] >= MIN_BARS)
         & (day["amount"] >= P["amt_min"]) & (day["ivol20"] > 0))
    m &= day["ts_code"].map(lambda c: board_of(c) in boards).to_numpy()
    return day.loc[m]


def _target(day, cfg, n, score):
    """当日目标组合(等权, 取 score 最高的 n 只)。score: 'ivol' | 'ivol_amt'。"""
    df = _pool(day, cfg)
    if len(df) < n:
        return []
    if score == "ivol":
        key = df["ivol20"].astype(float)
    else:
        z = lambda s: (s - s.mean()) / s.std() if s.std() else s * 0  # noqa: E731
        key = -(z(df["ivol20"].astype(float)) + z(df["amount"].astype(float)))
    return list(df.loc[key.nsmallest(n).index, "ts_code"])


def simulate(ctx, cfg, *, n=20, hold=20, score="ivol", gate=False):
    by_date, dates, reg = ctx["by_date"], ctx["main_dates"], ctx["reg"]
    if reg is not None and len(reg) and not isinstance(reg.index[0], str):
        reg = reg.copy(); reg.index = reg.index.astype(str)

    cash = 100000.0
    holds = {}                 # code -> shares
    last_px = {}
    pending = None             # (target_list, day_index)
    navs, turns, npos = [], [], []

    for j, d in enumerate(dates):
        day = by_date.get(d)
        if day is None:
            navs.append((d, cash + sum(s * last_px.get(c, 0) for c, s in holds.items())))
            continue

        # ---- T+1 开盘: 执行上一交易日生成的调仓单 ----
        if pending and pending[1] == j:
            target = set(pending[0])
            px = {}
            for c in set(list(holds) + list(target)):
                r = day.loc[day["ts_code"] == c]
                if len(r):
                    r = r.iloc[0]
                    if bool(r["is_paused"] == 0) and float(r["vol"]) > 0:
                        px[c] = (float(r["open_adj"]), float(r["open"]), float(r["high_limit"]))
            sold = bought = 0.0
            # 卖出不在目标里的
            for c in list(holds):
                if c not in target and c in px:
                    cash += holds[c] * px[c][0] * (1 - HALF)
                    sold += holds[c] * px[c][0]
                    del holds[c]
            # 等权买入目标里没有的
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
                    bought += budget
            turns.append((sold + bought) / equity if equity else 0.0)
            pending = None

        # ---- 收盘估值 ----
        mv = 0.0
        for c, s in holds.items():
            r = day.loc[day["ts_code"] == c]
            if len(r):
                r = r.iloc[0]
                if bool(r["is_paused"] == 0) and float(r["vol"]) > 0:
                    last_px[c] = float(r["close_adj"])
            mv += s * last_px.get(c, 0)
        navs.append((d, cash + mv))
        npos.append(len(holds))

        # ---- 收盘生成下一次调仓单 ----
        if j % hold == 0 and j + 1 < len(dates):
            ok = True
            if gate and reg is not None and d in reg.index:
                ok = bool(reg.loc[d, "trade_ok"])
            tgt = _target(day, cfg, n, score) if ok else []
            if tgt or holds:
                pending = (tgt, j + 1)

    nav = pd.Series({d: v for d, v in navs}, dtype=float)
    ret = nav.pct_change().dropna()
    years = len(nav) / 250.0
    total = nav.iloc[-1] / 100000.0 - 1
    cagr = (1 + total) ** (1 / years) - 1 if years > 0 and (1 + total) > 0 else -1.0
    mdd = float((nav / nav.cummax() - 1).min())
    sharpe = float(ret.mean() / ret.std() * np.sqrt(252)) if ret.std() else 0.0
    yr, prev = {}, nav.iloc[0]
    for y in sorted({str(d)[:4] for d in nav.index}):
        seg = nav[[d for d in nav.index if str(d)[:4] == y]]
        yr[y] = round((seg.iloc[-1] / prev - 1) * 100, 1); prev = seg.iloc[-1]
    return {"total": round(total * 100, 2), "cagr": round(cagr * 100, 2),
            "mdd": round(mdd * 100, 2),
            "calmar": round(cagr / abs(mdd), 2) if mdd else 0.0,
            "sharpe": round(sharpe, 2),
            "turn": round(float(np.mean(turns)) * 100, 1) if turns else 0.0,
            "pos": round(float(np.mean(npos)), 1), "years": yr}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=1500)
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    cfg = load_config()
    print(f"[ctx] 载入 {a.days} 日...", flush=True)
    ctx = prepare_context(cfg, a.days)
    print(f"[ctx] {len(ctx['main_dates'])} 日 {ctx['main_dates'][0]} ~ {ctx['main_dates'][-1]}",
          flush=True)

    variants = [
        ("R1 低波 N20 H20", dict(n=20, hold=20, score="ivol")),
        ("R2 低波+小额 N20 H20", dict(n=20, hold=20, score="ivol_amt")),
        ("R3 低波+小额 N10 H20", dict(n=10, hold=20, score="ivol_amt")),
        ("R4 低波 N40 H20", dict(n=40, hold=20, score="ivol")),
        ("R5 低波 N20 H40", dict(n=20, hold=40, score="ivol")),
        ("R6 低波+小额 N20 H20 闸门", dict(n=20, hold=20, score="ivol_amt", gate=True)),
    ]
    out = []
    print("%-24s %9s %8s %9s %7s %7s %7s %6s" % ("变体", "总收益", "年化", "回撤", "卡玛", "夏普", "换手", "持仓"))
    for name, kw in variants:
        r = simulate(ctx, cfg, **kw)
        out.append({"name": name, **r})
        print("%-24s %8.2f%% %7.2f%% %8.2f%% %7.2f %7.2f %6.1f%% %6.1f"
              % (name, r["total"], r["cagr"], r["mdd"], r["calmar"], r["sharpe"], r["turn"], r["pos"]),
              flush=True)

    years = sorted({y for r in out for y in r["years"]})
    print("\n分年(%):")
    print("变体".ljust(24) + "".join(y.rjust(8) for y in years))
    for r in out:
        print(r["name"].ljust(24) + "".join(str(r["years"].get(y, "")).rjust(8) for y in years))

    if a.json:
        Path(a.json).write_text(json.dumps({"days": a.days, "result": out}, ensure_ascii=False, indent=1),
                                encoding="utf-8")
        print("\n写出", a.json)


if __name__ == "__main__":
    main()
