"""第 201 轮：闲置资金扫描（cash sweep）。

主策略满仓单吊，T+1 开盘买、T+2 收盘卖 —— 全史 3000 日里有 2283 天空仓（76%），
1500 日里也有约一半时间拿着现金吃 1.5%。

第 197 轮测过"固定权重 60/40""动态权重"，都是**从主策略嘴里抢钱**；
本轮测的是完全不同的东西：**主策略优先占用 100% 资金，主策略空仓时才把现金交给副策略**。
副策略也空仓就吃货币收益。这不是配置权重，是填坑。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CASH = 0.015 / 252


def nav(f):
    d = json.load(open(ROOT / "output" / f))
    s = pd.Series({x["d"].replace("-", ""): x["nav"] for x in d["equity"]}).sort_index()
    return s


def masks(trades, entry, exit_):
    out = {}
    for t in trades:
        a, b = str(t.get(entry) or ""), str(t.get(exit_) or "")
        if not a or not b or a == "None" or b == "None":
            continue
        out.setdefault("span", []).append((a.replace("-", ""), b.replace("-", "")))
    return out["span"]


def span_mask(idx, spans):
    m = pd.Series(False, index=idx)
    pos = {d: i for i, d in enumerate(idx)}
    for a, b in spans:
        i, j = pos.get(a), pos.get(b)
        if i is None or j is None or j < i:
            continue
        m.iloc[i:j + 1] = True
    return m


def perf(r):
    nv = (1 + r).cumprod()
    n = len(r)
    cagr = nv.iloc[-1] ** (252 / n) - 1
    dd = (nv / nv.cummax() - 1).min()
    return dict(total=(nv.iloc[-1] - 1) * 100, cagr=cagr * 100, dd=dd * 100,
                calmar=cagr / abs(dd) if dd else np.nan, n=n)


def build(days):
    tag = "1500" if days == 1500 else "3000"
    em = nav("backtest.json" if days == 1500 else "backtest_3000.json")
    es = nav("satellite_bt.json" if days == 1500 else "satellite_bt_3000.json")
    idx = em.index
    es = es.reindex(idx).ffill()

    t = pd.read_csv(ROOT / "output" / f"exp_factor_197_trades_{days}.csv", dtype={"buy": str, "sell": str})
    t["buy"] = t.buy.str.replace("-", "")
    t["sell"] = t.sell.str.replace("-", "")
    m_main = span_mask(idx, list(zip(t.buy, t.sell)))

    s = json.load(open(ROOT / "output" / ("satellite_bt.json" if days == 1500 else "satellite_bt_3000.json")))
    m_sat = span_mask(idx, [(str(x["entry_date"]).replace("-", ""), str(x["exit_date"]).replace("-", ""))
                            for x in s["trades"]])

    rm = em.pct_change().fillna(0)
    rs = es.pct_change().fillna(0)
    return idx, rm, rs, m_main, m_sat, tag


def run(days, cap_sat=1.0, label=""):
    idx, rm, rs, m_main, m_sat = build(days)[:5]
    tag = build(days)[5]
    # 主策略优先；主策略空仓时副策略拿 cap_sat 比例的现金，其余吃货币
    w_sat = (~m_main) & m_sat
    r = rm.copy()
    r[w_sat] = cap_sat * rs[w_sat] + (1 - cap_sat) * CASH
    r[(~m_main) & (~m_sat)] = CASH
    base = perf(rm)
    res = perf(r)
    occ = dict(main=m_main.mean() * 100, sat=m_sat.mean() * 100,
               overlap=(m_main & m_sat).mean() * 100, sweep=w_sat.mean() * 100,
               idle=((~m_main) & (~m_sat)).mean() * 100)
    print(f"\n=== {tag} 日 {label} ===")
    print(f"  占用: 主 {occ['main']:.1f}%  副 {occ['sat']:.1f}%  重叠 {occ['overlap']:.1f}%  "
          f"可填坑 {occ['sweep']:.1f}%  全空 {occ['idle']:.1f}%")
    print(f"  基线主策略: 总 {base['total']:.0f}%  年化 {base['cagr']:.2f}%  回撤 {base['dd']:.2f}%  卡玛 {base['calmar']:.2f}")
    print(f"  填坑后  : 总 {res['total']:.0f}%  年化 {res['cagr']:.2f}%  回撤 {res['dd']:.2f}%  卡玛 {res['calmar']:.2f}"
          f"   Δ年化 {res['cagr']-base['cagr']:+.2f}pp  Δ回撤 {res['dd']-base['dd']:+.2f}pp")
    return dict(tag=tag, cap=cap_sat, base=base, res=res, occ=occ)


if __name__ == "__main__":
    out = []
    for days in (1500, 3000):
        for cap in (1.0, 0.5):
            out.append(run(days, cap, f"cap_sat={cap:.0%}"))
    json.dump(out, open(ROOT / "output" / "exp_cashsweep_201.json", "w"), ensure_ascii=False, indent=1)
