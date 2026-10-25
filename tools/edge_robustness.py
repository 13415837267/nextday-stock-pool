# -*- coding: utf-8 -*-
"""第四十三轮: 2024-09后 edge 稳健性三检验(事前注册, 零参数调整)。

背景: 十年回测(2500日, 484笔)证明全周期+140.85%但2016~2023八年连亏,
当前实盘依赖的edge完全是2024-09后市场微观结构的产物。本轮回答三个问题:
  1) 收益集中度——总收益是否靠极少数笔? 剔除top N后还剩多少?
  2) 滑点敏感性——净收益由 C*g-1 重构(g=毛收益比), 对滑点千1→千5/1%逐级重算;
  3) 容量——单吊全仓的组合规模 vs 执行日该票当日成交额(参与率上限口径)。

口径(与回测引擎一致):
  r = C * (sell_adj/buy_adj) - 1, C = (1-slip)(1-stamp)(1-comm)/((1+slip)(1+comm))
  comm=万2.5, stamp=万5, slip=千1(基准) → C≈0.997005, 往返成本约0.30%/笔。
  由 ret_pct 反解 g=(1+ret/100)/C_base, 再按新滑点 C' 重算——精确, 非近似。
  容量: 权益E_k=100万×∏(1+r_j)(单吊100%复利), 可承接规模=E≤参与率×当日amount。
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EX = json.loads((ROOT / "output" / "backtest_long.json").read_text()) if len(sys.argv) < 2 else \
    json.loads(Path(sys.argv[1]).read_text())
COMM, STAMP = 0.00025, 0.0005
SLIP_BASE = 0.001
ERA = "20240901"   # 与finance_pit_test/timing_signal_test同锚点

C_BASE = (1 - SLIP_BASE) * (1 - STAMP) * (1 - COMM) / ((1 + SLIP_BASE) * (1 + COMM))


def c_of(slip: float) -> float:
    return (1 - slip) * (1 - STAMP) * (1 - COMM) / ((1 + slip) * (1 + COMM))


def main() -> None:
    trades = [t for t in EX["trades_all"] if t.get("ret_pct") is not None]
    ret = np.array([t["ret_pct"] for t in trades])
    dates = [t["buy_date"] for t in trades]
    n = len(trades)
    print(f"== edge稳健性检验(第四十三轮, 事前注册): {n}笔已执行 "
          f"({min(dates)}~{max(dates)}) ==")
    era_m = np.array([d >= ERA for d in dates])
    print(f"分era: 前({ERA}前) {int((~era_m).sum())}笔 / 后 {int(era_m.sum())}笔\n")

    # ---------- 1) 收益集中度 ----------
    print("-- 检验1: 收益集中度(log贡献口径, 复利下每笔贡献=ln(1+r)) --")
    lg = np.log1p(ret / 100)
    tot = lg.sum()
    print(f"全周期复利 {100*(np.exp(tot)-1):.1f}% (ln合计{tot:.3f})")
    order = np.argsort(lg)[::-1]
    cum = np.cumsum(lg[order])
    for k in (1, 3, 5, 10, 20):
        if k <= n:
            share = cum[k - 1] / tot * 100 if tot > 0 else float("nan")
            rest = 100 * (np.exp(tot - cum[k - 1]) - 1)
            print(f"  剔除top{k:2d}笔(占ln贡献{share:5.1f}%)后总收益 {rest:8.1f}%   "
                  f"top{k}合计+{100*(np.exp(cum[k-1])-1):.0f}%")
    print("  top10笔明细:")
    for i in order[:10]:
        t = trades[i]
        print(f"    {t['signal_date']} {t['code']} {t['name']:<8} "
              f"{t['ret_pct']:+7.2f}%  ln贡献{lg[i]:+.3f}")
    bot = np.argsort(lg)[:5]
    print("  worst5笔:")
    for i in bot:
        t = trades[i]
        print(f"    {t['signal_date']} {t['code']} {t['name']:<8} {t['ret_pct']:+7.2f}%")
    # 同票重复度
    codes = pd.Series([t["code"] for t in trades]).value_counts()
    print(f"  同票重复: 独立票{codes.size}只, 最多1只{codes.iloc[0]}次"
          f"(top5票合计{int(codes.head(5).sum())}笔)")
    # era内集中度(注意: 逐笔复利口径假设每笔100%部署, 实际受pending_cash
    # 摩擦略低于此——全周期226.4% vs 实际140.85%; 相对占比仍有效)
    for lbl, m in (("era前", ~era_m), ("era后", era_m)):
        if m.sum() < 5:
            continue
        lgm, tm = lg[m], lg[m].sum()
        om = np.argsort(lgm)[::-1]
        k = max(1, int(round(m.sum() * 0.05)))
        top = lgm[om[:k]].sum()
        rest = tm - top
        print(f"  {lbl}: {int(m.sum())}笔, 合计{100*(np.exp(tm)-1):+.1f}%; "
              f"top5%笔({k}笔)合计{100*(np.exp(top)-1):+.1f}%, "
              f"其余{int(m.sum())-k}笔合计{100*(np.exp(rest)-1):+.1f}%")

    # ---------- 2) 滑点敏感性 ----------
    print("\n-- 检验2: 滑点敏感性(由g=毛收益比精确重算, 费用同口径) --")
    g = (1 + ret / 100) / C_BASE
    print(f"{'滑点':>8} {'单笔均值%':>9} {'胜率%':>6} {'总收益%':>9} "
          f"{'era后均值%':>10} {'era后总%':>9}")
    for slip in (0.0, 0.001, 0.002, 0.003, 0.005, 0.01):
        r2 = (c_of(slip) * g - 1) * 100
        r2e = r2[era_m]
        print(f"{slip*1000:6.0f}‰ {np.mean(r2):9.3f} {100*np.mean(r2>0):6.1f} "
              f"{100*(np.exp(np.log1p(r2/100).sum())-1):9.1f} "
              f"{np.mean(r2e):10.3f} {100*(np.exp(np.log1p(r2e/100).sum())-1):9.1f}")
    # 找破零点
    lo, hi = 0.001, 0.02
    for _ in range(40):
        mid = (lo + hi) / 2
        if np.mean((c_of(mid) * g - 1) * 100) > 0:
            lo = mid
        else:
            hi = mid
    print(f"  单笔均值归零的滑点≈{lo*1000:.1f}‰(双边合计≈{2*lo*100:.2f}%), "
          f"当前千1下均值+{np.mean(ret):.3f}%、往返成本≈0.30%")
    # 分年: 基准 vs 千3
    yrs = pd.Series([d[:4] for d in dates])
    r3 = (c_of(0.003) * g - 1) * 100
    tbl = pd.DataFrame({"y": yrs, "base": ret, "slip3": r3})
    piv = tbl.groupby("y").agg(n=("base", "size"), avg_b=("base", "mean"),
                               avg_3=("slip3", "mean"), wr=("base", lambda s: 100 * (s > 0).mean()))
    print("  分年(2024起加*): 均值%基准 → 千3滑点")
    for y, row in piv.iterrows():
        star = "*" if int(y) >= 2024 else ""
        print(f"    {y}{star}: n={int(row.n):3d}  {row.avg_b:+7.3f} → {row.avg_3:+7.3f}  (胜率{row.wr:.0f}%)")

    # ---------- 3) 容量 ----------
    print("\n-- 检验3: 容量(执行日当日成交额; 单吊100%权益参与率口径) --")
    bdates = sorted({t["buy_date"] for t in trades})

    def read_one(d: str) -> pd.Series:
        df = pd.read_csv(ROOT / "data" / "panel" / f"{d}.csv.gz", compression="gzip",
                         usecols=["ts_code", "amount"], dtype={"ts_code": str})
        return df.set_index("ts_code")["amount"]

    with ThreadPoolExecutor(8) as x:
        amt_map = dict(zip(bdates, x.map(read_one, bdates)))
    amts = np.array([float(amt_map[t["buy_date"]].get(t["code"], np.nan)) for t in trades])
    ok = ~np.isnan(amts)
    print(f"  执行日成交额(亿元): 中位{np.median(amts[ok])/1e8:.2f} "
          f"p25{np.percentile(amts[ok],25)/1e8:.2f} p75{np.percentile(amts[ok],75)/1e8:.2f} "
          f"最小{amts[ok].min()/1e8:.2f} 最大{amts[ok].max()/1e8:.2f} (缺{int((~ok).sum())}笔)")
    # 权益路径(单吊100%复利, 已执行笔间权益不变)
    eq = 1e6 * np.exp(np.cumsum(np.r_[0.0, lg[:-1]]))
    part = eq / amts * 100   # 我们的单笔成交额占该票当日成交额比例%
    pm = part[ok & (part > 0)]
    print(f"  单笔权益/当日成交额(参与率%): 中位{np.median(pm):.3f} "
          f"p90{np.percentile(pm,90):.2f} 最大{pm.max():.2f}")
    for cap in (0.5, 1.0, 2.0, 5.0):
        # 参与率上限cap%: 权益E可承接 ⇔ E ≤ cap%×amount; 报告当前路径可行笔占比
        feas = (part <= cap).mean() * 100
        # 反解: 该参与率下支持全部era后交易的中位规模上限
        lim = np.percentile(amts[ok], 50) * cap / 100
        print(f"  参与率≤{cap:.1f}%口径: 当前100万起步路径 {feas:.1f}% 笔可全额承接; "
              f"该口径单笔规模上限(中位数票)≈{lim/1e4:.0f}万")
    # 规模-可行性曲线(era后交易)
    print("  规模敏感性(era后交易可全额承接占比):")
    for E in (1e6, 3e6, 1e7, 3e7, 1e8):
        for cap in (1.0,):
            f = (amts[ok & era_m] * cap / 100 >= E).mean() * 100
            print(f"    权益{E/1e8:g}亿: {f:.1f}%")
    print("\n== 检验完(零参数调整, 仅诊断) ==")


if __name__ == "__main__":
    main()
