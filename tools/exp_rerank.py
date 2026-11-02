"""T+1 开盘横截面重排：评分名次 vs 竞价缺口，哪个才是更好的排序键？

背景
----
现行实现：T 日按池内 z 分排出候选 -> 只留第 1 名(exec_top_n=1) -> T+1 开盘只看它缺口是否 < 0
-> 不够深就空仓, 深就单吊满仓。排序键只有"评分"一个, 而评分是 T 日收盘后的横截面强弱。

但 T+1 9:25 集合竞价结束时, 前 N 名的竞价缺口**全部可见** —— 这是决策时点已知、却
完全没被使用的第二排序键。第一百零一轮的证据(缺口<=-1% 的 92 笔均值 +1.85%/胜率 75%,
缺口>=0.5% 的 120 笔 -0.15%)说明缺口有强区分力, 但那条证据只被用于"调仓位", 从未用于"换票"。
第七十八轮测过"扩池顺延"(top1 不合格就买 top2), 只多 9~12 笔 —— 那是"补漏", 不是"重排"。
本诊断测的是后者: top1 已经合格时, 是否该换成缺口更深的 top2。

三个问题
--------
Q1 前 N 名的缺口相关性多高？（若几乎同向, 重排根本无从发生）
Q2 前两名"都能买(缺口<0)"的日子里, 第 2 名比第 1 名差多少？（配对, 方差大幅缩小）
Q3 换成"前 N 名里缺口最深的那只", 单笔期望变好还是变差？

口径
----
裸信号层: T+1 开盘买 / T+2 收盘卖, 含滑点+双边佣金+印花税, 不含资金约束与任何仓位规则。
重排只使用 9:25 竞价缺口(T+1 开盘时点已知), **无未来函数**。

注意: 这是"孤立单笔期望"。裸信号层没有仓位/风控/复利路径, 因此只用于比较排序键,
不用于预测组合收益。
"""
import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.backtest import prepare_context
from src.config import load_config
from src.selector import select_stocks


def _paired(a: list[float], b: list[float]) -> dict:
    """配对差 a-b 的均值与正态近似 t 检验（n>30 时足够）。"""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = len(d)
    if n < 3:
        return {"n": n}
    m = float(d.mean())
    sd = float(d.std(ddof=1))
    se = sd / math.sqrt(n) if sd > 0 else 0.0
    t = m / se if se > 0 else 0.0
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    return {"n": n, "mean_diff": round(m, 4), "t": round(t, 2), "p": round(p, 4)}


def _stats(v: list[float]) -> dict:
    if not v:
        return {"n": 0}
    a = np.asarray(v, dtype=float)
    return {"n": len(a), "avg": round(float(a.mean()), 4),
            "med": round(float(np.median(a)), 4),
            "wr": round(float((a > 0).mean() * 100), 1),
            "sd": round(float(a.std(ddof=1)) if len(a) > 1 else 0.0, 3)}


def _corr(x: list[float], y: list[float]) -> float:
    if len(x) < 5:
        return float("nan")
    a, b = np.asarray(x, float), np.asarray(y, float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser(description="T+1 开盘横截面重排诊断")
    ap.add_argument("--days", type=int, default=1500)
    ap.add_argument("--topk", type=int, default=6, help="每日取主池前 K 名")
    ap.add_argument("--out", default="output/exp_rerank.json")
    args = ap.parse_args()

    cfg = load_config()
    ex = cfg["execution"]
    slip = float(ex.get("slippage", 0))
    comm = float(ex.get("commission_rate", 0))
    stamp = float(ex.get("stamp_tax", 0))
    gap_floor = float(ex.get("buy_gap_below", 0) or 0)
    min_score = ex.get("min_top1_score")
    fee = comm * 2 + stamp
    print(f"成本口径: 滑点{slip*1000:g}‰ + 佣金{comm*10000:g}万×2 + 印花税{stamp*10000:g}万")
    print(f"现行规则: 承接线 gap<={gap_floor:g}% / 评分门槛 {min_score}\n", flush=True)

    t0 = time.time()
    ctx = prepare_context(cfg, args.days)
    dates = ctx["main_dates"]
    by_date = ctx["by_date"]
    per_code = ctx["per_code"]
    names = ctx["names"]
    didx = {d: i for i, d in enumerate(dates)}
    print(f"面板就绪: {len(dates)} 日, {time.time()-t0:.0f}s\n", flush=True)

    recs: list[dict] = []      # 逐 (日, 名次) 记录
    n_day = 0
    n_pool0 = 0
    pool_sizes: list[int] = []
    for k, d in enumerate(dates):
        df = by_date.get(d)
        if df is None:
            continue
        i = didx.get(d)
        if i is None or i + 2 >= len(dates):
            continue
        picks, stats = select_stocks(d, df, cfg, names, name_filter=False)
        main = [p for p in picks if p.get("pool", "main") == "main"]
        pool_n = int((stats or {}).get("scored") or 0)
        if not main:
            n_pool0 += 1
            continue
        n_day += 1
        pool_sizes.append(pool_n)

        bd, sd = dates[i + 1], dates[i + 2]
        # T 日"市场典型个股涨跌"(中位数), 用于刀口保护口径
        pv = df["pct"].to_numpy(dtype="float64")
        pv = pv[~np.isnan(pv)]
        mkt_ret = float(np.median(pv)) if len(pv) else float("nan")

        for rank, p in enumerate(main[:args.topk], 1):
            code = p["code"]
            base = {"signal_date": d, "buy_date": bd, "rank": rank,
                    "code": code, "name": names.get(code, ""),
                    "score": round(float(p.get("score") or 0.0), 4),
                    "pool_n": pool_n}
            sub = per_code.get(code)
            if sub is None or bd not in sub.index or sd not in sub.index:
                recs.append({**base, "ok": False, "why": "缺行情"})
                continue
            rb, rs = sub.loc[bd], sub.loc[sd]
            if float(rb.get("is_paused") or 0) == 1 or float(rb.get("vol") or 0) <= 0:
                recs.append({**base, "ok": False, "why": "停牌"})
                continue
            if float(rb["open"]) >= float(rb["high_limit"]) - 1e-9:
                recs.append({**base, "ok": False, "why": "一字涨停"})
                continue
            sig = df[df["ts_code"] == code]
            if sig.empty:
                recs.append({**base, "ok": False, "why": "缺信号日行情"})
                continue
            pc = float(sig.iloc[0]["close_adj"])
            if pc <= 0:
                recs.append({**base, "ok": False, "why": "缺前收"})
                continue
            gap = float(rb["open_adj"]) / pc - 1.0
            bo, sc = float(rb["open_adj"]), float(rs["close_adj"])
            if bo <= 0 or sc <= 0:
                recs.append({**base, "ok": False, "why": "缺价格"})
                continue
            ret = ((sc * (1 - slip)) / (bo * (1 + slip)) - 1.0 - fee) * 100.0
            # 刀口保护: 市场不跌而个股超额下跌 => 现行为 mult=0(弃单)
            pct_sig = float(sig.iloc[0]["pct"]) if "pct" in sig.columns else float("nan")
            knife = (not math.isnan(mkt_ret) and not math.isnan(pct_sig)
                     and mkt_ret > -0.5 and (pct_sig - mkt_ret) <= -4.0)
            recs.append({**base, "ok": True, "gap_pct": round(gap * 100.0, 3),
                         "ret_pct": round(ret, 4), "mkt_ret": round(mkt_ret, 3),
                         "knife": bool(knife),
                         "gate_ok": (min_score is None or float(p.get("score") or 0) >= float(min_score)),
                         "gap_ok": gap * 100.0 <= gap_floor})

    print(f"扫描完成: {n_day} 个出票日 / {n_pool0} 个空池日 / {len(recs)} 条记录 "
          f"/ {time.time()-t0:.0f}s", flush=True)
    rdf = pd.DataFrame(recs)
    good = rdf[rdf["ok"] == True].copy()          # noqa: E712
    if good.empty:
        raise SystemExit("没有任何可用记录: 面板或选股返回为空")
    good["gap_pct"] = good["gap_pct"].astype(float)
    good["ret_pct"] = good["ret_pct"].astype(float)

    out: dict = {
        "meta": {
            "days": args.days, "topk": args.topk, "start": str(dates[0]),
            "end": str(dates[-1]), "signal_days": n_day, "empty_pool_days": n_pool0,
            "records": len(rdf), "usable": len(good),
            "slippage": slip, "fee_roundtrip": round(fee * 100, 4),
            "gap_floor": gap_floor, "min_score": min_score,
            "run_sec": round(time.time() - t0, 1),
        },
        "pool_sizes": {"median": int(np.median(pool_sizes)) if pool_sizes else 0,
                       "p10": int(np.percentile(pool_sizes, 10)) if pool_sizes else 0,
                       "p90": int(np.percentile(pool_sizes, 90)) if pool_sizes else 0},
    }

    # ---- Q0: 各名次的裸分布 ----
    print("\n" + "=" * 84)
    print("Q0  各名次的裸分布(不设任何门槛)")
    print("=" * 84)
    print(f"{'名次':>4} {'笔数':>5} {'均值%':>8} {'中位%':>8} {'胜率%':>7} {'均缺口%':>9} {'缺口<0占比':>10}")
    by_rank = {}
    for r, g in good.groupby("rank"):
        s = _stats(g["ret_pct"].tolist())
        by_rank[str(int(r))] = {**s, "gap_mean": round(float(g["gap_pct"].mean()), 3),
                                "gap_neg_share": round(float((g["gap_pct"] < 0).mean() * 100), 1)}
        print(f"{int(r):>4} {s['n']:>5} {s['avg']:>8.3f} {s['med']:>8.3f} {s['wr']:>7.1f} "
              f"{by_rank[str(int(r))]['gap_mean']:>9.3f} {by_rank[str(int(r))]['gap_neg_share']:>10.1f}")
    out["by_rank"] = by_rank

    # ---- Q1: 缺口相关性 ----
    print("\n" + "=" * 84)
    print("Q1  前 N 名缺口的相关性(同日横截面)")
    print("=" * 84)
    piv = good.pivot_table(index="signal_date", columns="rank", values="gap_pct")
    corr = {}
    for r in range(2, args.topk + 1):
        if 1 in piv.columns and r in piv.columns:
            sub = piv[[1, r]].dropna()
            if len(sub) >= 5:
                c = _corr(sub[1].tolist(), sub[r].tolist())
                corr[f"r1_vs_r{r}"] = {"n": len(sub), "corr": round(c, 3),
                                       "deep_share": round(float((sub[r] < sub[1]).mean() * 100), 1)}
                print(f"  rank1 vs rank{r:<2} n={len(sub):5d} corr={c:+.3f} "
                      f"rank{r} 更深的占比={corr[f'r1_vs_r{r}']['deep_share']:.1f}%")
    if 1 in piv.columns and 2 in piv.columns:
        sp = (piv[2] - piv[1]).dropna()
        if len(sp):
            out["gap_spread_r1_r2"] = {
                "n": len(sp), "mean": round(float(sp.mean()), 3),
                "p25": round(float(sp.quantile(.25)), 3), "med": round(float(sp.median()), 3),
                "p75": round(float(sp.quantile(.75)), 3),
            }
            print(f"\n  gap(rank2)-gap(rank1): 中位 {sp.median():+.3f}% / IQR "
                  f"[{sp.quantile(.25):+.3f}, {sp.quantile(.75):+.3f}]"
                  f" / |差|>1% 占 {(sp.abs() > 1).mean()*100:.1f}%")
    out["corr"] = corr

    # ---- Q2: 前两名都"能买"的配对 ----
    both = piv[[1, 2]].dropna().index if (1 in piv.columns and 2 in piv.columns) else []
    r1 = good[good["rank"] == 1].set_index("signal_date")
    r2 = good[good["rank"] == 2].set_index("signal_date")
    common = [d for d in both if d in r1.index and d in r2.index]
    b1 = r1.loc[common]
    b2 = r2.loc[common]
    buyable = (b1["gap_ok"]) & (b1["gate_ok"]) & (~b1["knife"]) & \
              (b2["gap_ok"]) & (b2["gate_ok"]) & (~b2["knife"])
    idx = b1.index[buyable]
    print("\n" + "=" * 84)
    print("Q2  前两名都符合现行买入条件(缺口<0 / 评分达标 / 未触发刀口)的配对比较")
    print("=" * 84)
    print(f"  可配对交易日: {len(idx)}")
    q2: dict = {"n_pairs": int(len(idx))}
    if len(idx) >= 5:
        a, b = b1.loc[idx], b2.loc[idx]
        pv_ = _paired(a["ret_pct"].tolist(), b["ret_pct"].tolist())
        q2["rank1_minus_rank2"] = pv_
        print(f"  rank1 均值 {a['ret_pct'].mean():+.3f}% / rank2 均值 {b['ret_pct'].mean():+.3f}%")
        print(f"  配对差(rank1-rank2) = {pv_['mean_diff']:+.4f}pp  t={pv_['t']}  p={pv_['p']}")
        # 按 gap 差分组: rank2 更深 vs 更浅
        dgap = (b["gap_pct"] - a["gap_pct"])
        for lab, m in [("rank2 缺口更深", dgap < 0), ("rank2 缺口更浅", dgap >= 0)]:
            if m.sum() < 3:
                continue
            sub_a, sub_b = a[m], b[m]
            pvs = _paired(sub_a["ret_pct"].tolist(), sub_b["ret_pct"].tolist())
            q2[lab] = {**pvs, "n": int(m.sum()),
                       "rank1_avg": round(float(sub_a["ret_pct"].mean()), 3),
                       "rank2_avg": round(float(sub_b["ret_pct"].mean()), 3),
                       "dgap_med": round(float(dgap[m].median()), 3)}
            print(f"    {lab:14s} n={m.sum():4d}  rank1 {sub_a['ret_pct'].mean():+7.3f}% "
                  f"rank2 {sub_b['ret_pct'].mean():+7.3f}%  差 {pvs['mean_diff']:+.3f}pp "
                  f"(t={pvs['t']}, p={pvs['p']}, Δgap中位 {dgap[m].median():+.2f}%)")
        # 只有"缺口差足够大"时才谈换票
        big = dgap <= -1.0
        if big.sum() >= 3:
            pvs = _paired(a[big]["ret_pct"].tolist(), b[big]["ret_pct"].tolist())
            q2["rank2_deeper_1pct"] = {**pvs, "n": int(big.sum()),
                                       "rank1_avg": round(float(a[big]["ret_pct"].mean()), 3),
                                       "rank2_avg": round(float(b[big]["ret_pct"].mean()), 3)}
            print(f"    其中 rank2 深 >=1%: n={big.sum():3d}  rank1 {a[big]['ret_pct'].mean():+7.3f}% "
                  f"rank2 {b[big]['ret_pct'].mean():+7.3f}%  差 {pvs['mean_diff']:+.3f}pp "
                  f"(t={pvs['t']}, p={pvs['p']})")
    out["pair_both_buyable"] = q2

    # ---- Q3: 换排序键的配对比较（严格同日子集: 只在 rank1 可买的日子上换） ----
    print("\n" + "=" * 84)
    print("Q3  换排序键: 在 rank1 可买的同一批日子上, 逐日比较各规则选出的票")
    print("=" * 84)
    # cand[d] = [(rank, gap, ret, score), ...] 仅含"符合现行买入条件"的票
    cand: dict[str, list[tuple]] = {}
    for r in range(1, args.topk + 1):
        rr = good[good["rank"] == r].set_index("signal_date")
        rr = rr[(rr["gap_ok"]) & (rr["gate_ok"]) & (~rr["knife"])]
        for d_, row in rr.iterrows():
            cand.setdefault(d_, []).append((int(r), float(row["gap_pct"]),
                                            float(row["ret_pct"]), float(row["score"])))
    for d_ in cand:
        cand[d_].sort(key=lambda x: x[0])
    rules = {"base_rank1": lambda v: v[0],
             "deep_in_2": lambda v: min(v[:2], key=lambda x: x[1]),
             "deep_in_3": lambda v: min(v[:3], key=lambda x: x[1]),
             "deep_in_5": lambda v: min(v[:5], key=lambda x: x[1]),
             "score_x_gap": lambda v: max(v[:3], key=lambda x: x[3] * (-x[1]))}
    base_days = sorted(d_ for d_, v in cand.items() if v[0][0] == 1)
    q3 = {}
    if len(base_days) >= 5:
        base_ret = [cand[d_][0][2] for d_ in base_days]
        s0 = _stats(base_ret)
        q3["base_rank1"] = {**s0, "days": len(base_days),
                            "avg_gap": round(float(np.mean([cand[d_][0][1] for d_ in base_days])), 3)}
        print(f"  {'base_rank1':12s} 天数={len(base_days):4d} 均值={s0['avg']:+7.3f}% "
              f"胜率={s0['wr']:5.1f}% 均缺口={q3['base_rank1']['avg_gap']:+6.2f}%  (基准)")
        for name, fn in rules.items():
            if name == "base_rank1":
                continue
            picks = [fn(cand[d_]) for d_ in base_days]
            rets = [p[2] for p in picks]
            pv_ = _paired(rets, base_ret)
            q3[name] = {**_stats(rets), **pv_, "days": len(base_days),
                        "avg_gap": round(float(np.mean([p[1] for p in picks])), 3),
                        "switch_share": round(float(np.mean([p[0] != 1 for p in picks]) * 100), 1)}
            print(f"  {name:12s} 天数={len(base_days):4d} 均值={q3[name]['avg']:+7.3f}% "
                  f"胜率={q3[name]['wr']:5.1f}% 均缺口={q3[name]['avg_gap']:+6.2f}% "
                  f"换票率={q3[name]['switch_share']:4.1f}% "
                  f"vs base {pv_['mean_diff']:+.3f}pp (t={pv_['t']}, p={pv_['p']})")
    out["rules"] = q3

    # ---- 补位样本: rank1 不可买但 rank2+ 可买 ----
    print("\n" + "=" * 84)
    print("Q4  补位样本: rank1 不可买、rank2~K 里有可买的(现行规则下这天是空仓)")
    print("=" * 84)
    fill = {d_: min(v, key=lambda x: x[1]) for d_, v in cand.items()
            if d_ not in set(base_days)}
    if fill:
        vals = [v[2] for v in fill.values()]
        s = _stats(vals)
        out["fill_days"] = {**s, "days": len(fill),
                            "avg_gap": round(float(np.mean([v[1] for v in fill.values()])), 3)}
        print(f"  可补位 {len(fill)} 天(占出票日 {len(fill)/max(n_day,1)*100:.1f}%) "
              f"均值 {s['avg']:+.3f}% 中位 {s['med']:+.3f}% 胜率 {s['wr']:.1f}% "
              f"均缺口 {out['fill_days']['avg_gap']:+.2f}%")
        print("  （第一百二十七轮 diag_skipped 的 939 笔跳过票均值 -0.489%, 与本行同向/反向即为交叉验证）")
    else:
        out["fill_days"] = {"days": 0}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"\n→ {args.out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
