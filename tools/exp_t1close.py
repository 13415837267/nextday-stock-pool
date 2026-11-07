# -*- coding: utf-8 -*-
"""第 169 轮: seg1 是 alpha 主体, 但买入当天卖不掉 —— 它可预判吗? 条件化卖出能兑现吗?

背景(第 168 轮): 353 笔的毛收益三段分解为 seg1(T+1 日内) +1.153% / seg2(隔夜) +0.069% /
seg3(T+2 日内) +0.500%; 对数份额 68.6% / 3.6% / 27.8%。alpha 在"买进当天上午"就释放完,
但 T+1 制度强制过夜。于是只剩两个可操作的问题:

  A. seg1 的强度, 在 T+1 开盘(9:25 竞价成交)那一刻能不能预判?
     —— 若可, 仓位该按"事件强度"而不是评分下; 若不可, 就不该为它加任何条件。
  B. 既然 seg1 已经兑现成价格, 那"T+1 收盘有多强"能不能预示 T+2 隔夜/日内的回吐?
     —— 这是第 166/168 轮没测过的**可实现**版本: 166 轮按 T+2 竞价缺口 g 条件化, 但 g
        要在 T+2 09:25 才知道, 无法据它下"开盘卖"的单; 而 T+1 收盘强度在 T+1 14:55 就已知,
        完全可以据此决定次日"开盘卖 or 收盘卖"。

口径: 与主策略逐位一致(T 日信号 → T+1 开盘全额买 → T+2 收盘卖), 全复权, 毛收益(未扣费)。
      费率对"开盘卖/收盘卖"两腿相同, 不影响两腿的相对比较; 逐笔复利只作相对口径(与 168 轮一致)。
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ["BT_FULL_TRADES"] = "1"

from src.backtest import prepare_context, run_backtest  # noqa: E402
from src.config import load_config  # noqa: E402

OUT = os.path.join(ROOT, "output")
RNG = np.random.default_rng(20261108)
BINS_GAP = [-99, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 2, 99]


def norm(d):
    return str(d).replace("-", "")[:8]


def market_series(by_date, dates):
    """全市场等权指数(用中位涨跌幅), 返回 {date: level}。"""
    lvl, cur = {}, 1.0
    for d in dates:
        df = by_date.get(d)
        if df is None or not len(df):
            continue
        v = df["pct"].to_numpy(dtype="float64")
        v = v[~np.isnan(v)]
        if len(v):
            cur *= 1 + float(np.median(v)) / 100.0
        lvl[d] = cur
    return lvl


def main():
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    cfg = load_config()
    ctx = prepare_context(cfg, days)
    r = run_backtest(cfg, days, ctx)
    m = r["metrics"]
    print(f"\n基线: {m['总收益率%']}% / 回撤 {m['最大回撤%']}% / 卡玛 {m['卡玛比率']} / {m['交易次数']} 笔")

    dates = list(ctx["by_date"].keys())
    idx = {d: i for i, d in enumerate(dates)}
    per = ctx["per_code"]
    lvl = market_series(ctx["by_date"], dates)

    rows = []
    for t in r["trades_all"]:
        if t.get("ret_pct") is None:
            continue
        d1, d2 = norm(t["buy_date"]), norm(t["sell_date"])
        i = idx.get(d1)
        if i is None or i == 0 or d2 not in idx:
            continue
        sub = per.get(t["code"])
        if sub is None or d1 not in sub.index or d2 not in sub.index:
            continue
        a, b = sub.loc[d1], sub.loc[d2]
        c0 = sub.loc[dates[i - 1]]
        if min(a["open_adj"], a["close_adj"], b["open_adj"], b["close_adj"], c0["close_adj"]) <= 0:
            continue
        s1 = a["close_adj"] / a["open_adj"] - 1
        s2 = b["open_adj"] / a["close_adj"] - 1
        s3 = b["close_adj"] / b["open_adj"] - 1
        gap = a["open_adj"] / c0["close_adj"] - 1
        rng_ = float(a["high"]) - float(a["low"])
        cp = (a["close_adj"] - a["open_adj"]) / rng_ if rng_ > 0 else np.nan
        cp = min(1.0, max(0.0, cp)) if cp == cp else np.nan
        mk = lvl.get(d2, np.nan) / lvl.get(dates[i - 1], np.nan) - 1
        rows.append({"signal_date": norm(t["signal_date"]), "d1": d1, "year": d1[:4],
                     "code": t["code"], "ret": t["ret_pct"] / 100, "score": t.get("score"),
                     "pool_n": t.get("pool_n"), "gap_exec": t.get("gap_pct"),
                     "gap": gap, "s1": s1, "s2": s2, "s3": s3, "mkt": mk, "cp": cp,
                     "pos": t.get("pos_frac"), "hold_open": b["open_adj"] / a["open_adj"] - 1,
                     "hold_close": b["close_adj"] / a["open_adj"] - 1})
    df = pd.DataFrame(rows)
    print(f"可算 {len(df)} 笔  {df['d1'].min()} ~ {df['d1'].max()}")

    out = {"n": int(len(df)), "days": days, "baseline": m}

    # ---------------- A. seg1 可预判性 ----------------
    print("\n=== A. seg1(T+1 日内) 的分桶均值: 按买入日竞价缺口 ===")
    df["gb"] = pd.cut(df["gap"] * 100, BINS_GAP)
    g = df.groupby("gb", observed=True).agg(n=("s1", "size"), s1=("s1", "mean"), total=("ret", "mean"),
                                            win=("ret", lambda x: (x > 0).mean() * 100))
    g["s1"] = (g["s1"] * 100).round(3)
    g["total"] = (g["total"] * 100).round(3)
    g["win"] = g["win"].round(1)
    print(g.to_string())
    out["gap_bucket"] = {str(k): {"n": int(v["n"]), "s1": float(v["s1"]), "total": float(v["total"]),
                                 "win": float(v["win"])} for k, v in g.iterrows()}

    print("\n=== A2. seg1 vs 同日候选池深 / 评分 ===")
    for col, bins in (("pool_n", [0, 5, 10, 20, 30, 50, 9999]),
                      ("score", [0, 1.2, 1.6, 2.0, 2.5, 99])):
        sub = df[df[col].notna()].copy()
        if not len(sub):
            continue
        sub["b"] = pd.cut(sub[col], bins)
        gg = sub.groupby("b", observed=True).agg(n=("s1", "size"), s1=("s1", "mean"), tot=("ret", "mean"))
        gg["s1"] = (gg["s1"] * 100).round(3)
        gg["tot"] = (gg["tot"] * 100).round(3)
        print(f"-- {col} --")
        print(gg.to_string())

    ics = {}
    for col in ("gap", "pool_n", "score"):
        sub = df[["s1", col]].dropna()
        if len(sub) > 30:
            ics[col] = float(sub["s1"].corr(sub[col], method="spearman"))
    print("Spearman IC vs seg1:", {k: round(v, 3) for k, v in ics.items()})
    out["ic"] = ics

    # 关键: seg1 越强, 后面两天(seg2+seg3)是不是越弱?
    df["rest"] = (1 + df["s1"]) * (1 + df["s2"]) * (1 + df["s3"]) - 1 - df["s1"]
    df["rest2"] = df["s2"] + df["s3"]
    print("\n=== A3. 按 seg1 分位看 \"剩下的两天\"(seg2+seg3) ===")
    df["sq"] = pd.qcut(df["s1"], 5, duplicates="drop")
    q = df.groupby("sq", observed=True).agg(n=("s1", "size"), s1=("s1", "mean"), rest=("rest2", "mean"),
                                            total=("ret", "mean"))
    q[["s1", "rest", "total"]] = (q[["s1", "rest", "total"]] * 100).round(3)
    print(q.to_string())
    out["seg1_quintile"] = {str(k): {"n": int(v["n"]), "s1": float(v["s1"]),
                                     "rest": float(v["rest"]), "total": float(v["total"])}
                            for k, v in q.iterrows()}

    # ---------------- B. 条件化卖出(T+1 收盘强度 -> T+2 开盘卖) ----------------
    print("\n=== B. 规则: T+1 收盘涨幅 >= 阈值 时 T+2 开盘卖, 否则收盘卖 ===")
    base_comp = float(np.prod(1 + df["hold_close"]) - 1) * 100
    rowsB = []
    for th in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0):
        sw = (df["s1"] * 100) >= th
        r_rule = np.where(sw, df["hold_open"], df["hold_close"])
        comp = float(np.prod(1 + r_rule) - 1) * 100
        # 随机对照: 同样切换比例, 随机选笔
        k = int(sw.sum())
        trials = 300
        comps = np.empty(trials)
        for ti in range(trials):
            pick = RNG.random(len(df)) < (k / len(df))
            comps[ti] = np.prod(np.where(pick, df["hold_open"], df["hold_close"])) - 1
        pct = float((comps < comp / 100).mean() * 100)
        rowsB.append({"th": th, "n_open": k, "share": k / len(df) * 100, "comp": comp,
                      "delta": comp - base_comp, "mean": float(np.mean(r_rule) * 100),
                      "rand_p50": float(np.median(comps) * 100),
                      "rand_p5": float(np.percentile(comps, 5) * 100),
                      "rand_p95": float(np.percentile(comps, 95) * 100),
                      "pctile": pct})
        print(f"  阈值 {th:>4}%: 开盘卖 {k:>3} 笔({k/len(df)*100:4.1f}%) 逐笔复利 {comp:>10.1f}% "
              f"Δ {comp-base_comp:+8.1f}pp 均值 {np.mean(r_rule)*100:+.3f}% "
              f"| 随机中位 {np.median(comps)*100:>9.1f}% 5~95 [{np.percentile(comps,5)*100:.0f}, "
              f"{np.percentile(comps,95)*100:.0f}] 分位 {pct:.0f}")
    print(f"  基线(全收盘卖) 逐笔复利 {base_comp:.1f}%")
    out["rule"] = {"base_comp": base_comp, "rows": rowsB}

    # 反向: 弱收盘也开盘卖, 只为确认方向
    print("\n=== B2. 反向对照: T+1 收盘涨幅 <= 阈值 时开盘卖(其余收盘卖) ===")
    for th in (-1.0, -2.0, -3.0):
        sw = (df["s1"] * 100) <= th
        r_rule = np.where(sw, df["hold_open"], df["hold_close"])
        comp = float(np.prod(1 + r_rule) - 1) * 100
        print(f"  阈值 {th:>5}%: 开盘卖 {int(sw.sum()):>3} 笔 复利 {comp:.1f}% Δ {comp-base_comp:+.1f}pp")

    # B4. 用 T+1 的"收盘位置"(收盘在当日振幅中的位置)代替涨幅 —— 同样 T+1 14:55 可得
    print("\n=== B4. T+1 收盘位置(0=最低 1=最高) 分桶 -> 后续两天 ===")
    sub = df[df["cp"].notna()].copy()
    if len(sub) > 50:
        sub["cb"] = pd.cut(sub["cp"], [0, 0.2, 0.4, 0.6, 0.8, 1.0])
        t = sub.groupby("cb", observed=True).agg(n=("cp", "size"), s1=("s1", "mean"),
                                                 rest=("s2", "mean"))
        t["rest3"] = sub.groupby("cb", observed=True)["s3"].mean()
        t["rest"] = t.pop("rest") + t.pop("rest3")
        t[["s1", "rest"]] = (t[["s1", "rest"]] * 100).round(3)
        print(t.to_string())
        out["close_pos"] = {str(k): {"n": int(v["n"]), "s1": float(v["s1"]), "rest": float(v["rest"])}
                            for k, v in t.iterrows()}
        for th in (0.6, 0.8):
            sw = sub["cp"] >= th
            r_rule = np.where(sw, sub["hold_open"], sub["hold_close"])
            print(f"  规则 收盘位置>={th}: 开盘卖 {int(sw.sum())} 笔 复利 "
                  f"{(np.prod(1+r_rule)-1)*100:.1f}% vs 全收盘 {(np.prod(1+sub['hold_close'])-1)*100:.1f}%")

    # 分年: 规则 vs 基线(阈值取 2%)
    print("\n=== B3. 分年(阈值 2%): 收盘卖 vs 条件卖 ===")
    sw = (df["s1"] * 100) >= 2.0
    df["rule"] = np.where(sw, df["hold_open"], df["hold_close"])
    y = df.groupby("year").apply(lambda x: pd.Series({
        "n": len(x), "base": np.prod(1 + x["hold_close"]) - 1,
        "rule": np.prod(1 + x["rule"]) - 1}))
    y[["base", "rule"]] = (y[["base", "rule"]] * 100).round(1)
    print(y.to_string())
    out["yearly_rule2"] = {str(k): {"n": int(v["n"]), "base": float(v["base"]), "rule": float(v["rule"])}
                           for k, v in y.iterrows()}

    # ---------------- C. 市场 beta 占比 ----------------
    print("\n=== C. 单笔两日收益的市场 beta ===")
    sub = df[["ret", "mkt"]].dropna()
    if len(sub) > 50:
        b = float(np.cov(sub["ret"], sub["mkt"])[0, 1] / np.var(sub["mkt"], ddof=1))
        a = float(sub["ret"].mean() - b * sub["mkt"].mean())
        corr = float(sub["ret"].corr(sub["mkt"]))
        r2 = corr ** 2
        hedged = sub["ret"] - b * sub["mkt"]
        print(f"  beta {b:.3f}  alpha {a*100:+.3f}%/笔  corr {corr:.3f}  R² {r2:.3f}")
        print(f"  对冲后: 均值 {hedged.mean()*100:+.3f}%  逐笔复利 "
              f"{(np.prod(1+hedged)-1)*100:.1f}% vs 未对冲 {(np.prod(1+sub['ret'])-1)*100:.1f}%")
        print(f"  标准差: 未对冲 {sub['ret'].std()*100:.3f}% -> 对冲后 {hedged.std()*100:.3f}%")
        out["beta"] = {"beta": b, "alpha_pct": a * 100, "corr": corr, "r2": r2,
                       "hedged_mean": float(hedged.mean() * 100),
                       "hedged_comp": float((np.prod(1 + hedged) - 1) * 100),
                       "raw_comp": float((np.prod(1 + sub["ret"]) - 1) * 100),
                       "sd_raw": float(sub["ret"].std() * 100),
                       "sd_hedged": float(hedged.std() * 100)}

    # ---------------- D. 仓位标定审计 ----------------
    print("\n=== D. 现有下注档位 vs 实际单笔收益(仓位=下注乘数) ===")
    sub = df[df["pos"].notna()].copy()
    if len(sub) > 50:
        sub["pos"] = sub["pos"].astype(float)
        sub["pb"] = pd.cut(sub["pos"], [0, 0.35, 0.55, 0.8, 0.99, 1.01])
        a = sub.groupby("pb", observed=True).agg(n=("ret", "size"), mean=("ret", "mean"),
                                                 win=("ret", lambda x: (x > 0).mean() * 100),
                                                 s1=("s1", "mean"))
        a[["mean", "win", "s1"]] = (a[["mean", "s1"]] * 100).round(3)
        a["win"] = a["win"].round(1)
        print(a.to_string())
        # 若把所有笔都按 1.0 仓(不复利差异)近似对比: 单位仓位复利 vs 现行
        unit = float(np.prod(1 + sub["ret"] / sub["pos"]) - 1) * 100
        cur = float(np.prod(1 + sub["ret"]) - 1) * 100
        print(f"  单位仓位口径复利 {unit:.0f}%  vs 现行加权 {cur:.0f}%")
        out["pos_audit"] = {"buckets": {str(k): {"n": int(v["n"]), "mean": float(v["mean"]),
                                                 "win": float(v["win"]), "s1": float(v["s1"])}
                                        for k, v in a.iterrows()},
                            "unit_comp": unit, "cur_comp": cur}

    json.dump(out, open(os.path.join(OUT, "exp_t1close.json"), "w"), ensure_ascii=False, indent=1)
    print("\nsaved output/exp_t1close.json")


if __name__ == "__main__":
    main()
