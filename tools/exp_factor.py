# -*- coding: utf-8 -*-
"""组合层副策略可行性筛查: 横截面因子 + 月度换仓 + 等权分散(2026-11-05)。

为什么必须换一个层次
--------------------
第八十九轮已把八个信号族当过"独立主策略"（50/50 独立资金）做十年检验，era 前每笔全负 ——
但那些检验都跑在**同一套执行框架**里：单票满仓、T+1 开盘买、T+2 收盘卖、深低开承接。
换句话说，被否掉的是"换个选股入口，仍用 dip 的出口"。

本脚本测的是另一件事，此前从未覆盖：
  **月度换仓 + 等权 30 只 + 不做缺口择时 + 不看单日形态** 的横截面组合策略。
它与 dip 的 alpha 来源不同（横截面定价偏差 vs 日内恐慌错杀），持有期不同（1 个月 vs 2 天），
资金占用形态不同（分散 vs 单吊），因此才谈得上"完全独立"。

判据（预注册，先写死再看结果）
------------------------------
  1. 多头组合年化**超额**（相对同池等权基准）> 0，且月度超额 t >= 2.0；
  2. 多空十分位价差（诊断用，不可执行）月度均值 > 0 且 t >= 2.0 —— 证明因子本身存在；
  3. 最大回撤 < 35%，卡玛 > 0.5；
  4. 与 dip 月收益相关性 |rho| < 0.3（否则不是"第二条腿"，是同一笔风险的两种表达）。
四条全过才进入建设阶段；只过 1~2 条视为"有苗头但不可用"。

口径
----
  · 信号日 T = 每月最后一个交易日收盘；执行 = 次一交易日**开盘价**买入（与主策略同口径，
    避免用 T 日收盘价成交的乐观偏差）；
  · 持有到下一个调仓日的次一交易日开盘卖出；等权，不做止损、不做择时、不加杠杆；
  · 成本 = 换手率 × 0.30%（滑点千1×2 + 佣金万2.5×2 + 印花税万5），按实际换名比例计；
  · 股票池：非停牌、非 ST、上市满 250 日、当日成交额 >= 1 亿（避免流动性陷阱）；
  · 因子只用 T 日收盘可得的字段；**基准 = 同一股票池的等权月度收益**，超额必须减掉它 ——
    否则测出来的全是 beta（第一百三十四轮那个 +0.622% 的巧合就是 beta 的样本）。

用法
----
  python3 tools/exp_factor.py                # 1500 日
  python3 tools/exp_factor.py --days 3378    # 全史
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context  # noqa: E402
from src.config import board_of, load_config  # noqa: E402

COST = 0.0030
TOP_N = 30
BOT_N = 30
AMT_MIN = 1e8


# ----------------------------------------------------------------------
# 因子定义: (名称, 函数, 方向)  方向 +1 = 值越大越好, -1 = 值越小越好
def _mv(df):
    """流通市值代理: 成交额 / 换手率 (换手率是百分比, 故 ×100)。"""
    return np.log10((df["amount"] * 100 / df["turnover_rate"].replace(0, np.nan)).replace(0, np.nan))


FACTORS = [
    ("size_small", lambda d: _mv(d), -1),          # 小市值
    ("rev_1m", lambda d: -d["mom20"], +1),         # 一个月反转(买输家)
    ("low_ivol", lambda d: -d["ivol20"], +1),      # 低特质波动
    ("low_turn", lambda d: -d["turnover_rate"], +1),   # 低换手
    ("low_amp", lambda d: -d["amp_pct"], +1),      # 低振幅
    ("mom_1m", lambda d: d["mom20"], +1),          # 一个月动量(对照, 预期为负)
    ("near_high", lambda d: d["dist_high20"], +1),  # 逼近20日新高
    ("illiq", lambda d: np.log10((d["amp_pct"] * 1e8 / d["amount"].replace(0, np.nan)).replace(0, np.nan)), +1),
]


def _rank(x):
    """秩(平均秩处理并列), 用于 Spearman。"""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), dtype=float)
    r[order] = np.arange(1, len(x) + 1)
    # 并列取平均秩
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return r


def _spearman(a, b):
    if len(a) < 30:
        return None
    ra, rb = _rank(a), _rank(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else None


def _tstat(v):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 3:
        return None
    s = v.std(ddof=1)
    return float(v.mean() / (s / np.sqrt(len(v)))) if s > 0 else None


def _nav_stats(rets):
    """月度收益序列 -> 年化/最大回撤/卡玛。"""
    r = np.asarray(rets, dtype=float)
    if len(r) == 0:
        return {}
    nav = np.cumprod(1 + r)
    years = len(r) / 12.0
    ann = float(nav[-1] ** (1 / years) - 1) if years > 0 and nav[-1] > 0 else None
    peak = np.maximum.accumulate(nav)
    dd = float((nav / peak - 1).min())
    calmar = (ann / abs(dd)) if ann is not None and dd < 0 else None
    return {"ann": round(ann * 100, 2) if ann is not None else None,
            "mdd": round(dd * 100, 2), "calmar": round(calmar, 2) if calmar else None,
            "months": len(r), "nav": round(float(nav[-1]), 4)}


# ----------------------------------------------------------------------
def _month_pairs(main_dates):
    """每月最后一个交易日为信号日, 次一交易日为执行日。"""
    pairs = []
    for i in range(len(main_dates) - 1):
        if main_dates[i][:6] != main_dates[i + 1][:6]:
            pairs.append((main_dates[i], main_dates[i + 1]))
    return pairs


def _universe(day, cfg):
    """股票池: 非停牌/非ST/上市满250日/成交额达标/板块在内。"""
    boards = set(cfg["universe"]["boards"])
    df = day
    m = ((df["is_paused"] == 0) & (df["is_st"] == 0) & (df["bars"] >= 250)
         & (df["amount"] >= AMT_MIN))
    df = df.loc[m]
    df = df.loc[df["ts_code"].map(lambda c: board_of(c) in boards)]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=1500)
    ap.add_argument("--out", default="output/exp_factor.json")
    args = ap.parse_args()

    cfg = load_config()
    print("[1/2] 构建上下文 %d 日 ..." % args.days, flush=True)
    t0 = time.time()
    ctx = prepare_context(cfg, args.days)
    by_date, main_dates = ctx["by_date"], ctx["main_dates"]
    print("      ctx ready %.0fs, %d 个交易日" % (time.time() - t0, len(main_dates)), flush=True)

    pairs = _month_pairs(main_dates)
    print("[2/2] 月度调仓 %d 期 ..." % len(pairs), flush=True)

    # 逐月: 计算因子值、前瞻收益、基准
    monthly = []
    for k in range(len(pairs) - 1):
        sig_d, exe_d = pairs[k]
        _, exe_next = pairs[k + 1]
        day = _universe(by_date[sig_d], cfg)
        if len(day) < 200:
            continue
        entry = by_date[exe_d]
        exit_ = by_date[exe_next]
        e = entry[["ts_code", "open", "open_adj", "high_limit", "is_paused"]].rename(
            columns={"open": "o", "open_adj": "oa", "high_limit": "hl", "is_paused": "pz"})
        x = exit_[["ts_code", "open_adj"]].rename(columns={"open_adj": "xa"})
        m = day.merge(e, on="ts_code", how="inner").merge(x, on="ts_code", how="left")
        m = m[(m["pz"] == 0) & (m["oa"] > 0) & (m["o"] < m["hl"] - 1e-9)]
        m = m[m["xa"].notna() & (m["xa"] > 0)]
        if len(m) < 200:
            continue
        m["fwd"] = (m["xa"] / m["oa"] - 1) * 100
        base = float(m["fwd"].mean())   # 同池等权基准

        rec = {"sig": sig_d, "exe": exe_d, "exit": exe_next, "n_pool": len(m), "base": base, "f": {}}
        for name, fn, direction in FACTORS:
            try:
                v = fn(m)
            except Exception:  # noqa: BLE001
                continue
            v = np.asarray(v, dtype=float) * direction
            ok = np.isfinite(v)
            if ok.sum() < 200:
                continue
            idx = np.argsort(-v[ok])
            fwd = m["fwd"].to_numpy()[ok]
            n_top = min(TOP_N, len(fwd))
            n_bot = min(BOT_N, len(fwd))
            top = fwd[idx[:n_top]]
            bot = fwd[idx[-n_bot:]]
            # 十分位(用于 IC 之外的诊断)
            ic = _spearman(v[ok], fwd)
            rec["f"][name] = {"top": float(top.mean()), "bot": float(bot.mean()),
                              "ls": float(top.mean() - bot.mean()), "ic": ic,
                              "codes": [str(c) for c in m["ts_code"].to_numpy()[ok][idx[:n_top]]]}
        monthly.append(rec)
        if (k + 1) % 12 == 0:
            print("      %s 完成 %d 期" % (sig_d, k + 1), flush=True)

    print("\n=== 组合层因子筛查 (%d 期, 等权 %d 只, 月度换仓, 扣 %.2f%% 往返) ==="
          % (len(monthly), TOP_N, COST * 100), flush=True)
    print("  %-11s %8s %8s %8s %8s %7s %7s | %8s %7s | %8s %8s" % (
        "因子", "多头年化", "基准年化", "超额年化", "超额t", "IC", "ICIR", "回撤", "卡玛", "多空均值", "多空t"))
    out = {"days": args.days, "n_months": len(monthly), "top_n": TOP_N, "cost_pct": COST * 100,
           "factors": {}}

    # 基准序列
    base_rets = [r["base"] / 100 for r in monthly]
    bst = _nav_stats(base_rets)
    print("  %-11s %8s %8s %8s %8s | 基准: %d 期, 年化 %.2f%%, 回撤 %.2f%%" % (
        "[基准]", "", "", "", "", bst.get("months", 0), bst.get("ann") or 0, bst.get("mdd") or 0), flush=True)
    out["benchmark"] = bst

    # 用实际持仓重叠度算换手成本
    for name, _, _ in FACTORS:
        rows = [r for r in monthly if name in r["f"]]
        if len(rows) < 12:
            continue
        port, ls, ics = [], [], []
        prev = None
        turn_sum = 0.0
        for r in rows:
            d = r["f"][name]
            codes = set(d["codes"])
            if prev is None:
                turn = 1.0
            else:
                turn = 1.0 - len(codes & prev) / max(1, len(codes))
            prev = codes
            turn_sum += turn
            port.append(d["top"] / 100 - turn * COST)   # 多头组合净收益
            ls.append(d["ls"] / 100)                    # 多空价差为诊断口径, 报告税前(不可执行)
            ics.append(d["ic"])
        excess = [p - b for p, b in zip(port, base_rets)]
        ps, bs = _nav_stats(port), _nav_stats(base_rets)
        ex_ann = (ps["ann"] or 0) - (bs["ann"] or 0)
        icm = float(np.mean([x for x in ics if x is not None]))
        icsd = float(np.std([x for x in ics if x is not None], ddof=1))
        icir = icm / icsd if icsd > 0 else 0
        st = {"ann": ps["ann"], "mdd": ps["mdd"], "calmar": ps["calmar"],
              "excess_ann": round(ex_ann, 2), "excess_t": round(_tstat(excess) or 0, 2),
              "ic": round(icm, 4), "icir": round(icir, 3),
              "ic_t": round(icm / (icsd / np.sqrt(len(ics))) if icsd > 0 else 0, 2),
              "ls_mean": round(float(np.mean(ls)) * 100, 3), "ls_t": round(_tstat(ls) or 0, 2),
              "turnover": round(turn_sum / len(rows), 3), "n_months": len(rows)}
        out["factors"][name] = st
        print("  %-11s %+7.1f%% %+7.1f%% %+7.1f%% %+7.2f %7.3f %7.2f | %+7.1f%% %7.2f | %+7.2f%% %6.2f" % (
            name, ps["ann"] or 0, bs["ann"] or 0, ex_ann, st["excess_t"], icm, icir,
            ps["mdd"] or 0, ps["calmar"] or 0, st["ls_mean"], st["ls_t"]), flush=True)

    # 与 dip 的相关性
    dip_monthly = {}
    try:
        bt = json.loads((ROOT / "output" / "backtest.json").read_text(encoding="utf-8"))
        mo = bt.get("monthly")
        if isinstance(mo, list):
            for x in mo:
                if isinstance(x, dict):
                    k = str(x.get("month") or x.get("ym") or "")
                    v = x.get("ret_pct", x.get("ret"))
                    if k and v is not None:
                        dip_monthly[k[:6]] = float(v)
        elif isinstance(mo, dict):
            dip_monthly = {str(k)[:6]: float(v) for k, v in mo.items()}
    except Exception as e:  # noqa: BLE001
        print("  (未能读取 output/backtest.json: %s)" % e, flush=True)

    if dip_monthly:
        print("\n=== 与 dip 月收益的相关性(|rho|<0.3 才算独立第二条腿) ===", flush=True)
        for name in out["factors"]:
            xs, ys = [], []
            for r in monthly:
                k = r["exe"][:6]
                if name in r["f"] and k in dip_monthly:
                    xs.append(r["f"][name]["top"] / 100)
                    ys.append(dip_monthly[k] / 100)
            if len(xs) >= 12:
                rho = float(np.corrcoef(xs, ys)[0, 1])
                out["factors"][name]["corr_dip"] = round(rho, 3)
                print("  %-11s rho=%+.3f (n=%d)" % (name, rho, len(xs)), flush=True)

    # 判据汇总
    print("\n=== 判据(预注册): 超额>0 且 t>=2.0 / 多空 t>=2.0 / 回撤<35% 且卡玛>0.5 / |rho|<0.3 ===",
          flush=True)
    for name, st in out["factors"].items():
        c = [st["excess_ann"] > 0 and st["excess_t"] >= 2.0,
             st["ls_t"] >= 2.0,
             (st["mdd"] or -99) > -35 and (st["calmar"] or 0) > 0.5,
             abs(st.get("corr_dip", 0)) < 0.3]
        out["factors"][name]["pass"] = c
        print("  %-11s %s  (%d/4)" % (name, "".join("PASS " if x else "fail " for x in c), sum(c)),
              flush=True)

    op = Path(args.out)
    if not op.is_absolute():
        op = ROOT / op
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print("\n结果已写 %s" % op, flush=True)


if __name__ == "__main__":
    main()
