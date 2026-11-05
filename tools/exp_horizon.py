# -*- coding: utf-8 -*-
"""出口依赖性检验: 同一批横截面因子在不同持有期下的 IC 衰减(2026-11-05)。

要回答的质疑
------------
前面若干轮的否定 —— 八族独立主策略(89)、trend(133/134)、组合层因子(135) ——
**全部共享同一个出口**: T+1 开盘买、持有 1~2 日、单票集中。
如果 alpha 有半衰期, 用 2 天的尺子去量 60 天的信号, 必然什么都量不到。
于是"只有抄底能赚钱"可能不是市场的性质, 而是**出口筛选出来的**。

本脚本不选股、不模拟资金、不做风控, 只做一件事:
  对 8 个横截面因子, 计算持有 k = 1/2/3/5/10/20/60/120 日的 Spearman IC,
  观察 IC 的符号与强度随 k 怎么变。

判据(预注册, 先写死再看结果)
----------------------------
  A. 若某因子在 k<=5 为负、k>=20 显著转正  -> 它是慢 alpha, 被短出口滤掉了;
  B. 若所有因子在长 k 都不翻正              -> "只有反转"是数据性质, 不是出口产物;
  C. 若反转类 IC 随 k 单调衰减到 0           -> 反转是"快 alpha", 短出口是对的。

口径
----
  · 截面: 每 5 个交易日取一次(增加样本, 避免只有 161 个月度截面);
  · 收益: close_adj[T+1+k] / open_adj[T+1] - 1, 与主策略同"次日开盘买"口径;
  · 剔除: T+1 停牌 / T+1 开盘即一字涨停(买不到) / T+k 无行情(停牌未复牌);
  · IC = 因子值与前瞻收益的秩相关, 不扣成本(排序统计, 成本对所有分位同向);
  · 因子方向统一为"值越大越看多", 因此 IC>0 即方向正确;
  · 股票池: 非停牌/非ST/上市满250日/成交额>=1亿/板块在内(与组合层同池)。

用法
----
  python3 tools/exp_horizon.py                # 1500 日
  python3 tools/exp_horizon.py --days 3378    # 全史
  python3 tools/exp_horizon.py --step 3       # 更密截面
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

KS = [1, 2, 3, 5, 10, 20, 60, 120]
AMT_MIN = 1e8
MIN_POOL = 200


# ----------------------------------------------------------------------
def _rank(x):
    """平均秩(处理并列), 用于 Spearman。"""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), dtype=float)
    r[order] = np.arange(1, len(x) + 1)
    xs = x[order]
    i, n = 0, len(xs)
    while i < n:
        j = i
        while j + 1 < n and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return r


def _spearman(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < MIN_POOL:
        return None
    ra, rb = _rank(a[ok]), _rank(b[ok])
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else None


def _tstat(v):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], dtype=float)
    if len(v) < 5:
        return None
    s = v.std(ddof=1)
    return float(v.mean() / (s / np.sqrt(len(v)))) if s > 0 else None


# ----------------------------------------------------------------------
def _mv(df):
    """流通市值代理: 成交额 / 换手率。"""
    return np.log10((df["amount"] * 100 / df["turnover_rate"].replace(0, np.nan)).replace(0, np.nan))


FACTORS = [
    ("rev_1m", lambda d: -d["mom20"], +1),            # 一个月反转(买输家)
    ("mom_1m", lambda d: d["mom20"], +1),             # 一个月动量(买赢家)
    ("low_ivol", lambda d: -d["ivol20"], +1),         # 低特质波动
    ("low_turn", lambda d: -d["turnover_rate"], +1),  # 低换手
    ("low_amp", lambda d: -d["amp_pct"], +1),         # 低振幅
    ("near_high", lambda d: d["dist_high20"], +1),    # 逼近20日新高
    ("size_small", lambda d: _mv(d), -1),             # 小市值
    ("illiq", lambda d: np.log10((d["amp_pct"] * 1e8 / d["amount"].replace(0, np.nan)).replace(0, np.nan)), +1),
]


def _universe(day, cfg):
    boards = set(cfg["universe"]["boards"])
    m = ((day["is_paused"] == 0) & (day["is_st"] == 0) & (day["bars"] >= 250)
         & (day["amount"] >= AMT_MIN))
    df = day.loc[m]
    return df.loc[df["ts_code"].map(lambda c: board_of(c) in boards)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=1500)
    ap.add_argument("--step", type=int, default=5)
    ap.add_argument("--out", default="output/exp_horizon.json")
    args = ap.parse_args()

    cfg = load_config()
    print("[1/3] 构建上下文 %d 日 ..." % args.days, flush=True)
    t0 = time.time()
    ctx = prepare_context(cfg, args.days)
    by_date, main_dates = ctx["by_date"], ctx["main_dates"]
    print("      ctx ready %.0fs, %d 个主窗口交易日" % (time.time() - t0, len(main_dates)), flush=True)

    # ---- 价格矩阵: 行=股票, 列=交易日 ----
    full_dates = sorted(by_date)
    codes = sorted({c for d in full_dates for c in by_date[d]["ts_code"]})
    cpos = {c: i for i, c in enumerate(codes)}
    n_c, n_d = len(codes), len(full_dates)
    print("[2/3] 价格矩阵 %d 票 x %d 日 ..." % (n_c, n_d), flush=True)
    t1 = time.time()
    close_m = np.full((n_c, n_d), np.nan, dtype=np.float32)
    open_m = np.full((n_c, n_d), np.nan, dtype=np.float32)
    hlim_m = np.full((n_c, n_d), np.nan, dtype=np.float32)
    dpos = {d: j for j, d in enumerate(full_dates)}
    for j, d in enumerate(full_dates):
        df = by_date[d]
        idx = df["ts_code"].map(cpos).to_numpy()
        close_m[idx, j] = df["close_adj"].to_numpy()
        open_m[idx, j] = df["open_adj"].to_numpy()
        hlim_m[idx, j] = df["high_limit"].to_numpy()
    print("      矩阵 ready %.0fs" % (time.time() - t1), flush=True)

    # ---- 逐截面算 IC ----
    max_k = max(KS)
    j_start = dpos[main_dates[0]]
    j_end = n_d - max_k - 1
    spots = list(range(j_start, j_end, args.step))
    print("[3/3] 截面 %d 个 (每 %d 日), 持有期 %s ..." % (len(spots), args.step, KS), flush=True)

    ics = {name: {k: [] for k in KS} for name, _, _ in FACTORS}
    navs = {k: [] for k in KS}      # 全池等权前瞻收益(诊断基准)
    # 因子间截面相关性: 判断"低波/低换手/低振幅/反转"是同一个因子还是四条独立的路
    nf = len(FACTORS)
    corr_sum = np.zeros((nf, nf))
    corr_n = 0
    done = 0
    t2 = time.time()
    for j in spots:
        d = full_dates[j]
        day = _universe(by_date[d], cfg)
        if len(day) < MIN_POOL:
            continue
        ci = day["ts_code"].map(cpos).to_numpy()

        # T+1 可买入: 有开盘价、非一字涨停、非停牌
        e_open = open_m[ci, j + 1]
        e_hl = hlim_m[ci, j + 1]
        buyable = np.isfinite(e_open) & (e_open > 0) & ~(e_open >= e_hl - 1e-9)
        if buyable.sum() < MIN_POOL:
            continue

        # 按每个 k 算前瞻收益
        rets = {}
        for k in KS:
            x_close = close_m[ci, j + 1 + k]
            r = np.full(len(ci), np.nan)
            ok = buyable & np.isfinite(x_close) & (x_close > 0)
            r[ok] = (x_close[ok] / e_open[ok] - 1) * 100
            rets[k] = r
            v = r[np.isfinite(r)]
            if len(v) >= MIN_POOL:
                navs[k].append(v.mean())

        for name, fn, direction in FACTORS:
            try:
                fv = np.asarray(fn(day), dtype=float) * direction
            except Exception:  # noqa: BLE001
                continue
            for k in KS:
                # 只在"可买 + 有退出价 + 因子有效"的交集上算 IC
                ic = _spearman(fv, rets[k])
                if ic is not None:
                    ics[name][k].append(ic)

        # 因子间相关(截面平均): 用 rank 后算 Pearson = Spearman
        try:
            FM = np.column_stack([np.asarray(fn(day), dtype=float) * dr
                                  for _, fn, dr in FACTORS])
            okr = np.isfinite(FM).all(axis=1)
            if okr.sum() >= MIN_POOL:
                RM = np.column_stack([_rank(FM[okr, i]) for i in range(nf)])
                corr_sum += np.corrcoef(RM, rowvar=False)
                corr_n += 1
        except Exception:  # noqa: BLE001
            pass
        done += 1
        if done % 50 == 0:
            print("      %d/%d 个截面, %.0fs" % (done, len(spots), time.time() - t2), flush=True)

    # ---- 汇总 ----
    out = {"days": args.days, "step": args.step, "n_spot": done,
           "ks": KS, "ic": {}, "base": {}, "verdict": {}}
    print("\n%-11s %s" % ("因子", "  ".join("k=%-3d" % k for k in KS)), flush=True)
    for name, _, _ in FACTORS:
        row = {}
        cells = []
        for k in KS:
            v = ics[name][k]
            m = float(np.mean(v)) if v else None
            row[k] = {"n": len(v), "ic": round(m, 4) if m is not None else None,
                      "t": round(_tstat(v), 2) if v and _tstat(v) is not None else None}
            cells.append(("%+.3f" % m) if m is not None else "  -  ")
        out["ic"][name] = row
        print("%-11s %s" % (name, "  ".join(cells)), flush=True)

    for k in KS:
        v = navs[k]
        out["base"][k] = {"n": len(v), "avg_pct": round(float(np.mean(v)), 3) if v else None}

    # 判据
    for name, _, _ in FACTORS:
        row = out["ic"][name]
        short = [row[k]["ic"] for k in (1, 2, 3) if row[k]["ic"] is not None]
        long = [row[k]["ic"] for k in (20, 60, 120) if row[k]["ic"] is not None]
        if not short or not long:
            continue
        s, l = float(np.mean(short)), float(np.mean(long))
        flip = "慢alpha" if (s < 0 < l) else ("快alpha" if (s > 0 > l) else
                                             ("全程正" if s > 0 and l > 0 else "全程负"))
        out["verdict"][name] = {"short_ic": round(s, 4), "long_ic": round(l, 4), "type": flip}

    print("\n=== 短(k=1~3) vs 长(k=20~120) ===", flush=True)
    for name, v in out["verdict"].items():
        print("  %-11s 短 %+.4f -> 长 %+.4f   %s" % (name, v["short_ic"], v["long_ic"], v["type"]), flush=True)

    # 因子间平均相关矩阵
    if corr_n:
        C = corr_sum / corr_n
        names = [n for n, _, _ in FACTORS]
        out["corr"] = {"n_spot": corr_n,
                       "matrix": {names[i]: {names[j]: round(float(C[i, j]), 3) for j in range(nf)}
                                  for i in range(nf)}}
        print("\n=== 因子间截面秩相关(均值, n=%d 截面) ===" % corr_n, flush=True)
        print("%-11s %s" % ("", " ".join("%6s" % n[:6] for n in names)), flush=True)
        for i, n in enumerate(names):
            print("%-11s %s" % (n, " ".join("%6.2f" % C[i, j] for j in range(nf))), flush=True)
        # 同向正 IC 因子之间的平均相关(剔除对角)
        pos = [i for i, n in enumerate(names)
               if out["verdict"].get(n, {}).get("short_ic", 0) > 0
               and out["verdict"].get(n, {}).get("long_ic", 0) > 0]
        if len(pos) > 1:
            vals = [C[i, j] for i in pos for j in pos if i < j]
            out["corr"]["pos_group"] = [names[i] for i in pos]
            out["corr"]["pos_avg"] = round(float(np.mean(vals)), 3)
            print("  同向正 IC 因子的组内平均相关: %.2f  (%s)"
                  % (np.mean(vals), ", ".join(names[i] for i in pos)), flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n结果已写 %s" % args.out, flush=True)


if __name__ == "__main__":
    main()
