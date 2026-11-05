# -*- coding: utf-8 -*-
"""独立副策略候选池: 六种"不同形态"的月度策略同台比较(2026-11-05)。

为什么不是再筛一遍因子
----------------------
第一百三十五轮筛的是"横截面排序因子"，第一百三十六轮证明它们同属一个族
（交易强度，组内相关 0.46）。再筛因子只是在同一个族里换切面。
本条要的是**另一本书**：不同的 alpha 来源、不同的持有期、不同的资金形态。

六个候选(全部月度调仓、等权 20 只、扣 0.30% 往返、次日开盘成交)
----------------------------------------------------------------
  1. tsmom_12_1   12-1 时序动量: 250 日收益剔除最近 20 日, 取前 20
                  (Jegadeesh-Titman 经典定义; 本仓此前从未测过 —— 第一百三十六轮
                   的 mom_1m 只是 mom20, 与 rev_1m 完全共线, 不等于时序动量)
  2. tsmom_abs    1 + 绝对动量: 只留自身 12-1 收益 > 0 的票
  3. tsmom_ma     1 + 市场择时: 等权指数在 120 日线下方时整月空仓
  4. lowvol_ma    低波(ivol20 最低 20 只) + 市场择时
  5. hi52w        52 周新高: 距 250 日最高价最近的 20 只(锚定效应)
  6. dualma       双均线: close > MA60 > MA200, 按 12-1 动量取前 20

判据(建设导向, 与前面"证伪导向"不同)
------------------------------------
  · 年化 > 8%(独立成书的下限, 低于此不值得占用注意力);
  · 最大回撤 < 30%;
  · 与 dip 月收益 |rho| < 0.30(否则是同一笔风险的两种表达);
  · 三条全过 = 可建; 只过两条 = 降级为观察仓。

口径
----
  · 信号日 = 每月最后一个交易日收盘; 执行 = 次一交易日**开盘**买入;
  · 卖出 = 下一个月调仓日的次一交易日开盘;
  · 成本 = 换手率 × 0.30%(滑点千1×2 + 佣金万2.5×2 + 印花税万5);
  · 股票池同主策略: 非停牌/非ST/上市满250日/成交额>=1亿/板块在内;
  · 所有指标仅用 T 日及之前数据。

用法
----
  python3 tools/exp_satellite.py                # 1500 日
  python3 tools/exp_satellite.py --days 3378    # 全史
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
TOP_N = 20
AMT_MIN = 1e8
MIN_POOL = 200
MKT_MA = 120          # 市场择时均线(半年线): warmup 160 日足够, 无需 200 日


# ----------------------------------------------------------------------
def _month_pairs(main_dates):
    """每月最后一个交易日为信号日 -> (signal, exec, next_exec)。"""
    pairs = []
    for i in range(len(main_dates) - 1):
        if main_dates[i][:6] != main_dates[i + 1][:6]:
            pairs.append((main_dates[i], main_dates[i + 1]))
    triples = []
    for k in range(len(pairs) - 1):
        triples.append((pairs[k][0], pairs[k][1], pairs[k + 1][1]))
    return triples


def _nav(rets):
    r = np.asarray(rets, dtype=float)
    if len(r) == 0:
        return {}
    nav = np.cumprod(1 + r)
    years = len(r) / 12.0
    ann = float(nav[-1] ** (1 / years) - 1) if nav[-1] > 0 else None
    peak = np.maximum.accumulate(nav)
    mdd = float((nav / peak - 1).min())
    return {"ann": round(ann * 100, 2) if ann is not None else None,
            "mdd": round(mdd * 100, 2),
            "calmar": round(ann / abs(mdd), 2) if ann and mdd < 0 else None,
            "win": round(100 * float((r > 0).mean()), 1),
            "months": len(r), "nav": round(float(nav[-1]), 3)}


def _universe(day, cfg):
    boards = set(cfg["universe"]["boards"])
    m = ((day["is_paused"] == 0) & (day["is_st"] == 0) & (day["bars"] >= 250)
         & (day["amount"] >= AMT_MIN))
    df = day.loc[m]
    return df.loc[df["ts_code"].map(lambda c: board_of(c) in boards)]


def _top(codes, vals, n, asc=False):
    """按 vals 取前 n 个 code; asc=True 取最小。"""
    v = np.asarray(vals, dtype=float)
    ok = np.isfinite(v)
    if ok.sum() < n:
        return []
    c = np.asarray(codes)[ok]
    vv = v[ok]
    order = np.argsort(vv if asc else -vv)
    return [str(x) for x in c[order[:n]]]


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=1500)
    ap.add_argument("--out", default="output/exp_satellite.json")
    args = ap.parse_args()

    cfg = load_config()
    print("[1/4] 构建上下文 %d 日 ..." % args.days, flush=True)
    t0 = time.time()
    ctx = prepare_context(cfg, args.days)
    by_date, main_dates, reg = ctx["by_date"], ctx["main_dates"], ctx["reg"]
    print("      ctx ready %.0fs, %d 日" % (time.time() - t0, len(main_dates)), flush=True)

    full_dates = sorted(by_date)
    dpos = {d: j for j, d in enumerate(full_dates)}
    codes = sorted({c for d in full_dates for c in by_date[d]["ts_code"]})
    cpos = {c: i for i, c in enumerate(codes)}
    n_c, n_d = len(codes), len(full_dates)
    print("[2/4] 价格矩阵 %d 票 x %d 日 ..." % (n_c, n_d), flush=True)

    close_m = np.full((n_c, n_d), np.nan, dtype=np.float32)
    open_m = np.full((n_c, n_d), np.nan, dtype=np.float32)
    for j, d in enumerate(full_dates):
        df = by_date[d]
        idx = df["ts_code"].map(cpos).to_numpy()
        close_m[idx, j] = df["close_adj"].to_numpy()
        open_m[idx, j] = df["open_adj"].to_numpy()

    # 市场择时序列(仅用 T 日及之前)
    mkt = reg["mkt_idx"].astype(float)
    mkt_ma = mkt.rolling(MKT_MA).mean()
    risk_on = (mkt >= mkt_ma)

    triples = _month_pairs(main_dates)
    print("[3/4] 月度调仓 %d 期, 6 个候选 ..." % len(triples), flush=True)

    NAMES = ["tsmom_12_1", "tsmom_abs", "tsmom_ma", "lowvol_ma", "hi52w", "dualma",
             "lowvol", "lowvol_rp", "lowvol_rev"]
    rets = {n: [] for n in NAMES}
    hold = {n: [] for n in NAMES}
    months, base_rets, mkt_state = [], [], []
    prev = {n: None for n in NAMES}
    turn_acc = {n: 0.0 for n in NAMES}
    n_skip = 0

    t1 = time.time()
    for si, (sig_d, exe_d, next_d) in enumerate(triples):
        j = dpos[sig_d]
        if j < 250 or (j + 1) >= n_d:
            n_skip += 1
            continue
        day = _universe(by_date[sig_d], cfg)
        if len(day) < MIN_POOL:
            n_skip += 1
            continue
        ci = day["ts_code"].map(cpos).to_numpy()
        cc = np.asarray(day["ts_code"], dtype=object)

        # ---- T 日可见指标 ----
        px_now = close_m[ci, j]
        px_20 = close_m[ci, j - 20]
        px_250 = close_m[ci, j - 250]
        mom12_1 = px_20 / px_250 - 1                          # 250日剔除最近20日
        ma60 = np.nanmean(close_m[ci, j - 59:j + 1], axis=1)
        ma200 = np.nanmean(close_m[ci, j - 199:j + 1], axis=1)
        hi250 = np.nanmax(close_m[ci, j - 249:j + 1], axis=1)
        dist_hi250 = px_now / hi250 - 1
        ivol = np.asarray(day["ivol20"], dtype=float)
        tj = dpos[exe_d]
        o_buy = open_m[ci, tj]
        x_px = open_m[ci, dpos[next_d]]
        buyable = np.isfinite(o_buy) & (o_buy > 0) & np.isfinite(x_px) & (x_px > 0)
        fwd = np.where(buyable, (x_px / o_buy - 1) * 100, np.nan)

        # 基准: 全池等权
        b = float(np.nanmean(fwd))
        base_rets.append(b / 100)
        months.append(exe_d[:6])
        on = bool(risk_on.get(sig_d, False)) if sig_d in risk_on.index else True
        mkt_state.append(1 if on else 0)

        # ---- 六个候选的持仓 ----
        amap_pre = {str(c): k for k, c in enumerate(cc)}
        picks = {}
        valid = np.isfinite(fwd)

        m = np.where(valid, mom12_1, np.nan)
        picks["tsmom_12_1"] = (_top(cc, m, TOP_N), None)

        m_abs = np.where(valid & (mom12_1 > 0), mom12_1, np.nan)
        picks["tsmom_abs"] = (_top(cc, m_abs, TOP_N), None)

        picks["tsmom_ma"] = ((_top(cc, m, TOP_N) if on else []), None)

        lv = np.where(valid, -ivol, np.nan)
        picks["lowvol_ma"] = ((_top(cc, lv, TOP_N) if on else []), None)

        picks["hi52w"] = (_top(cc, np.where(valid, dist_hi250, np.nan), TOP_N), None)

        ma_ok = (px_now > ma60) & (ma60 > ma200)
        picks["dualma"] = (_top(cc, np.where(valid & ma_ok, mom12_1, np.nan), TOP_N), None)

        # --- 低波三个变体: 纯等权 / 风险平价 / 低波池内做反转 ---
        lv_codes = _top(cc, lv, TOP_N)
        picks["lowvol"] = (lv_codes, None)
        # 风险平价: 权重 ∝ 1/ivol(低波票给更高权重), 仍在低波池内
        w_rp = {}
        for c in lv_codes:
            kk = amap_pre.get(c)
            if kk is not None and np.isfinite(ivol[kk]) and ivol[kk] > 0:
                w_rp[c] = 1.0 / ivol[kk]
        picks["lowvol_rp"] = (lv_codes, [w_rp.get(c, 0.0) for c in lv_codes])
        # 低波池内做反转: 先取 ivol 最低 40%, 再按 -mom20 排序取 20
        n_wide = max(TOP_N * 4, int(len(day) * 0.4))
        wide = _top(cc, lv, n_wide)
        wide_set = set(wide)
        mom20_arr = np.asarray(day["mom20"], dtype=float)
        rev_in_lv = np.where(valid & np.array([str(c) in wide_set for c in cc]), -mom20_arr, np.nan)
        picks["lowvol_rev"] = (_top(cc, rev_in_lv, TOP_N), None)

        amap = {str(c): k for k, c in enumerate(cc)}
        for nm in NAMES:
            cs, ws = picks[nm]
            if not cs:
                rets[nm].append(0.0)
                hold[nm].append(0)
                continue
            pairs_k = [(amap[c], ws[i] if ws is not None else 1.0)
                       for i, c in enumerate(cs) if c in amap]
            if not pairs_k:
                rets[nm].append(0.0)
                hold[nm].append(0)
                continue
            kk = [x[0] for x in pairs_k]
            ww = np.asarray([x[1] for x in pairs_k], dtype=float)
            v = fwd[kk]
            ok = np.isfinite(v) & np.isfinite(ww) & (ww > 0)
            if ok.sum() == 0:
                rets[nm].append(0.0)
                hold[nm].append(0)
                continue
            cur = set(cs)
            if prev[nm] is None:
                turn = 1.0
            else:
                turn = 1.0 - len(cur & prev[nm]) / max(1, len(cur))
            prev[nm] = cur
            turn_acc[nm] += turn
            r = float(np.sum(v[ok] * ww[ok]) / np.sum(ww[ok]))
            rets[nm].append(r / 100 - turn * COST)
            hold[nm].append(int(ok.sum()))
        if (si + 1) % 24 == 0:
            print("      %d/%d 期, %.0fs" % (si + 1, len(triples), time.time() - t1), flush=True)

    print("[4/4] 汇总 ...", flush=True)
    out = {"days": args.days, "n_months": len(months), "top_n": TOP_N,
           "cost_pct": COST * 100, "mkt_ma": MKT_MA,
           "months": months, "mkt_on_pct": round(100 * sum(mkt_state) / max(1, len(mkt_state)), 1),
           "benchmark": _nav(base_rets), "series_base": [round(x * 100, 3) for x in base_rets],
           "strategies": {}, "series": {}}

    # dip 月度(用于相关性)
    dip_m = {}
    try:
        bt = json.loads((ROOT / "output" / "backtest.json").read_text(encoding="utf-8"))
        mo = bt.get("monthly")
        if isinstance(mo, list):
            for x in mo:
                if isinstance(x, dict):
                    k = str(x.get("month") or x.get("ym") or "")[:6]
                    v = x.get("ret_pct", x.get("ret"))
                    if k and v is not None:
                        dip_m[k] = float(v) / 100
        elif isinstance(mo, dict):
            dip_m = {str(k)[:6]: float(v) / 100 for k, v in mo.items()}
    except Exception as e:  # noqa: BLE001
        print("  (未读到 backtest.json: %s)" % e, flush=True)

    print("\n=== 独立副策略候选 (%d 期, 等权 %d 只, 月度, 扣 %.2f%%) ==="
          % (len(months), TOP_N, COST * 100), flush=True)
    print("  %-11s %8s %8s %8s %6s %6s %8s %8s" % (
        "策略", "年化", "回撤", "卡玛", "月胜率", "持仓", "换手", "与dip rho"), flush=True)
    bst = out["benchmark"]
    print("  %-11s %+7.1f%% %+7.1f%% %7.2f %5.1f%% %6s %8s %8s" % (
        "[基准]", bst["ann"], bst["mdd"], bst["calmar"] or 0, bst["win"], "-", "-", "-"), flush=True)

    for nm in NAMES:
        st = _nav(rets[nm])
        rho = None
        if dip_m:
            xs, ys = [], []
            for k, r in zip(months, rets[nm]):
                if k in dip_m:
                    xs.append(r)
                    ys.append(dip_m[k])
            if len(xs) >= 12:
                rho = float(np.corrcoef(xs, ys)[0, 1])
        st["turnover"] = round(turn_acc[nm] / max(1, len([x for x in hold[nm] if x > 0])), 3)
        st["avg_hold"] = round(float(np.mean([x for x in hold[nm] if x > 0]) or 0), 1)
        st["corr_dip"] = round(rho, 3) if rho is not None else None
        st["ann_excess"] = round((st["ann"] or 0) - (bst["ann"] or 0), 2)
        out["strategies"][nm] = st
        out["series"][nm] = [round(x * 100, 3) for x in rets[nm]]
        print("  %-11s %+7.1f%% %+7.1f%% %7.2f %5.1f%% %6.1f %8.2f %+9.3f" % (
            nm, st["ann"] or 0, st["mdd"], st["calmar"] or 0, st["win"],
            st["avg_hold"], st["turnover"], rho if rho is not None else 0), flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print("\n结果已写 %s" % args.out, flush=True)


if __name__ == "__main__":
    main()
