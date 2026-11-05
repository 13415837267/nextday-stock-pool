# -*- coding: utf-8 -*-
"""副策略"每日选 top1"可行性扫描(2026-11-05, 第一百三十八轮)。

问题
----
第一百三十七轮上线的副策略是**月度调仓、等权 20 只**。用户要求"每日选 top1"。
这两件事的差别不在选股, 在**换手**:
  · 每天持 1 只、天天换 → 日换手 100%, 成本 0.3%×250 = 75%/年, 必死;
  · 每天**新增** 1 只、滚动持有 H 日 → 稳态持仓 H 只, 日换手 1/H,
    H=20 时年换手 12.5 倍、成本 3.75%/年, 这是唯一能活下来的"每日 top1"。

本脚本测的就是后者: 每日 1 个批次, 持有 H 个交易日, 组合是过去 H 个批次的等权平均。
H=1 就是"每天换一只", 作为对照组一起打印。

口径
----
  · 信号 = T 日收盘后的 ivol20 横截面排名(最低者 top1);
  · 建仓/结算 = T 日收盘价(与卫星账本 satellite_run.py 完全一致);
  · 批次 i 在 T_i 收盘建仓, T_{i+H} 收盘结算, 持有期恰好 H 个交易日;
  · 组合日收益 = 在持批次的等权平均; 成本 = 每日 0.30%/H(往返 0.3% 摊到 H 天);
  · 股票池与主策略/卫星一致: 非停牌/非ST/上市满250日/成交额≥1亿/板块在内;
  · 并行对照: 每日 topK(K=3/5)把资金拆成 K 份, 等价于"每天选 K 只"。

用法
----
  python3 tools/exp_satellite_daily.py --days 300      # 冒烟
  python3 tools/exp_satellite_daily.py --days 1500     # 主窗口
  python3 tools/exp_satellite_daily.py --days 3378     # 全史
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
HOLDS = [1, 2, 3, 5, 10, 20, 30, 40, 60, 120]
TOPS = [1, 3, 5]


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
            "sharpe": round(sharpe, 2), "days": len(r),
            "nav": round(float(nav[-1]), 3),
            "win": round(100 * float((r > 0).mean()), 1)}


def _universe(day, cfg):
    boards = set(cfg["universe"]["boards"])
    m = ((day["is_paused"] == 0) & (day["is_st"] == 0) & (day["bars"] >= 250)
         & (day["amount"] >= AMT_MIN))
    df = day.loc[m]
    return df.loc[df["ts_code"].map(lambda c: board_of(c) in boards)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=1500)
    ap.add_argument("--out", default="output/exp_satellite_daily.json")
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
    n_c, n_d = len(codes), len(dates)
    print("[2/4] 收盘价矩阵 %d 票 x %d 日 ..." % (n_c, n_d), flush=True)

    close_m = np.full((n_c, n_d), np.nan, dtype=np.float32)
    for j, d in enumerate(dates):
        df = by_date[d]
        close_m[df["ts_code"].map(cpos).to_numpy(), j] = df["close_adj"].to_numpy()

    # 日收益(收盘到收盘), 只保留两个价格都有效的
    ret_m = close_m[:, 1:] / close_m[:, :-1] - 1.0     # 列 j 对应 dates[j+1]
    ret_m = np.where(np.isfinite(ret_m), ret_m, np.nan).astype(np.float32)

    # 每日候选: 只算主窗口内(信号日), 按 ivol 升序
    print("[3/4] 逐日排序 top1 ...", flush=True)
    picks = {}          # date -> [code order top5]
    pool_sz = {}
    t1 = time.time()
    for k, d in enumerate(main_dates):
        j = dpos[d]
        if j < 1 or (j + 1) >= n_d:
            continue
        day = _universe(by_date[d], cfg)
        if len(day) < MIN_POOL:
            continue
        v = pd.to_numeric(day["ivol20"], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(v)
        if ok.sum() < 25:
            continue
        cc = np.asarray(day["ts_code"], dtype=object)[ok]
        vv = v[ok]
        order = np.argsort(vv, kind="mergesort")[:5]
        picks[d] = [str(cc[i]) for i in order]
        pool_sz[d] = int(ok.sum())
        if (k + 1) % 250 == 0:
            print("      %d/%d 日, %.0fs" % (k + 1, len(main_dates), time.time() - t1), flush=True)

    sig_dates = [d for d in main_dates if d in picks]
    print("      有效信号日 %d, 平均股票池 %.0f" % (len(sig_dates), np.mean(list(pool_sz.values()))), flush=True)

    # 收益日期序列: 信号日 T 的收益从 T+1 开始记(收盘建仓)
    # 用 ret_m[:, j] 表示 dates[j] -> dates[j+1] 的收益
    res = {"days": args.days, "cost_pct": COST * 100, "holds": HOLDS, "tops": TOPS,
           "strategies": {}, "series": {}}

    print("[4/4] 组合回放 ...", flush=True)
    for topk in TOPS:
        for H in HOLDS:
            nav = 1.0
            rets = []
            held = []            # [(code, expire_idx)]
            for si, d in enumerate(sig_dates):
                j = dpos[d]
                # 清理到期批次(持有 H 日: 在 T+H 收盘结算, 这里用 j 索引判断)
                held = [(c, exp) for (c, exp) in held if exp > j]
                # 建仓 topk 批(每批资金 1/(H*topk))
                for c in picks[d][:topk]:
                    held.append((c, j + H))
                # 当日在持批次收益: 从 j 到 j+1
                if j >= len(dates) - 1:
                    break
                rs = []
                for (c, exp) in held:
                    ci = cpos.get(c)
                    if ci is None:
                        continue
                    r = ret_m[ci, j]
                    if np.isfinite(r):
                        rs.append(float(r))
                if not rs:
                    rets.append(0.0)
                    continue
                # 稳态日换手 = topk/(H*topk) = 1/H
                r_port = float(np.mean(rs)) - COST / H
                nav *= (1 + r_port)
                rets.append(r_port)
            st = _stats(rets)
            st["H"] = H
            st["topk"] = topk
            st["avg_hold_n"] = round(float(np.mean([min(H, len(sig_dates))])), 1)
            key = "top%d_H%d" % (topk, H)
            res["strategies"][key] = st
            res["series"][key] = [round(x * 100, 4) for x in rets]

    # 基准: 全池等权日收益
    bench = []
    for d in sig_dates[:-1]:
        j = dpos[d]
        day = _universe(by_date[d], cfg)
        ci = day["ts_code"].map(cpos).to_numpy()
        r = ret_m[ci, j]
        r = r[np.isfinite(r)]
        bench.append(float(np.mean(r)) if len(r) else 0.0)
    res["benchmark"] = _stats(bench)

    print("\n=== 每日 topK 扫描 (%d 日, 扣 %.2f%%/次) ===" % (len(sig_dates), COST * 100), flush=True)
    print("  %-10s %8s %8s %8s %7s %7s" % ("组合", "年化", "回撤", "卡玛", "夏普", "日胜率"), flush=True)
    b = res["benchmark"]
    print("  %-10s %+7.1f%% %+7.1f%% %7.2f %7.2f %6.1f%%" % (
        "[基准]", b["ann"], b["mdd"], b["calmar"] or 0, b["sharpe"], b["win"]), flush=True)
    for topk in TOPS:
        for H in HOLDS:
            st = res["strategies"]["top%d_H%d" % (topk, H)]
            print("  %-10s %+7.1f%% %+7.1f%% %7.2f %7.2f %6.1f%%" % (
                "top%d H=%d" % (topk, H), st["ann"], st["mdd"], st["calmar"] or 0,
                st["sharpe"], st["win"]), flush=True)
        print("", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print("结果已写 %s" % out, flush=True)


if __name__ == "__main__":
    main()
