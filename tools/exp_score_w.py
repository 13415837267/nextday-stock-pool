"""第一百七十八轮: 评分权重重估 —— 用 IC 显著的因子替换零信息权重。

上一实验(exp_factor_ic)的结论:
  · 现行 dip 评分横截面 IC ≈ 0 (t=+0.15), 第 1 名甚至不如全池均值
  · 在可买样本(gap<=0)里, 唯二显著的是 pct(+0.0349, t=+3.31) 与 ivol20(-0.0279, t=-2.60)
  · low_tail 在**全池**是显著负的(t=-2.24), 现行权重却给了 +0.03
  · 占权重 70% 的 neg_bias20 是零信息(t=-0.26)

本脚本用**轻量模拟**扫评分权重, 口径尽量贴回测主干:
  每日取评分第 1 名 -> T+1 开盘缺口需达标(正常日 <=0 / 风控日 <=-2.5)
  -> T+1 开买 T+2 收卖, 扣同口径成本 -> 累乘成净值。
  不做资金/并发模拟(exec_top_n=1, max_buy_per_day=1, 最多 2 只并发, 简化影响很小),
  目的只是**排序比较**, 不作最终绩效口径 —— 入选者再跑完整回测复核。

用法: python tools/exp_score_w.py [--days 1500] [--grid quick|full]
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

from src.backtest import prepare_context   # noqa: E402
from src.config import load_config        # noqa: E402
from src.selector import _zs, select_stocks  # noqa: E402

COST = (2.5 * 2 + 5 + 10) / 1e4

# 候选权重组合(只动 dip_weights, 不动筛选阈值)
GRID = {
    "base":      {"mom20": 0.2, "neg_bias20": 0.7, "neg_amount": 0.05, "neg_vol_ratio": 0.02, "low_tail": 0.03},
    "pct_only":  {"pct": 1.0},
    "pct_ivol":  {"pct": 0.6, "neg_ivol": 0.4},
    "pct_ivol_bias": {"pct": 0.5, "neg_ivol": 0.3, "neg_bias20": 0.2},
    "pct_ivol_amt":  {"pct": 0.5, "neg_ivol": 0.3, "neg_amount": 0.2},
    "pct_ivol_lt_neg": {"pct": 0.5, "neg_ivol": 0.3, "neg_low_tail": 0.2},
    "pic":       {"pct": 0.4, "neg_ivol": 0.3, "neg_bias20": 0.2, "neg_amount": 0.1},
    "pct_mom":   {"pct": 0.6, "mom20": 0.4},
    "neg_ivol":  {"neg_ivol": 1.0},
    "pct_turn":  {"pct": 0.6, "neg_turnover": 0.4},
}


def build_score(df: pd.DataFrame, w: dict) -> pd.Series:
    """按权重组合重构评分(与 selector 同源, 用 _zs 标准化)。"""
    s = pd.Series(0.0, index=df.index)
    for k, v in w.items():
        if k == "pct":
            s = s + v * _zs(df["pct"])
        elif k == "mom20":
            s = s + v * _zs(df["mom20"])
        elif k == "neg_bias20":
            s = s + v * _zs(-df["bias20"])
        elif k == "neg_amount":
            s = s + v * _zs(-np.log1p(df["amount"]))
        elif k == "neg_vol_ratio":
            s = s + v * _zs(-df["vol_ratio"])
        elif k == "low_tail":
            s = s + v * _zs(df["low_tail"])
        elif k == "neg_low_tail":
            s = s + v * _zs(-df["low_tail"])
        elif k == "neg_ivol":
            s = s + v * _zs(-df["ivol20"])
        elif k == "neg_turnover":
            s = s + v * _zs(-df["turnover_rate"])
        elif k == "close_pos":
            s = s + v * _zs(df["close_pos"])
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=1500)
    ap.add_argument("--out", default="output/exp_score_w.json")
    a = ap.parse_args()

    cfg = load_config()
    ctx = prepare_context(cfg, a.days)
    by_date, dates = ctx["by_date"], ctx["main_dates"]
    reg = ctx["reg"]
    pos = {d: i for i, d in enumerate(dates)}
    ex = cfg.get("execution") or {}
    fl_norm = float(ex.get("buy_gap_below", 0))
    fl_risk = float(ex.get("block_gap_below", -2.5))

    # 先缓存每个信号日的候选池(筛选结果与评分权重无关, 只算一次)
    cache: list[dict] = []
    for d in dates:
        i = pos[d]
        if i + 2 >= len(dates):
            continue
        picks, _ = select_stocks(d, by_date[d], cfg, None, top_n=300, name_filter=False)
        picks = [p for p in picks if p.get("pool", "main") == "main"]
        if not picks:
            continue
        codes = [p["code"] for p in picks]
        sub = by_date[d]
        sub = sub[sub["ts_code"].isin(codes)]
        if sub.empty:
            continue
        cache.append({"d": d, "i": i, "rows": sub, "codes": set(codes)})
    print(f"候选日 {len(cache)}")

    results = {}
    for tag, w in GRID.items():
        nav = 1.0
        navs = [1.0]
        n_trade = 0
        rets = []
        for c in cache:
            d, i, rows = c["d"], c["i"], c["rows"]
            sc = build_score(rows, w)
            if sc.isna().all():
                navs.append(nav)
                continue
            top = rows.loc[sc.idxmax()]
            code = top["ts_code"]
            bd, sd = dates[i + 1], dates[i + 2]
            dfb, dfs = by_date[bd], by_date[sd]
            mb = dfb[dfb["ts_code"] == code]
            ms = dfs[dfs["ts_code"] == code]
            if mb.empty or ms.empty:
                navs.append(nav)
                continue
            rb, rs = mb.iloc[0], ms.iloc[0]
            o, cc = float(rb["open_adj"]), float(rs["close_adj"])
            pc = float(rb["prev_close_adj"])
            if o <= 0 or cc <= 0 or pc <= 0:
                navs.append(nav)
                continue
            o_raw = float(rb["open"])
            if o_raw >= float(rb["high_limit"]) - 1e-9 or o_raw <= float(rb["low_limit"]) + 1e-9:
                navs.append(nav)
                continue
            gap = (o / pc - 1.0) * 100.0
            ok = bool(reg.loc[d, "trade_ok"]) if d in reg.index else True
            floor = fl_norm if ok else fl_risk
            if gap > floor:
                navs.append(nav)
                continue
            ret = (cc / o - 1.0 - COST) * 100.0
            nav *= (1 + ret / 100.0)
            rets.append(ret)
            n_trade += 1
            navs.append(nav)

        years = len(navs) / 252.0
        cagr = (nav ** (1 / years) - 1) * 100 if nav > 0 else -100.0
        s = pd.Series(navs)
        mdd = float((s / s.cummax() - 1).min()) * 100
        arr = np.array(rets)
        results[tag] = {
            "年化%": round(cagr, 2), "回撤%": round(mdd, 2),
            "卡玛": round(cagr / abs(mdd), 2) if mdd else None,
            "笔数": n_trade, "单笔%": round(float(arr.mean()), 3) if len(arr) else None,
            "胜率%": round(float((arr > 0).mean() * 100), 1) if len(arr) else None,
            "weights": w,
        }
        print(f"{tag:<18} 年化 {results[tag]['年化%']:>8.2f}%  回撤 {mdd:>7.2f}%  "
              f"卡玛 {results[tag]['卡玛']:>6.2f}  笔 {n_trade:>4}  单笔 {results[tag]['单笔%']:>+6.3f}%  "
              f"胜率 {results[tag]['胜率%']}%")

    ranked = sorted(results.items(), key=lambda kv: -(kv[1]["卡玛"] or 0))
    print(f"\n卡玛排序: {' > '.join(k for k, _ in ranked[:5])}")

    p = ROOT / a.out
    p.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"写出 {p}")


if __name__ == "__main__":
    main()
