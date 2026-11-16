"""补位信号（第 240 轮产出）：大盘破位日，买“跌得最透”的那只

规则（2021-2026 六年 93 次，全部为正，卡玛 9.47 vs 基准 6.63）:
  1) 市场条件: 全市场等权指数距其 20 日均线 <= -3.51%
     —— 用当日全市场 pct 均值累积出的等权净值，与 features/market.csv 同一口径
  2) 个股条件: 收盘距前 60 日最高价 <= -24.46%（复权口径，前 60 日不含当日）
  3) 从命中票里按流通市值从小到大取 1 只（3 只版本收益略低、回撤略高）
  4) 次日(T+1)开盘买，T+2 收盘卖；只在主策略空仓日使用

用法:
  python3 tools/backfill_daily.py            # 看最新信号日 + 最近 250 日触发记录
  python3 tools/backfill_daily.py --days 480 # 回看更长
  python3 tools/backfill_daily.py --json     # 只输出最新一日 JSON（供 daily_run 调用）
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data" / "panel"
OUT = ROOT / "output"

IDX_DIST_MAX = -3.51      # 等权指数距 20 日线
DIST_HIGH60_MAX = -24.46  # 距 60 日前高
MIN_LISTED_BARS = 250


def load_panel(dates: list[str]) -> pd.DataFrame:
    frames = []
    cols = ["ts_code", "trade_date", "open", "high", "low", "close", "pre_close",
            "pct", "amount", "factor", "turnover_rate", "is_paused", "is_st"]
    for d in dates:
        f = PANEL / f"{d}.csv.gz"
        if not f.exists():
            continue
        try:
            frames.append(pd.read_csv(f, usecols=cols, dtype={"ts_code": "str"}))
        except Exception:      # noqa: BLE001
            continue
    if not frames:
        raise SystemExit("没有可用的面板缓存 data/panel/")
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=250)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    all_dates = sorted(p.stem.replace(".csv", "") for p in PANEL.glob("*.csv.gz"))
    use = all_dates[-(a.days + 90):]              # 多取 90 天做 60 日窗口预热
    d = load_panel(use)
    print(f"面板 {use[0]}~{use[-1]}  {len(d):,} 行")

    # ---- 市场: 等权指数距 20 日线（与 build_features_234 完全同口径） ----
    g = d.groupby("trade_date", sort=True)
    st = pd.DataFrame({"eq_ret": g["pct"].mean(), "med_pct": g["pct"].median()}).reset_index()
    st["eq_nav"] = (1 + st["eq_ret"] / 100).cumprod()
    st["ma20"] = st["eq_nav"].rolling(20, min_periods=10).mean()
    st["idx_dist"] = (st["eq_nav"] / st["ma20"] - 1) * 100
    st = st.set_index("trade_date")
    mkt = st.reindex(sorted(d["trade_date"].unique()))

    # ---- 个股: 复权价距前 60 日最高 ----
    d = d[(d["is_paused"] == 0) & (d["is_st"] == 0)]
    d = d[d["pct"].between(-11.5, 11.5)]
    # ★ 只留主板: 3 万本金开不了创业板(10万)/科创板(50万)/北交所(50万), 且北交所
    #   30% 涨跌幅会把"小市值弹性"放大成虚假收益 —— 240 轮就是栽在这(49 笔里 22 笔是 BJ)。
    d = d[d["ts_code"].str[:2].isin(["60", "00"])]
    d["adj"] = d["close"] * d["factor"]
    d["adj_h"] = d["high"] * d["factor"]
    d = d.sort_values(["ts_code", "trade_date"])
    d["h60"] = d.groupby("ts_code")["adj_h"].transform(
        lambda s: s.shift(1).rolling(60, min_periods=40).max())
    d["dist_high60"] = (d["adj"] / d["h60"] - 1) * 100
    d["bars"] = d.groupby("ts_code")["close"].transform("size")   # 窗口内有效根数
    # 流通市值(亿元): amount(千元)/换手率(%) -> 流通市值
    # amount 单位千元, turnover_rate 单位%: 流通市值(亿元) = amount/换手%/1000
    d["circ_mv"] = d["amount"] / d["turnover_rate"].replace(0, np.nan) / 1e6

    win = d[(d["bars"] >= 60) & d["dist_high60"].notna()].copy()   # 60 日高点已足够排除次新
    win["idx_dist"] = win["trade_date"].map(mkt["idx_dist"])
    trig = win[(win["idx_dist"] <= IDX_DIST_MAX) & (win["dist_high60"] <= DIST_HIGH60_MAX)]

    days = sorted(trig["trade_date"].unique())
    print(f"\n近 {len(use)-90} 个交易日里，触发 {len(days)} 次")

    rows = []
    for dt in days:
        s = trig[trig["trade_date"] == dt].sort_values("circ_mv")
        top = s.head(3)
        rows.append({"signal": dt, "idx_dist": round(float(s["idx_dist"].iloc[0]), 2),
                     "n": len(s),
                     "top1": f"{top['ts_code'].iloc[0]}  mv{top['circ_mv'].iloc[0]:.1f}亿 "
                             f"距高{top['dist_high60'].iloc[0]:.1f}%",
                     "top3": ",".join(top["ts_code"].tolist())})
    hist = pd.DataFrame(rows)
    if len(hist):
        print(hist.to_string(index=False))
        hist.to_csv(OUT / "backfill_history.csv", index=False, encoding="utf-8-sig")

    last = all_dates[-1]
    li = float(mkt["idx_dist"].get(int(last), np.nan))
    hit = trig[trig["trade_date"] == int(last)].sort_values("circ_mv")
    print(f"\n最新交易日 {last}: 等权指数距20日线 {li:+.2f}%  "
          f"{'✅ 触发' if li <= IDX_DIST_MAX else '❌ 未触发'}")
    if len(hit):
        print(hit.head(3)[["ts_code", "close", "pct", "dist_high60", "circ_mv"]].to_string(index=False))

    payload = {"signal_date": last, "idx_dist": round(li, 2),
               "triggered": bool(li <= IDX_DIST_MAX),
               "rule": {"idx_dist_max": IDX_DIST_MAX, "dist_high60_max": DIST_HIGH60_MAX},
               "candidates": [{"code": r.ts_code, "close": float(r.close),
                               "pct": float(r.pct), "dist_high60": round(float(r.dist_high60), 2),
                               "circ_mv_yi": round(float(r.circ_mv), 2)}
                              for r in hit.head(3).itertuples()]}
    (OUT / "backfill_latest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if a.json:
        print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
