"""补位信号（第 240 轮产出，第 241 轮修正执行形态）

规则:
  1) 市场条件: 全市场等权指数距其 20 日均线 <= -3.51%
  2) 个股条件: 收盘距前 60 日最高价 <= -24.46%（复权，前 60 日不含当日）
  3) 当日(T+1)开盘买，T+2 收盘卖；只在主策略空仓且资金空闲的日子使用

★ 第 241 轮修正（重要）:
  这条规则赚的是"破位日买什么都弹"的 beta，不是选股 alpha。
  证据: 破位日 172 天里, 7 个选股键(市值/换手/振幅/波动/收盘位置/相对大盘/距60日高)
  五档的 y_ex（扣当日全市场均值）全部落在 ±0.2pp; walk-forward 下 alpha 合计为负。
  => 执行形态必须反过来: 越分散越安全。
     全市场等权 100% 仓: 年化 88.0%->112.6%, 回撤 -24.2%->-23.6%, 卡玛 3.63->4.77
     最小市值 1 只 100% 仓: 回撤恶化到 -43.5%, 卡玛 3.63->3.19（不可用）
★ 单吊版（第 242 轮按要求回退）:
  每天只买 1 只, 选"距前 60 日高最远"（跌得最透）的那只, 同分取成交额大的保证能买。
  满仓: 年化 88.0%->152.7%, 回撤 -24.2%->-45.2%, 卡玛 3.63->3.38（年化高但回撤大）
  30% 仓: 年化 107.2%, 回撤 -23.6%, 卡玛 4.54（回撤不变）
  回撤 -45% 是结构性的: 单只票单日最差 -16.7%, 满仓 10 年里必然撞上。
  唯一能压回撤的旋钮是仓位, 不是选股 —— 21 个入场缺口区间里只有 [-8,-3] 好看且端点一动就崩, 判定为过拟合, 未采用。

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
    # 流通市值(亿元): amount 单位元, turnover_rate 单位%
    #   流通市值 = amount / (turnover_rate/100) = amount/turnover_rate*100, 再 /1e8 得亿元
    #   => amount/turnover_rate/1e6
    d["circ_mv"] = d["amount"] / d["turnover_rate"].replace(0, np.nan) / 1e6

    win = d[(d["bars"] >= 60) & d["dist_high60"].notna()].copy()   # 60 日高点已足够排除次新
    win["idx_dist"] = win["trade_date"].map(mkt["idx_dist"])
    trig = win[(win["idx_dist"] <= IDX_DIST_MAX) & (win["dist_high60"] <= DIST_HIGH60_MAX)]

    days = sorted(trig["trade_date"].unique())
    print(f"\n近 {len(use)-90} 个交易日里，触发 {len(days)} 次")

    def single(g: pd.DataFrame) -> pd.DataFrame:
        """单吊: 距 60 日高最远(跌最透), 同分取成交额大的保证买得到"""
        return (g.dropna(subset=["dist_high60"])
                 .sort_values(["dist_high60", "amount"], ascending=[True, False]).head(1))

    rows = []
    for dt in days:
        s = trig[trig["trade_date"] == dt]
        top = single(s)
        if not len(top):
            continue
        rows.append({"signal": dt, "idx_dist": round(float(s["idx_dist"].iloc[0]), 2),
                     "n": len(s), "code": top["ts_code"].iloc[0],
                     "dist_high60": round(float(top["dist_high60"].iloc[0]), 1)})
    hist = pd.DataFrame(rows)
    if len(hist):
        print(hist.to_string(index=False))
        hist.to_csv(OUT / "backfill_history.csv", index=False, encoding="utf-8-sig")

    last = all_dates[-1]
    li = float(mkt["idx_dist"].get(int(last), np.nan))
    hit = single(trig[trig["trade_date"] == int(last)])
    print(f"\n最新交易日 {last}: 等权指数距20日线 {li:+.2f}%  "
          f"{'✅ 触发' if li <= IDX_DIST_MAX else '❌ 未触发'}")
    if len(hit):
        print(hit[["ts_code", "close", "pct", "dist_high60", "circ_mv", "amount"]].to_string(index=False))
        print("  建议: 单吊这一只。满仓(年化152.7%/回撤-45.2%) 或 30%仓(107.2%/-23.6%)，自行选档")

    payload = {"signal_date": last, "idx_dist": round(li, 2),
               "triggered": bool(li <= IDX_DIST_MAX),
               "rule": {"idx_dist_max": IDX_DIST_MAX, "dist_high60_max": DIST_HIGH60_MAX},
               "how_to_trade": "单吊下列首选一只; T+1开盘买, T+2收盘卖; 仓位自选(满仓/30%)",
               "candidates": [{"code": r.ts_code, "close": float(r.close),
                               "pct": float(r.pct), "dist_high60": round(float(r.dist_high60), 2),
                               "circ_mv_yi": round(float(r.circ_mv), 2)}
                              for r in hit.itertuples()]}
    (OUT / "backfill_latest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if a.json:
        print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
