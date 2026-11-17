"""补位信号（第 245 轮定稿：9 窗口集成 + T+1 开盘重选）

规则:
  1) 市场条件: 全市场等权指数距其 20 日均线 <= -3.51%
  2) 破位门槛: 收盘距前 60 日最高价 <= -24.46%（复权，前 60 日不含当日）
  3) 排序: 九个窗口 60/90/120/150/180/200/250/300/360 各自把当日候选按"距前 W 日最高
     最远"排名, 换成排名百分比(0~1), 九个求平均。平均越小 = 越多个窗口都认为它跌得透。
  4) 同一只票 10 个交易日内不重复买
  5) ★ 执行: T 日收盘只把候选排好序; T+1 集合竞价(9:25)看开盘价, 从排好的名单从上往下,
     第一只"开盘价 <= 昨收 x 1.0005"的就是今天的标的, 按开盘价买入, T+2 收盘卖出。
  6) 仓位 50%

★ 为什么是"T+1 开盘重选"而不是"T 日选好再祈祷低开"（第 245 轮）:
  244 轮的做法是 T 日选出 top1, T+1 看它是否低开, 高开就空仓 —— 结果 198 次里 82 次(41%)
  只能放弃。但实盘根本不是这么做的: T+1 9:25 开盘价是公开信息, 完全可以"在低开的票里
  重新挑一只"。同一批数据:
    244 (T日选/T+1放弃): 116 笔, 单笔 2.447%, 123.0% / -23.59% / 5.21
    245 (T+1 重选):      198 笔, 单笔 2.656%, 138.1% / -25.15% / 5.49
  被 244 放弃的 82 天, 245 做成后单笔 2.19%、胜率 65%、合计 +179pp —— 这 82 天不是没得做,
  只是不能买"全体最弱"的那只(它高开说明已经反弹), 要买"低开票里最弱"的那只。

★ 前情（第 244 轮）:
  ① 逐个窗口挑最优是过拟合。历史卡玛排名 -> 次年卡玛排名 的秩相关 = -0.158(9 年 3 正)。
     walk-forward 每年重选得 5.17, 而 45 个组合随机选的中位数就是 4.88 —— 是运气不是本事。
  ② 把 9 个窗口的排名平均, 等于取消"挑窗口"。单窗口中位 4.73 / 最好 5.13 / 集成 5.56;
     随机抽 2/3/5/7/9 个窗口中位 4.75/4.94/5.11/5.37/5.55 单调上升。
  ③ 候选池预过滤 n1_ok 是未来信息（提前知道明天低开）, 244 已剥离。

★ 已知局限:
  - 198 笔 / 11 年, 每年 18 笔。2020 年只有 12 次、单笔 0.21%, 基本没贡献;
    收益集中在 2022 和 2024。
  - 245 与 244 在"244 也成交的 116 天"里逐日配对: 平均 +0.542pp 但胜率只有 12%,
    说明是"少数几次大赢"撑起来的, 不是稳定小胜。
  - 加成交额 >= 0.5 亿 的流动性约束后: 130.8% / -24.82% / 5.27（单笔 2.32%）。

★ 已排除（不要再试）:
  - ETF 抄底: 破位日宽基 ETF 开->次日收 0.39%(胜率 57%), 股票 ETF 0.37%, 债券 0.06%,
    黄金 0.12%; 而个股单吊 2.45%。ETF 当日开->收 全是负的(-0.39%)。差 6 倍, 不用了。
  - 高开日买"最强"的票: 高开组内分五档, 最弱档 0.79% > 最强档 0.24%, 单调, 反着买更差。
  - 244 之前的: 放宽 idx_dist 到 -2.0 / 破位深度定仓 / 跌停家数定仓 / 每波限次 /
    连亏降仓 / 部分止盈止损 / T+1 收盘买（补位单笔掉到 0.059%, 收益 100% 来自低开买入）。

用法:
  python3 tools/backfill_daily.py            # 最新信号日 + 最近 250 日回放
  python3 tools/backfill_daily.py --days 480
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

IDX_DIST_MAX = -3.51       # 等权指数距 20 日线
DIST_HIGH_MAX = -24.46     # 距前 60 日最高（破位门槛，几乎不筛人）
WINS = (60, 90, 120, 150, 180, 200, 250, 300, 360)
MAX_WIN = 360
COOLDOWN = 10              # 同一只票的冷却期(交易日)
POSITION = 0.50            # 单吊仓位
MIN_LISTED_BARS = 250
N1_GAP_MAX = 0.05          # T+1 开盘 <= 昨收 x (1+0.05%) 才算低开
TOP_N = 10                 # 输出前 N 只排序候选, 供 T+1 开盘从上往下挑


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
    use = all_dates[-(a.days + MAX_WIN + 260):]     # 预热: 最长窗口 + 250(上市天数)
    d = load_panel(use)
    print(f"面板 {use[0]}~{use[-1]}  {len(d):,} 行")

    # ---- 市场: 等权指数距 20 日线（与 build_features_234 同口径） ----
    g = d.groupby("trade_date", sort=True)
    st = pd.DataFrame({"eq_ret": g["pct"].mean(), "med_pct": g["pct"].median()}).reset_index()
    st["eq_nav"] = (1 + st["eq_ret"] / 100).cumprod()
    st["ma20"] = st["eq_nav"].rolling(20, min_periods=10).mean()
    st["idx_dist"] = (st["eq_nav"] / st["ma20"] - 1) * 100
    mkt = st.set_index("trade_date").reindex(sorted(d["trade_date"].unique()))

    # ---- 个股: 必须先在未过滤序列上算滚动量, 再过滤股票池 ----
    #   若先删停牌行再 rolling, "前 W 个交易日"会变成"前 W 个有效行", 与回测口径不一致。
    d["adj"] = d["close"] * d["factor"]
    d["adj_h"] = d["high"] * d["factor"]
    d["adj_o"] = d["open"] * d["factor"]
    d = d.sort_values(["ts_code", "trade_date"])
    minp = {W: max(W // 3, 40) for W in WINS}
    for W in WINS:
        # 不含当日: 先 shift(1) 再 rolling（含当日会低估信号强度）
        hi = d.groupby("ts_code")["adj_h"].transform(
            lambda s, W=W: s.shift(1).rolling(W, min_periods=minp[W]).max())
        d[f"dh{W}"] = (d["adj"] / hi - 1) * 100
    d["n1_gap"] = (d.groupby("ts_code")["adj_o"].shift(-1) / d["adj"] - 1) * 100
    d["bars"] = d.groupby("ts_code").cumcount() + 1

    # ---- 过滤股票池: 主板 / 非停牌 / 非 ST / 涨跌幅正常 ----
    # 3 万本金开不了创业板(10万)/科创板(50万)/北交所(50万); 北交所 30% 涨跌幅会放大假收益
    d = d[(d["is_paused"] == 0) & (d["is_st"] == 0)]
    d = d[d["pct"].between(-11.5, 11.5)]
    d = d[d["ts_code"].str[:2].isin(["60", "00"])]

    win = d[(d["bars"] >= MIN_LISTED_BARS) & d["dh60"].notna()].copy()
    win["idx_dist"] = win["trade_date"].map(mkt["idx_dist"])
    win["circ_mv"] = win["amount"] / win["turnover_rate"].replace(0, np.nan) / 1e6
    trig = win[(win["idx_dist"] <= IDX_DIST_MAX) & (win["dh60"] <= DIST_HIGH_MAX)]

    days = sorted(trig["trade_date"].unique())
    print(f"\n近 {a.days} 个交易日里，触发 {len(days)} 次")

    dhcols = [f"dh{W}" for W in WINS]

    def ranked(g: pd.DataFrame, exclude: set | None = None) -> pd.DataFrame:
        """9 窗口排名百分比平均, 升序返回（名列前茅 = 9 个窗口都认为跌得透）"""
        g2 = g.dropna(subset=dhcols).copy()
        if not len(g2):
            return g2
        if exclude:
            g3 = g2[~g2["ts_code"].isin(exclude)]
            if len(g3):
                g2 = g3
        denom = {c: max(int(g2[c].notna().sum()), 1) for c in dhcols}
        rk = pd.DataFrame({c: g2[c].rank(ascending=True) / denom[c] for c in dhcols})
        g2["mr"] = rk.mean(axis=1, skipna=True)
        return g2.sort_values(["mr", "amount"], ascending=[True, False])

    di = {int(d): i for i, d in enumerate(all_dates)}
    last_seen: dict[str, int] = {}
    rows = []
    for dt in days:
        s = trig[trig["trade_date"] == dt]
        ex = {c for c, i in last_seen.items() if di[int(dt)] - i < COOLDOWN}
        cand = ranked(s, ex)
        if not len(cand):
            continue
        # ★ T+1 开盘: 从上往下第一只低开的
        pick = cand[cand["n1_gap"] <= N1_GAP_MAX]
        if not len(pick):
            rows.append({"signal": dt, "idx_dist": round(float(s["idx_dist"].iloc[0]), 2),
                         "n": len(s), "code": "-", "mr": np.nan, "gap": np.nan})
            continue
        top = pick.head(1)
        last_seen[top["ts_code"].iloc[0]] = di[int(dt)]
        r = {"signal": dt, "idx_dist": round(float(s["idx_dist"].iloc[0]), 2),
             "n": len(s), "code": top["ts_code"].iloc[0],
             "mr": round(float(top["mr"].iloc[0]), 3),
             "gap": round(float(top["n1_gap"].iloc[0]), 2)}
        for W in WINS:
            r[f"dh{W}"] = round(float(top[f"dh{W}"].iloc[0]), 1)
        rows.append(r)
    hist = pd.DataFrame(rows)
    if len(hist):
        ok = hist[hist["code"] != "-"]
        print(hist[["signal", "idx_dist", "n", "code", "mr", "gap", "dh60", "dh250"]]
              .to_string(index=False))
        print(f"\n回放: 触发 {len(hist)} 次, 成交 {len(ok)} 次"
              + (f", 平均低开 {ok['gap'].mean():.2f}%" if len(ok) else ""))
        hist.to_csv(OUT / "backfill_history.csv", index=False, encoding="utf-8-sig")

    # ---- 最新交易日: 只给排序名单, 开盘价明天才知道 ----
    last = all_dates[-1]
    li = float(mkt["idx_dist"].get(int(last), np.nan))
    ex_now = {c for c, i in last_seen.items() if di[int(last)] - i < COOLDOWN}
    cand = ranked(trig[trig["trade_date"] == int(last)], ex_now)
    print(f"\n最新交易日 {last}: 等权指数距20日线 {li:+.2f}%  "
          f"{'✅ 触发' if li <= IDX_DIST_MAX else '❌ 未触发'}")
    if len(cand):
        show = cand.head(TOP_N)
        print(show[["ts_code", "close", "pct", "mr", "dh60", "dh250", "dh360", "amount"]]
              .to_string(index=False))
        print(f"  ★ 操作: 明天 9:25 按上表从上往下, 第一只开盘价 <= 昨收×1.0005 的买入")
        print(f"    仓位 50%, 后天收盘卖出。今天只排序, 不下单。")
        print(f"    历史: 年化 138% / 回撤 -25.2% / 卡玛 5.49（244 的 T日选口径 123%/-23.6%/5.21）")

    payload = {"signal_date": last, "idx_dist": round(li, 2),
               "triggered": bool(li <= IDX_DIST_MAX),
               "rule": {"idx_dist_max": IDX_DIST_MAX, "windows": list(WINS),
                        "rank": "nine-window percentile rank mean",
                        "dist_high60_max": DIST_HIGH_MAX, "position": POSITION,
                        "cooldown_days": COOLDOWN, "n1_gap_max": N1_GAP_MAX,
                        "top_n": TOP_N},
               "how_to_trade": ("T+1 9:25 按 candidates 顺序, 第一只开盘价 <= 昨收×1.0005 "
                                "的按开盘价买入; 50% 仓位; T+2 收盘卖出"),
               "candidates": [{"rank": i + 1, "code": r.ts_code, "close": float(r.close),
                               "pct": float(r.pct), "rank_mean": round(float(r.mr), 4),
                               "dh60": round(float(r.dh60), 2),
                               "dh250": round(float(r.dh250), 2),
                               "dh360": round(float(r.dh360), 2),
                               "amt_yi": round(float(r.amount) / 1e8, 2),
                               "circ_mv_yi": round(float(r.circ_mv), 2)}
                              for i, r in enumerate(cand.head(TOP_N).itertuples())]}
    (OUT / "backfill_latest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if a.json:
        print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
