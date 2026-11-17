"""补位信号（第 244 轮定稿：9 窗口排名集成）

规则:
  1) 市场条件: 全市场等权指数距其 20 日均线 <= -3.51%
  2) 个股条件: 收盘距前 60 日最高价 <= -24.46%（复权，前 60 日不含当日）—— 仅作破位门槛
  3) 选股: 九个窗口 60/90/120/150/180/200/250/300/360 各自把当日候选按"距前 W 日最高
     最远"排名, 取排名百分比, 九个百分比求平均, 取平均最小的一只（9 个窗口都认为跌得透）
  4) 同一只票 10 个交易日内不重复买
  5) 当日(T+1)开盘买，T+2 收盘卖；只在主策略空仓且资金空闲的日子使用；50% 仓位

★ 为什么改成集成（第 244 轮）:
  ① 逐个窗口挑最优是过拟合。用 2016-2025 数据算"历史卡玛排名 -> 次年卡玛排名"的秩相关
     = -0.158（9 年只有 3 年为正）。按历史表现挑窗口, 下一年平均更差。
     walk-forward 每年重选窗口得卡玛 5.17, 而 45 个参数组合随机选的中位数就是 4.88 ——
     5.17 只是分布里的运气抽样, 不是本事。
  ② 243 轮定稿的 (250 日窗口 + 20 日冷却) 在 2017 年以后只有 4.41, 比随机选参数的中位数
     还低。它是全样本挑出来的, 全样本 5.70 里含 2016-2021 的运气。
  ③ 把 9 个窗口的排名平均, 等于把"挑窗口"这件事取消了。实测(冷却10/仓位50%):
     单窗口中位 4.73、最好的单窗口 5.13、9 窗口平均 5.56 —— 比最好的单窗口还高。
     随机抽 2/3/5/7/9 个窗口做同样的事, 卡玛中位 4.75/4.94/5.11/5.37/5.55 单调上升,
     说明增益来自"平均"本身, 不是"9"这个数字碰巧。
  ④ 定稿表现(198 次): 年化 136.5% / 回撤 -24.55% / 卡玛 5.56 / 141 只票 / 最长连买 2 次
     对照 243 定稿同口径: 117.2% / -27.09% / 4.33 / 138 只票
     三段卡玛 2.11 / 10.72 / 11.89 全面高于 243 的 1.49 / 9.40 / 11.30
  ⑤ 参数面是平台不是孤岛: 冷却 5~30 × 仓位 40~50% 的卡玛全在 5.15~5.56。

★ 第 244 轮追加（实盘口径，最重要的一条）:
  前面所有轮次的回测都用了 build_features_234 的 n1_ok = (T+1 开盘 <= T 收盘+0.05%) 去过滤
  候选池, 等于 T 日收盘就提前知道"哪只票明天低开"。实盘做不到。
  改成"T 日选一只, T+1 集合竞价若高开就放弃(空仓)"之后:
    198 次信号里 82 次(41%)只能放弃;  成交 116 次
    243 定稿(250日/冷却20/70%): 117.2/-27.09/4.33 -> 100.9/-24.17/4.17
    244 集成(9窗口/冷却10/70%): 158.3/-31.12/5.09 -> 123.0/-23.59/5.21
  两个发现:
    a) 放弃的 82 笔是差交易 —— 破位日高开=已经反弹, 你追在反弹的高点买入再持一天。
       空仓比"退而求其次买次选票"更好(卡玛 5.09 -> 5.21)。
    b) 回撤从 -31.1 改善到 -23.6, 所以实盘口径下仓位可以加回 70%
       (旧口径 P=50% 最优 5.56, 实盘口径 P=70% 最优 5.21, P=50% 只有 4.77)。
  冷却档在实盘口径下不再平坦: C=5 -> 5.51, C=10 -> 5.21, C=15 -> 4.26。
  116 笔样本太少, 单个冷却值不可信; 取 10 日(约两周年化交易)居中, 报 5.2 而不是 5.5。

★ 已知局限:
  - 集成 vs 单窗口 250 的同日配对差 +0.779pp/笔, t=1.73, 198 笔 —— 统计上还不算显著,
    只能说"更好且更分散", 不能说"已被证明"。
  - 2020 年只有 12 次、合计 +9%, 基本没贡献; 收益集中在 2022 和 2024 两年。
  - 单笔 2.58% 是 11 年回测的均值, 实盘受集合竞价成交、容量、情绪影响。

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

IDX_DIST_MAX = -3.51       # 等权指数距 20 日线
DIST_HIGH_MAX = -24.46     # 距前 60 日最高（破位门槛，几乎不筛人）
WINS = (60, 90, 120, 150, 180, 200, 250, 300, 360)
MAX_WIN = 360
COOLDOWN = 10              # 同一只票的冷却期(交易日)
POSITION = 0.70            # 单吊仓位（实盘口径下回撤降到 -23.6%, 可以加回 70%）
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
    use = all_dates[-(a.days + MAX_WIN + 260):]     # 预热: 最长窗口+250(上市天数判定)
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

    # ---- 个股: 复权价距前 W 日最高（不含当日） ----
    # ★★ 顺序很重要: 必须先在全量(含停牌/ST)序列上算 rolling, 再过滤股票池。
    #    若先删停牌行再 rolling, "前 W 个交易日"会变成"前 W 个有效行", 窗口跨度被拉长,
    #    与回测(build 在未过滤面板上算)口径不一致 —— 实测会有一半以上的日子选股不同。
    d["adj"] = d["close"] * d["factor"]
    d["adj_h"] = d["high"] * d["factor"]
    d = d.sort_values(["ts_code", "trade_date"])
    minp = {W: max(W // 3, 40) for W in WINS}
    for W in WINS:
        # ★ 不含当日: 先 shift(1) 再 rolling。含当日会低估信号强度（243 轮口径 bug）
        hi = d.groupby("ts_code")["adj_h"].transform(
            lambda s, W=W: s.shift(1).rolling(W, min_periods=minp[W]).max())
        d[f"dh{W}"] = (d["adj"] / hi - 1) * 100
    d["bars"] = d.groupby("ts_code").cumcount() + 1     # 面板内已上市交易日数(近似)
    d["circ_mv"] = d["amount"] / d["turnover_rate"].replace(0, np.nan) / 1e6

    # ---- 过滤股票池: 主板 / 非停牌 / 非 ST / 涨跌幅正常 ----
    # ★ 只留主板: 3 万本金开不了创业板(10万)/科创板(50万)/北交所(50万), 且北交所
    #   30% 涨跌幅会把"小市值弹性"放大成虚假收益 —— 240 轮就是栽在这(49 笔里 22 笔是 BJ)。
    d = d[(d["is_paused"] == 0) & (d["is_st"] == 0)]
    d = d[d["pct"].between(-11.5, 11.5)]
    d = d[d["ts_code"].str[:2].isin(["60", "00"])]

    win = d[(d["bars"] >= MIN_LISTED_BARS) & d["dh60"].notna()].copy()
    win["idx_dist"] = win["trade_date"].map(mkt["idx_dist"])
    trig = win[(win["idx_dist"] <= IDX_DIST_MAX) & (win["dh60"] <= DIST_HIGH_MAX)]

    days = sorted(trig["trade_date"].unique())
    print(f"\n近 {a.days} 个交易日里，触发 {len(days)} 次")

    dhcols = [f"dh{W}" for W in WINS]

    def rank_one(g: pd.DataFrame, exclude: set | None = None) -> pd.DataFrame:
        """9 窗口排名百分比平均, 取最小。exclude=冷却期内已买过的票。
        全被排除时退回不排除，宁可重复也不空仓"""
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
        return g2.sort_values(["mr", "amount"], ascending=[True, False]).head(1)

    di = {int(d): i for i, d in enumerate(all_dates)}
    last_seen: dict[str, int] = {}
    rows = []
    for dt in days:
        s = trig[trig["trade_date"] == dt]
        ex = {c for c, i in last_seen.items() if di[int(dt)] - i < COOLDOWN}
        top = rank_one(s, ex)
        if not len(top):
            continue
        last_seen[top["ts_code"].iloc[0]] = di[int(dt)]
        r = {"signal": dt, "idx_dist": round(float(s["idx_dist"].iloc[0]), 2),
             "n": len(s), "code": top["ts_code"].iloc[0],
             "mr": round(float(top["mr"].iloc[0]), 3)}
        for W in WINS:
            r[f"dh{W}"] = round(float(top[f"dh{W}"].iloc[0]), 1)
        rows.append(r)
    hist = pd.DataFrame(rows)
    if len(hist):
        print(hist[["signal", "idx_dist", "n", "code", "mr", "dh60", "dh250", "dh360"]]
              .to_string(index=False))
        hist.to_csv(OUT / "backfill_history.csv", index=False, encoding="utf-8-sig")

    last = all_dates[-1]
    li = float(mkt["idx_dist"].get(int(last), np.nan))
    ex_now = {c for c, i in last_seen.items() if di[int(last)] - i < COOLDOWN}
    hit = rank_one(trig[trig["trade_date"] == int(last)], ex_now)
    print(f"\n最新交易日 {last}: 等权指数距20日线 {li:+.2f}%  "
          f"{'✅ 触发' if li <= IDX_DIST_MAX else '❌ 未触发'}")
    if len(hit):
        cols = ["ts_code", "close", "pct", "mr"] + dhcols + ["circ_mv", "amount"]
        print(hit[cols].to_string(index=False))
        chill = f"；冷却期内跳过 {sorted(ex_now)}" if ex_now else ""
        print(f"  建议: 单吊这一只、70% 仓位{chill}")
        print("        ★ T+1 集合竞价开盘价若高于昨收 +0.05% -> 放弃, 空仓一天")
        print("        集成 + 10日冷却 + 70%仓 + 高开放弃: 年化 123.0% / 回撤 -23.6% / 卡玛 5.21"
              "（243 定稿同口径 100.9/-24.2/4.17）")

    payload = {"signal_date": last, "idx_dist": round(li, 2),
               "triggered": bool(li <= IDX_DIST_MAX),
               "entry_rule": "T+1 集合竞价: 开盘价 <= 昨收*1.0005 才买, 否则放弃",
               "rule": {"idx_dist_max": IDX_DIST_MAX, "windows": list(WINS),
                        "rank": "nine-window percentile rank mean",
                        "dist_high60_max": DIST_HIGH_MAX, "position": POSITION,
                        "cooldown_days": COOLDOWN},
               "how_to_trade": "单吊下列首选一只, 70% 仓位; T+1 集合竞价若开盘价高于昨收+0.05% 就放弃(空仓), 否则开盘价买入, T+2 收盘卖",
               "candidates": [{"code": r.ts_code, "close": float(r.close),
                               "pct": float(r.pct), "rank_mean": round(float(r.mr), 3),
                               "dh60": round(float(r.dh60), 2),
                               "dh250": round(float(r.dh250), 2),
                               "dh360": round(float(r.dh360), 2),
                               "circ_mv_yi": round(float(r.circ_mv), 2)}
                              for r in hit.itertuples()]}
    (OUT / "backfill_latest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if a.json:
        print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
