"""破位 / 深低开补位 —— 合入主策略引擎的一条并行腿（第 248 轮定稿）

规则（与 tools/backfill_daily.py、scan_247k/248c 完全同口径）:

  触发（T 日收盘可知 / T+1 9:25 可知）:
    a. 破位（优先）: 等权指数距 20 日线 <= -3.51%  且  当日 9 窗口排名第一的票
       距前 60 日最高 <= -35%。T+1 开盘从排名往下, 第一只开盘 <= 昨收 x 1.0005 的买入。
    b. 深低开（破位不成立时）: T+1 开盘价落在 -9.5% ~ -5% 之间（含 -5%, 不含 -9.5%）,
       只在前 10 名里挑, 取排名最好的。上界 -9.5% 排除"跌停开盘"（买得到、卖不掉）。

  仓位: 破位 30% / 深低开 50%（按两条路单笔期望 1.5% : 2.5% 配的）。
  持有: T+1 开盘买 → T+2 收盘卖, 不挂限价单。
  限次: 同一只票历史上最多买 8 次。

  ★ 与主策略的资金调度（248）:
    - 主策略 signal 日 / 买入日, 补位不做（钱被占着）。
    - 主策略 signal 日恰好补位也有候选 → **主策略停手, 只做补位**
      （48 天里主策略 +0.52% vs 补位 +2.84%, 且破位日主策略本来就只发过 7 次信号）。

本模块只负责"算出候选名单", 下单/持仓/结算全部复用主引擎的 holding 与 do_sell,
因此补位持仓与主策略持仓共享同一个账户、同一条净值曲线。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WINS = (60, 90, 120, 150, 180, 200, 250, 300, 360)
DH_COLS = [f"dh{W}" for W in WINS]
DEFAULTS = {
    "enabled": False,
    "idx_dist_max": -3.51,
    "dist_high60_max": -35.0,
    "deep_lo": -9.5,
    "deep_hi": -5.0,
    "deep_rank_max": 10,
    "pos_break": 0.30,
    "pos_deep": 0.50,
    "max_times": 8,
    "min_listed_bars": 250,
    "n1_gap_max": 0.05,
    "list_n": 10,
    "let_way": True,
}


def params(cfg: dict) -> dict:
    p = dict(DEFAULTS)
    p.update({k: v for k, v in (cfg.get("fillin") or {}).items() if v is not None})
    return p


def add_indicators(panel: pd.DataFrame, wins=WINS, progress=None) -> pd.DataFrame:
    """给面板补 dh{W} 列 = 距前 W 个交易日最高价（先 shift(1) 再 rolling，不含当日）。

    与 tools/backfill_daily.py 的算法逐字一致：不先删停牌行，因为"前 W 个交易日"
    要按自然交易日算，删行会变成"前 W 个有效行"。
    """
    k = panel["close_adj"] / panel["close"].replace(0, np.nan)
    hi = panel["high"] * k
    hi = hi.astype("float32", copy=False) if panel["close"].dtype == np.float32 else hi
    g = hi.groupby(panel["ts_code"], sort=False)
    for i, W in enumerate(wins, 1):
        col = f"dh{W}"
        if col in panel.columns:
            continue
        minp = max(W // 3, 40)
        h = g.transform(lambda s, W=W: s.shift(1).rolling(W, min_periods=minp).max())
        panel[col] = ((panel["close_adj"] / h - 1) * 100).astype(panel["close_adj"].dtype, copy=False)
        del h
        if progress:
            progress(i, len(wins))
    del hi
    return panel


def market_dist(panel: pd.DataFrame, ma: int = 20) -> pd.Series:
    """全市场等权指数距 20 日线（%）。

    ★ 不能用 src.selector.regime 的 mkt_idx: 那条序列只算主板+创业板、且剔除了
      ST/停牌/上市不足21日的票（regime_base 的口径，为风控服务）。补位的 -3.51%
      阈值来自 tools/build_features_234.py 的"全市场 pct 均值"，两者在 2026-11-17
      这天差 2.6pp（-0.79% vs -3.37%），直接搬阈值会把破位日判错一大片。
      这里按 feat234 的口径独立算一条：不过滤板块、不过滤 ST。
    """
    st = panel.groupby("trade_date")["pct"].mean().sort_index()
    nav = (1 + st / 100).cumprod()
    return (nav / nav.rolling(ma, min_periods=10).mean() - 1) * 100


def _pool(day: pd.DataFrame, p: dict) -> pd.DataFrame:
    """补位候选池：主板 / 非停牌 / 非 ST / 涨跌幅正常 / 上市满 N 日。"""
    d = day
    if "is_paused" in d.columns:
        d = d[d["is_paused"] == 0]
    if "is_st" in d.columns:
        d = d[d["is_st"] == 0]
    d = d[d["pct"].between(-11.5, 11.5)]
    d = d[d["ts_code"].astype(str).str[:2].isin(["60", "00"])]
    if "bars" in d.columns:
        d = d[d["bars"] >= int(p["min_listed_bars"])]
    return d


def rank_day(day: pd.DataFrame, p: dict) -> pd.DataFrame:
    """9 窗口排名百分比平均，升序。名列前茅 = 越多个窗口都认为它跌得透。"""
    d = _pool(day, p)
    if not len(d):
        return d
    d = d.dropna(subset=[c for c in DH_COLS if c in d.columns])
    if not len(d):
        return d
    den = {c: max(int(d[c].notna().sum()), 1) for c in DH_COLS if c in d.columns}
    rk = pd.DataFrame({c: d[c].rank(ascending=True) / den[c] for c in den})
    out = d.copy()
    out["mr"] = rk.mean(axis=1, skipna=True)
    out = out.sort_values(["mr", "amount"], ascending=[True, False])
    out["rk"] = np.arange(1, len(out) + 1)
    return out


def plan(day: pd.DataFrame, idx_dist: float, p: dict) -> dict | None:
    """T 日收盘: 产出补位计划。返回 None = 今天两条路都不成立。

    返回 {"path": "破位"/"深低开", "cands": [{code, rk, close_adj, dh60}]}
    cands 按排名从好到差, T+1 开盘时从上往下挑。
    """
    d = rank_day(day, p)
    if not len(d):
        return None
    top = d.iloc[0]
    broke = (float(idx_dist) <= float(p["idx_dist_max"])
             and float(top["dh60"]) <= float(p["dist_high60_max"]))
    if broke:
        c = d.head(int(p["list_n"]))
        path = "破位"
    else:
        c = d.head(int(p["deep_rank_max"]))
        path = "深低开"
    if not len(c):
        return None
    return {"path": path,
            "cands": [{"code": str(r.ts_code), "rk": int(r.rk),
                       "close_adj": float(r.close_adj), "dh60": float(r.dh60)}
                      for r in c.itertuples()]}


def choose(pl: dict | None, day: pd.DataFrame, p: dict) -> tuple[str, dict] | None:
    """T+1 开盘(9:25): 用开盘价在名单里从上往下挑第一只合格的。"""
    if not pl:
        return None
    px = day.set_index(day["ts_code"].astype(str))
    for cd in pl["cands"]:
        code = cd["code"]
        if code not in px.index:
            continue
        r = px.loc[code]
        if isinstance(r, pd.DataFrame):
            r = r.iloc[0]
        if int(r.get("is_paused", 0) or 0) == 1:
            continue
        o = float(r["open_adj"])
        if not np.isfinite(o) or cd["close_adj"] <= 0:
            continue
        gap = (o / cd["close_adj"] - 1) * 100
        if pl["path"] == "破位":
            if gap <= float(p["n1_gap_max"]):
                return code, {"gap": gap, "path": "破位", "rk": cd["rk"], "open_adj": o}
        else:
            if float(p["deep_lo"]) < gap <= float(p["deep_hi"]):
                return code, {"gap": gap, "path": "深低开", "rk": cd["rk"], "open_adj": o}
    return None


def pos_of(path: str, p: dict) -> float:
    return float(p["pos_break"] if path == "破位" else p["pos_deep"])
