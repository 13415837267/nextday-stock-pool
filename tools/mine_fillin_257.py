"""第257轮: 从全历史行情里挖补位条件（事件级，不是回测级）

背景
----
256 轮把补位改成"漏斗三层"，但所有阈值（dist20 <= -5%、top1 距 60 日高 <= -42%、
深低开 -8%~-5%、跳过前 3 名）都是 2500 日里 50 笔成交拟合出来的。
50 笔上做四五个参数的切分 = 过拟合。

做法
----
不跑回测，直接把 2009-2026 全历史（4301 日）里**所有**满足 dist20 <= -3% 的
极端日拉出来，逐日逐层记录：
  · 当日市场状态（dist20 / 下跌广度 / 5 日、20 日指数涨跌）
  · 候选池前 8 名的 9 窗口距前高、当日跌幅、成交额、换手
  · 次日开盘缺口、是否买得进、次日开盘买 T+2 收盘卖的两日收益

输出 output/mine/events.parquet，一行 = 一个(极端日, 层)。
之后所有条件筛选都在这个事件表上做，样本从 50 笔变成几百天 × 8 层。

内存: cgroup 8GB。按 600 日分块读, 每块带 400 日预热(最大窗口 360)。
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data" / "panel"
OUT = ROOT / "output" / "mine"
OUT.mkdir(parents=True, exist_ok=True)

WINS = (60, 90, 120, 150, 180, 200, 250, 300, 360)
TOPN = 8
DIST_MIN = -3.0        # 只收 dist20 低于此值的日子(放宽, 便于看阈值曲线)
BLOCK, WARM = 600, 400

USECOLS = ["ts_code", "trade_date", "open", "high", "low", "close", "pre_close",
           "pct", "vol", "amount", "factor", "turnover_rate", "is_paused", "is_st",
           "high_limit"]


def load_files(files: list[Path]) -> pd.DataFrame:
    frames = []
    for f in files:
        d = pd.read_csv(f, usecols=USECOLS)
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    for c in ("open", "high", "low", "close", "pre_close", "pct", "amount",
              "factor", "turnover_rate", "high_limit"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    df["vol"] = pd.to_numeric(df["vol"], errors="coerce").astype("float32")
    df["is_paused"] = pd.to_numeric(df["is_paused"], errors="coerce").fillna(0).astype("int8")
    df["is_st"] = pd.to_numeric(df["is_st"], errors="coerce").fillna(0).astype("int8")
    df["trade_date"] = df["trade_date"].astype("int32")
    df["ts_code"] = df["ts_code"].astype("category")
    return df


def market_series(files: list[Path]) -> pd.DataFrame:
    """全历史每日等权指数状态（与 fillin.market_dist 同口径：不过滤板块/ST）。"""
    rows = []
    for f in files:
        d = pd.read_csv(f, usecols=["trade_date", "pct", "pre_close"])
        d["pct"] = pd.to_numeric(d["pct"], errors="coerce")
        d.loc[pd.to_numeric(d["pre_close"], errors="coerce") <= 0, "pct"] = np.nan
        rows.append((int(d["trade_date"].iloc[0]),
                     float(d["pct"].mean()),
                     float((d["pct"] < 0).mean()),
                     int(d["pct"].notna().sum())))
    s = pd.DataFrame(rows, columns=["trade_date", "mkt_pct", "breadth_down", "n_stock"])
    s = s.sort_values("trade_date").reset_index(drop=True)
    nav = (1 + s["mkt_pct"] / 100).cumprod()
    s["dist20"] = (nav / nav.rolling(20, min_periods=10).mean() - 1) * 100
    s["idx_ret5"] = nav.pct_change(5) * 100
    s["idx_ret20"] = nav.pct_change(20) * 100
    s["dist60"] = (nav / nav.rolling(60, min_periods=30).mean() - 1) * 100
    return s


def block_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """复权价 + 9 个窗口的距前高（shift(1) 再 rolling，与 fillin.add_indicators 同口径）。"""
    df["adj_close"] = df["close"] * df["factor"]
    df["adj_open"] = df["open"] * df["factor"]
    df["adj_high"] = df["high"] * df["factor"]
    df["adj_low"] = df["low"] * df["factor"]
    g = df.groupby("ts_code", sort=False, observed=True)
    df["bars"] = g.cumcount() + 1
    for W in WINS:
        minp = max(W // 3, 40)
        h = g["adj_high"].transform(lambda s, W=W, minp=minp: s.shift(1).rolling(W, min_periods=minp).max())
        df[f"dh{W}"] = ((df["adj_close"] / h - 1) * 100).astype("float32")
        del h
        gc.collect()
    df["bias20"] = df.groupby("ts_code", sort=False, observed=True)["adj_close"].transform(
        lambda s: (s / s.shift(1).rolling(20, min_periods=10).mean() - 1) * 100).astype("float32")
    return df


def pool_filter(d: pd.DataFrame) -> pd.DataFrame:
    d = d[(d["is_paused"] == 0) & (d["is_st"] == 0)]
    d = d[d["pct"].between(-11.5, 11.5)]
    d = d[d["ts_code"].astype(str).str[:2].isin(["60", "00"])]
    d = d[d["bars"] >= 250]
    return d


def rank_pool(d: pd.DataFrame) -> pd.DataFrame:
    d = d.dropna(subset=[f"dh{W}" for W in WINS])
    if not len(d):
        return d
    den = {f"dh{W}": max(int(d[f"dh{W}"].notna().sum()), 1) for W in WINS}
    rk = pd.DataFrame({c: d[c].rank(ascending=True) / den[c] for c in den})
    d = d.copy()
    d["mr"] = rk.mean(axis=1, skipna=True)
    d = d.sort_values(["mr", "amount"], ascending=[True, False])
    d["rk"] = np.arange(1, len(d) + 1)
    return d


def main() -> None:
    files = sorted(PANEL.glob("*.csv.gz"))
    lo = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    files = files[lo:]
    print(f"面板 {len(files)} 个交易日: {files[0].name} ~ {files[-1].name}", flush=True)

    print("[1/2] 市场状态序列...", flush=True)
    mkt = market_series(files)
    mkt.to_parquet(OUT / "market.parquet", index=False)
    ext = mkt[mkt["dist20"] <= DIST_MIN]
    print(f"  dist20 <= {DIST_MIN} 的极端日 {len(ext)} 天 / 共 {len(mkt)} 天", flush=True)
    ext_dates = set(ext["trade_date"].tolist())
    mkt_idx = mkt.set_index("trade_date")

    dates = [int(f.name[:8]) for f in files]
    rows: list[dict] = []
    i = 0
    nblk = 0
    while i < len(files):
        start = max(0, i - WARM)
        end = min(len(files), i + BLOCK)
        nblk += 1
        sub = files[start:end]
        print(f"[2/2] 块{nblk}: {sub[0].name[:8]}~{sub[-1].name[:8]} ({len(sub)} 文件)", flush=True)
        df = load_files(sub)
        df = block_indicators(df)
        # 只处理本块新覆盖的日期，末尾 2 天留给下一块（需要 T+2 数据）
        d0 = dates[i]
        d1 = dates[end - 1]
        stop = end - 1 if end < len(files) else end
        # 全样本：深低开腿与指数无关，任何一天都可能触发 -> 所有交易日都记
        handle = [d for d in dates[i:stop]]

        # 预建日期索引，避免逐日全表扫描
        df = df.sort_values("trade_date", kind="stable")
        by_date = dict(tuple(df.groupby("trade_date", sort=False)))
        all_dates_in_block = sorted(by_date)
        pos = {v: i for i, v in enumerate(all_dates_in_block)}

        for di, d in enumerate(handle):
            if d not in by_date:
                continue
            k = pos[d]
            if k + 2 >= len(all_dates_in_block):
                continue
            d1_ = all_dates_in_block[k + 1]
            d2_ = all_dates_in_block[k + 2]
            day = pool_filter(by_date[d])
            if len(day) < 10:
                continue
            day = rank_pool(day)
            if not len(day):
                continue
            nxt = by_date[d1_].set_index(by_date[d1_]["ts_code"].astype(str))
            nx2 = by_date[d2_].set_index(by_date[d2_]["ts_code"].astype(str))
            mrow = mkt_idx.loc[d]
            top = day.head(TOPN)
            for r in top.itertuples():
                code = str(r.ts_code)
                rec = {"date": d, "dist20": float(mrow["dist20"]),
                       "breadth_down": float(mrow["breadth_down"]),
                       "idx_ret5": float(mrow["idx_ret5"]),
                       "idx_ret20": float(mrow["idx_ret20"]),
                       "dist60": float(mrow["dist60"]),
                       "is_ext": int(d in ext_dates),
                       "pool_n": int(len(day)), "layer": int(r.rk), "code": code,
                       "dh60": float(r.dh60), "dh90": float(r.dh90), "dh120": float(r.dh120),
                       "pct": float(r.pct), "amount": float(r.amount),
                       "turnover": float(r.turnover_rate), "close": float(r.close),
                       "bars": int(r.bars), "bias20": float(r.bias20),
                       "ret_d1": np.nan, "ret_d2": np.nan, "gap": np.nan, "buyable": 0}
                if code in nxt.index:
                    x = nxt.loc[code]
                    if isinstance(x, pd.DataFrame):
                        x = x.iloc[0]
                    o1 = float(x["adj_open"])
                    if np.isfinite(o1) and r.adj_close > 0:
                        rec["gap"] = (o1 / float(r.adj_close) - 1) * 100
                        rec["buyable"] = int(int(x["is_paused"]) == 0 and float(x["open"]) < float(x["high_limit"]) - 1e-6)
                        if code in nx2.index:
                            y = nx2.loc[code]
                            if isinstance(y, pd.DataFrame):
                                y = y.iloc[0]
                            for tag, px in (("ret_d1", float(x["adj_close"])), ("ret_d2", float(y["adj_close"]))):
                                if np.isfinite(px):
                                    rec[tag] = (px / o1 - 1) * 100
                rows.append(rec)
        del df, by_date
        gc.collect()
        i = stop
        if (OUT / "events.parquet").exists() and nblk == 1:
            pass

    ev = pd.DataFrame(rows)
    ev.to_parquet(OUT / "events.parquet", index=False)
    print(f"事件表 {len(ev):,} 行 / {ev['date'].nunique()} 天 -> {OUT/'events.parquet'}", flush=True)


if __name__ == "__main__":
    main()
