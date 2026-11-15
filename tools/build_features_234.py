"""第二百三十四轮: 全史全市场特征表(4298日 x 5579只 = 约2400万行)

范式切换 —— 前233轮是"演绎": 人先写条件, 再扫参数。搜索空间被人的先验限死,
只能验证"手写条件的最优档位", 永远发现不了手写条件之外的东西。
本轮是"归纳": 全样本 -> 找上涨样本的共性 -> 从共性生成条件 -> 再验证。

关键设计(避免229轮踩过的坑):
  特征: 仅用 T 日收盘及以前 (可执行)
  标签: T+1开盘买入 -> T+2收盘卖出 (策略真实持仓口径)
  样本: 只保留 T+1 低开(open<=close) 的样本 —— 这是挂前收价必然成交的真实子集。
        该条件不含未来函数(挂单在T日收盘后盲挂, 是否成交由T+1竞价决定),
        但它是"成交机制"而非"选股条件", 故作为样本筛选, 不作为特征。

输出: output/feat234/f{year}.parquet, 每年一个文件(约130万行), 避免2400万行同时驻留内存。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data" / "panel"
OUT = ROOT / "output" / "feat234"


def load_days(files: list[Path]) -> pd.DataFrame:
    cols = ["ts_code", "trade_date", "open", "high", "low", "close",
            "pre_close", "pct", "vol", "amount", "factor",
            "high_limit", "low_limit", "turnover_rate", "is_paused", "is_st"]
    parts = []
    for f in files:
        try:
            parts.append(pd.read_csv(f, usecols=cols,
                                     dtype={"ts_code": "str", "trade_date": "int32"}))
        except Exception as e:                                    # noqa: BLE001
            print(f"  skip {f.name}: {type(e).__name__}", flush=True)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def market_state(d: pd.DataFrame) -> pd.DataFrame:
    """逐日市场状态(等权口径), 只用当日及以前。"""
    g = d.groupby("trade_date", sort=True)
    st = pd.DataFrame({
        "breadth": g["pct"].apply(lambda s: float((s > 0).mean())),
        "med_pct": g["pct"].median(),
        "eq_ret": g["pct"].mean(),                 # 等权指数日收益 %
        "amt_yi": g["amount"].sum() / 1e8,
        "limit_up": g["pct"].apply(lambda s: float((s >= 9.7).sum())),
        "limit_dn": g["pct"].apply(lambda s: float((s <= -9.7).sum())),
    }).reset_index()
    st["eq_nav"] = (1 + st["eq_ret"] / 100).cumprod()
    st["idx_ma20"] = st["eq_nav"].rolling(20, min_periods=10).mean()
    st["idx_ma20_dist"] = (st["eq_nav"] / st["idx_ma20"] - 1) * 100
    st["idx_ret20"] = (st["eq_nav"] / st["eq_nav"].shift(20) - 1) * 100
    st["breadth_ma5"] = st["breadth"].rolling(5, min_periods=3).mean()
    st["limit_up_ma5"] = st["limit_up"].rolling(5, min_periods=3).mean()
    st["amt_yi_ma20"] = st["amt_yi"].rolling(20, min_periods=10).mean()
    st["amt_shrink"] = st["amt_yi"] / st["amt_yi_ma20"]
    return st


def build_year(year_files: list[Path], mkt: pd.DataFrame,
               first_seen: dict[str, int], day_idx: dict[int, int]) -> pd.DataFrame:
    d = load_days(year_files)
    if d.empty:
        return d

    # 后复权价: close*factor, 比例类特征不变
    d["adj"] = d["close"] * d["factor"].fillna(1.0)
    d["adj_o"] = d["open"] * d["factor"].fillna(1.0)
    d["adj_h"] = d["high"] * d["factor"].fillna(1.0)
    d["adj_l"] = d["low"] * d["factor"].fillna(1.0)

    d = d.sort_values(["ts_code", "trade_date"], kind="mergesort").reset_index(drop=True)

    def roll(col: str, w: int, fn: str = "mean", mp: int | None = None) -> pd.Series:
        mp = mp if mp is not None else max(2, w // 2)
        r = d.groupby("ts_code", sort=False)[col].rolling(w, min_periods=mp)
        s = getattr(r, fn)().reset_index(level=0, drop=True)
        return s.sort_index()

    def shift(col: str, k: int) -> pd.Series:
        return d.groupby("ts_code", sort=False)[col].shift(k)

    # ---- 当日形态 ----
    span = (d["high"] - d["low"]).replace(0, np.nan)
    d["amp"] = span / d["pre_close"] * 100
    d["close_pos"] = ((d["close"] - d["low"]) / span).clip(0, 1)
    d["low_tail"] = (np.minimum(d["open"], d["close"]) - d["low"]) / d["pre_close"] * 100
    d["up_shadow"] = (d["high"] - np.maximum(d["open"], d["close"])) / d["pre_close"] * 100
    d["gap"] = (d["open"] / d["pre_close"] - 1) * 100
    d["intraday"] = (d["close"] / d["open"] - 1) * 100
    d["vr"] = d["vol"] / roll("vol", 5, mp=3).replace(0, np.nan)      # 量比
    d["amt_yi"] = d["amount"] / 1e8
    d["amt_ratio"] = d["amount"] / roll("amount", 20, mp=10).replace(0, np.nan)
    d["circ_mv"] = d["amount"] / d["turnover_rate"].replace(0, np.nan) / 1e6

    # ---- 位置 ----
    for w in (5, 10, 20, 60):
        ma = roll("adj", w, mp=max(3, w // 2))
        d[f"bias{w}"] = (d["adj"] / ma - 1) * 100
    d["dist_high60"] = (d["adj"] / roll("adj_h", 60, "max", mp=20) - 1) * 100
    d["dist_low60"] = (d["adj"] / roll("adj_l", 60, "min", mp=20) - 1) * 100
    d["dist_high20"] = (d["adj"] / roll("adj_h", 20, "max", mp=10) - 1) * 100

    # ---- 动量 ----
    for w in (3, 5, 10, 20):
        d[f"ret{w}"] = (d["adj"] / shift("adj", w) - 1) * 100
    d["vol20"] = roll("pct", 20, "std", mp=10)
    d["maxdd20"] = d["adj"] / roll("adj_h", 20, "max", mp=10) - 1

    # 连跌/连涨天数(用当日 pct 符号)
    sign = np.sign(d["pct"].fillna(0)).astype("int8")
    d["_dn"] = (sign < 0).astype("int8")
    d["_up"] = (sign > 0).astype("int8")
    dn = d.groupby("ts_code", sort=False)["_dn"]
    # 连续计数: 用累计"非跌"位置差
    idx = np.arange(len(d))
    d["_block"] = idx
    def streak(mask: pd.Series) -> pd.Series:
        # 连续 True 的长度
        s = mask.astype("int8").values
        code = d["ts_code"].values
        out = np.zeros(len(s), dtype="int8")
        c = 0
        for i in range(len(s)):
            if i and code[i] != code[i - 1]:
                c = 0
            c = c + 1 if s[i] else 0
            out[i] = min(c, 30)
        return pd.Series(out, index=d.index)
    d["dn_streak"] = streak(sign < 0)
    d["up_streak"] = streak(sign > 0)
    del dn, sign

    # ---- 相对强度(与市场比) ----
    m = mkt.set_index("trade_date")
    d["mkt_pct"] = d["trade_date"].map(m["eq_ret"])
    d["rel_pct"] = d["pct"] - d["mkt_pct"]
    d["mkt_ret20"] = d["trade_date"].map(m["idx_ret20"])
    d["rs20"] = d["ret20"] - d["mkt_ret20"]
    d["breadth"] = d["trade_date"].map(m["breadth"])
    d["idx_dist"] = d["trade_date"].map(m["idx_ma20_dist"])

    # ---- 上市天数(全史累积: 分年 cumcount 只能得到年内序号, 会漏掉"区间内上市")----
    _di = d["trade_date"].map(day_idx).astype("float32")
    _fs = d["ts_code"].map(first_seen).astype("float32")
    d["listed"] = (_di - _fs + 1).fillna(0)

    return d


def add_labels(d: pd.DataFrame) -> pd.DataFrame:
    """标签: T+1开盘买 -> T+2收盘卖。另存若干备选口径。"""
    g = d.groupby("ts_code", sort=False)
    n1_o = g["adj_o"].shift(-1)          # T+1 开盘
    n1_c = g["adj_c"] if "adj_c" in d else g["adj"].shift(-1)
    n2_c = g["adj"].shift(-2)            # T+2 收盘
    n3_c = g["adj"].shift(-3)
    d["y"] = (n2_c / n1_o - 1) * 100
    d["y_1d"] = (n1_c / n1_o - 1) * 100
    d["y_3d"] = (n3_c / n1_o - 1) * 100
    d["n1_gap"] = (n1_o / d["adj"] - 1) * 100       # T+1 缺口(相对T收)
    d["n1_ok"] = d["n1_gap"] <= 0.05                # 低开(留 0.05% 容差=跌停价挂单可成交)
    return d


FEATS = ["pct", "amp", "close_pos", "low_tail", "up_shadow", "gap", "intraday",
         "vr", "turnover_rate", "amt_yi", "amt_ratio", "circ_mv",
         "bias5", "bias10", "bias20", "bias60",
         "dist_high60", "dist_low60", "dist_high20",
         "ret3", "ret5", "ret10", "ret20", "vol20", "maxdd20",
         "dn_streak", "up_streak", "rel_pct", "rs20",
         "breadth", "idx_dist", "listed"]


def main() -> None:
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 2009
    end = int(sys.argv[2]) if len(sys.argv) > 2 else 2026
    OUT.mkdir(parents=True, exist_ok=True)

    files = sorted(PANEL.glob("*.csv.gz"))
    print(f"面板 {len(files)} 日", flush=True)

    # 市场状态需要全史(MA20 要历史), 但只用轻列
    t0 = time.time()
    all_d = load_days(files)
    mkt = market_state(all_d)
    print(f"市场状态 {len(mkt)} 日, {time.time()-t0:.0f}s", flush=True)
    mkt.to_csv(OUT / "market.csv", index=False)
    del all_d

    # 全局上市日期: 该股票在面板中首次出现的交易日序号
    day_idx = {int(t): i for i, t in enumerate(mkt["trade_date"].tolist())}
    print("扫描上市日期...", flush=True)
    fs_parts = []
    for f in files:
        fs_parts.append(pd.read_csv(f, usecols=["ts_code", "trade_date"],
                                    dtype={"ts_code": "str", "trade_date": "int32"}))
    _all = pd.concat(fs_parts, ignore_index=True)
    first_seen = _all.groupby("ts_code")["trade_date"].min().map(day_idx).to_dict()
    first_seen = {k: int(v) for k, v in first_seen.items()}
    print(f"  {len(first_seen)} 只股票", flush=True)
    del _all, fs_parts

    years = sorted({int(f.stem[:4]) for f in files})
    for y in years:
        if y < start or y > end:
            continue
        yf = [f for f in files if int(f.stem[:4]) == y]
        t = time.time()
        d = build_year(yf, mkt, first_seen, day_idx)
        if d.empty:
            continue
        d = add_labels(d)
        keep = ["ts_code", "trade_date", "is_paused", "is_st", "high_limit", "low_limit",
                "close", "y", "y_1d", "y_3d", "n1_gap", "n1_ok"] + FEATS
        d = d[[c for c in keep if c in d.columns]]
        for c in FEATS + ["y", "y_1d", "y_3d", "n1_gap"]:
            d[c] = pd.to_numeric(d[c], errors="coerce").astype("float32")
        d.to_parquet(OUT / f"f{y}.parquet", index=False, compression="zstd")
        print(f"{y}: {len(d)} 行 x {len(d.columns)} 列, {time.time()-t:.0f}s", flush=True)

    tot = sum(len(pd.read_parquet(p, columns=["y"])) for p in sorted(OUT.glob("f*.parquet")))
    print(f"完成, 总 {tot} 行", flush=True)


if __name__ == "__main__":
    main()
