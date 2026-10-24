# -*- coding: utf-8 -*-
"""择时层数据抓取器(第四十轮, 2026-10-24): 两融账户统计 + 期指基差, 日增量落盘。

数据源(akshare, 免凭据, 全量接口均为~数千行的轻量请求):
  1. stock_margin_account_info  东财"融资融券账户统计"(2012-09起3413+交易日);
     注意PIT: D日数据于D+1早8:30发布, T日收盘选股时最新可用的是T-1日。
  2. futures_main_sina("IF0")   沪深300期指主力连续日线(2017-01起); T日收盘即得。
  3. stock_zh_index_daily("sh000300")  沪深300现货指数, 用于计算基差率。

落盘: data/margin/{YYYYMMDD}.csv.gz 单日一行 + data/futures/{YYYYMMDD}.csv.gz。
日增量: 全量拉取后与本地已有文件diff, 只补缺失日(幂等, 兼作历史回填);
       单次抓取失败抛异常由调用方决定是否容忍, 不影响选股主流程。

变量族(检验见 tools/timing_signal_test.py, 此处只存原始字段):
  margin: rzye/rqye(亿元) rzbuy/rqsell(亿元) collateral(担保物总价值,亿元) mvr(维持担保比例%)
  future: if_close(期指收盘) idx_close(现货收盘) basis_pct(基差率%)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import ROOT

MARGIN_DIR = ROOT / "data" / "margin"
FUTURE_DIR = ROOT / "data" / "futures"

_MARGIN_COLS = {  # 中文列名 -> 规范化英文列名
    "日期": "trade_date", "融资余额": "rzye", "融券余额": "rqye",
    "融资买入额": "rzbuy", "融券卖出额": "rqsell",
    "担保物总价值": "collateral", "平均维持担保比例": "mvr",
}


def _fmt_date(s: str) -> str:
    return str(s).replace("-", "")


def _sync(df: pd.DataFrame, out_dir: Path, rename: dict | None = None) -> int:
    """通用日增量落盘: 单日一行 -> {YYYYMMDD}.csv.gz, 已存在则跳过, 返回新增天数。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    added = 0
    for _, row in df.iterrows():
        d = _fmt_date(row.iloc[0])
        p = out_dir / f"{d}.csv.gz"
        if p.exists():
            continue
        r = row.rename(rename) if rename else row
        pd.DataFrame([r]).to_csv(p, index=False, compression="gzip")
        added += 1
    return added


def sync_margin() -> int:
    """东财两融账户统计全量拉取 + 补缺落盘, 返回新增天数。"""
    import akshare as ak
    df = ak.stock_margin_account_info().rename(columns=_MARGIN_COLS)
    df = df[[c for c in _MARGIN_COLS.values() if c in df.columns]]  # 只保留规范化列
    return _sync(df, MARGIN_DIR)


def sync_future() -> int:
    """期指IF0 + 沪深300现货 -> 基差率, 全量拉取 + 补缺落盘, 返回新增天数。"""
    import akshare as ak
    fut = ak.futures_main_sina(symbol="IF0").rename(columns={
        "日期": "trade_date", "收盘价": "if_close"})
    fut["trade_date"] = fut["trade_date"].map(_fmt_date)
    idx = ak.stock_zh_index_daily(symbol="sh000300").rename(columns={
        "date": "trade_date", "close": "idx_close"})
    idx["trade_date"] = idx["trade_date"].map(_fmt_date)
    m = fut[["trade_date", "if_close"]].merge(
        idx[["trade_date", "idx_close"]], on="trade_date", how="inner")
    m["basis_pct"] = (m["if_close"] / m["idx_close"] - 1) * 100
    return _sync(m, FUTURE_DIR)


def _load_dir(d: Path) -> pd.DataFrame:
    files = sorted(d.glob("*.csv.gz"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(f, dtype={"trade_date": str}) for f in files],
                     ignore_index=True)


def load_margin() -> pd.DataFrame:
    """两融长表(trade_date 升序); 检验/接入时注意 D 日数据 D+1 8:30 才可得。"""
    df = _load_dir(MARGIN_DIR)
    return df.sort_values("trade_date").reset_index(drop=True) if len(df) else df


def load_future() -> pd.DataFrame:
    """期指基差长表(trade_date 升序); T日收盘即得, 无发布时滞。"""
    df = _load_dir(FUTURE_DIR)
    return df.sort_values("trade_date").reset_index(drop=True) if len(df) else df
