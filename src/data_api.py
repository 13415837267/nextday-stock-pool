# -*- coding: utf-8 -*-
"""数据层: 封装 zzshare 免费公开接口(限速/重试/缓存/字段标准化)。

字段标准化后的统一列:
  ts_code, date(YYYYMMDD str), open, high, low, close, pre_close, pct(%),
  vol(股), amount(元), factor, high_limit, low_limit, turnover_rate(%),
  is_paused, is_st
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd

from .config import load_config

# zzshare fields='all' 返回列 -> 统一列名
_RENAME = {
    "volume": "vol",
    "turnover": "amount",
    "quote_rate": "pct",
    "prev_close": "pre_close",
}
_KEEP = [
    "ts_code", "trade_date", "open", "high", "low", "close", "pre_close",
    "pct", "vol", "amount", "factor", "high_limit", "low_limit",
    "turnover_rate", "is_paused", "is_st",
]
_FLOAT_COLS = [
    "open", "high", "low", "close", "pre_close", "pct", "vol", "amount",
    "factor", "high_limit", "low_limit", "turnover_rate",
]

_MARKET_PAGE = 6000      # 全市场单次建议行数
_STOCK_PAGE = 1000       # 单股单次上限


class ZZClient:
    """zzshare DataApi 薄封装: 节流 + 重试 + 本地缓存。"""

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or load_config()["data"]
        from zzshare.client import DataApi
        token = os.environ.get(self.cfg.get("token_env", "ZZSHARE_TOKEN"), "").strip()
        self.api = DataApi(token=token or None)
        self._min_interval = float(self.cfg.get("min_interval_sec", 0.45))
        self._last_call = 0.0
        self._max_retries = int(self.cfg.get("max_retries", 5))
        self.cache_dir = Path(self.cfg.get("cache_dir", "data/cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def _throttle(self):
        wait = self._min_interval - (time.time() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.time()

    def _call(self, fn, **kwargs) -> pd.DataFrame:
        last_err: Exception | None = None
        for i in range(self._max_retries):
            try:
                self._throttle()
                df = fn(**kwargs)
                if df is None:
                    return pd.DataFrame()
                return df
            except Exception as e:  # noqa: BLE001 网络类异常统一重试
                last_err = e
                time.sleep(min(2 ** i, 20))
        raise RuntimeError(f"zzshare 请求失败(已重试{self._max_retries}次): {last_err}")

    @staticmethod
    def _normalize(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or len(df) == 0:
            return pd.DataFrame(columns=_KEEP)
        df = df.rename(columns=_RENAME)
        missing = [c for c in _KEEP if c not in df.columns]
        if missing:
            raise RuntimeError(f"接口返回缺少字段: {missing}, 实际列: {list(df.columns)}")
        df = df[_KEEP].copy()
        df["trade_date"] = df["trade_date"].astype(str).str.replace("-", "")
        for c in _FLOAT_COLS:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
        for c in ("is_paused", "is_st"):
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("int8")
        return df.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)

    # ------------------------------------------------------------------
    def trade_days(self, days: int = 800) -> list[str]:
        """返回最近N个交易日, 升序, 格式 YYYYMMDD。

        注: zzshare 的 days 参数为自然日, 此处按 1.7 倍换算并校验数量。
        """
        need = days
        cal = int(need * 1.7) + 40
        for _ in range(3):
            rows = self._call(self.api.trade_days, days=cal)
            out = sorted(str(d).replace("-", "") for d in list(rows))
            if len(out) >= need:
                return out[-need:]
            cal = int(cal * 1.5)
        return out[-need:] if len(out) > need else out

    def market_day(self, date: str, use_cache: bool = True) -> pd.DataFrame:
        """全市场某交易日快照(fields=all, 自动分页, 本地缓存)。date: YYYYMMDD"""
        cache = self.cache_dir / f"market_{date}.pkl"
        if use_cache and cache.exists():
            return pd.read_pickle(cache)
        frames, offset = [], 0
        while True:
            part = self._call(
                self.api.daily, trade_date=date, offset=offset,
                limit=_MARKET_PAGE, fields="all",
            )
            part = self._normalize(part)
            frames.append(part)
            if len(part) < _MARKET_PAGE:
                break
            offset += _MARKET_PAGE
            if offset > 20000:  # 安全上限
                break
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        df = df.drop_duplicates(subset=["ts_code"]).reset_index(drop=True)
        if use_cache:
            df.to_pickle(cache)
        return df

    def stock_history(self, ts_code: str, start: str, end: str) -> pd.DataFrame:
        """单股区间日线(>=1000条自动分页)。"""
        frames, offset = [], 0
        while True:
            part = self._call(
                self.api.daily, ts_code=ts_code, start_date=start, end_date=end,
                offset=offset, limit=_STOCK_PAGE, fields="all",
            )
            part = self._normalize(part)
            frames.append(part)
            if len(part) < _STOCK_PAGE:
                break
            offset += _STOCK_PAGE
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def stock_names(self) -> pd.DataFrame:
        """全市场代码-名称表(带7天缓存)。"""
        cache = self.cache_dir / "stock_names.pkl"
        if cache.exists() and time.time() - cache.stat().st_mtime < 7 * 86400:
            return pd.read_pickle(cache)
        df = self._call(
            self.api.stock_basic, exchange="ALL", list_status="L",
            fields="ts_code,symbol,name,exchange,list_status",
        )
        df.to_pickle(cache)
        return df


# ----------------------------------------------------------------------
def add_adjusted(hist: pd.DataFrame) -> pd.DataFrame:
    """按复权因子计算前复权价格列(open/close/high/low -> *_adj)。

    前复权锚定该序列最后一个交易日: adj = price * factor / factor_last。
    用于均线/动量等指标计算, 避免分红除权跳空造成假信号。
    """
    hist = hist.sort_values("trade_date")
    flast = hist["factor"].iloc[-1]
    if not flast or flast <= 0:
        flast = 1.0
    k = hist["factor"] / flast
    for c in ("open", "high", "low", "close"):
        hist[f"{c}_adj"] = hist[c] * k
    return hist
