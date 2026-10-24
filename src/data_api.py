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

from .config import ROOT, load_config

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
    """zzshare DataApi 薄封装: 节流 + 重试 + 仓库级持久缓存。

    持久缓存设计(data/panel/):
      - 每个交易日一个 csv.gz 文件(全市场快照), 随仓库版本化保存
      - 首次全量下载后, 后续运行仅需增量抓取缺失交易日
      - 兼容旧 pkl 缓存: 命中时自动转存为 csv.gz(本地迁移零成本)
    """

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
        self.panel_dir = ROOT / self.cfg.get("panel_dir", "data/panel")
        self.panel_dir.mkdir(parents=True, exist_ok=True)

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

    def is_trading_day(self, date: str) -> bool:
        """判断某自然日是否为A股交易日(区间查询含未来日期, 可用于当日开盘前判断)。

        注: trade_days(days=N) 只返回已过去的交易日, 不能用于判断"今天";
        单日区间查询(day_start=day_end)则含未来日历, 周末/法定假日返回空。
        """
        d = str(date).replace("-", "")
        if len(d) != 8:
            return False
        ds = f"{d[:4]}-{d[4:6]}-{d[6:]}"
        rows = self._call(self.api.trade_days, day_start=ds, day_end=ds)
        return d in {str(x).replace("-", "") for x in list(rows)}

    def trade_calendar(self, start: str, end: str) -> list[str]:
        """返回[start, end]闭区间内全部A股交易日(升序YYYYMMDD)。

        区间查询含未来日历(trade_days(days=N)只含已过去交易日), 用于推算
        最新信号日的T+1买入日/T+2卖出日。
        """
        def _dash(x: str) -> str:
            x = str(x).replace("-", "")
            return f"{x[:4]}-{x[4:6]}-{x[6:]}" if len(x) == 8 else x
        rows = self._call(self.api.trade_days, day_start=_dash(start), day_end=_dash(end))
        return sorted(str(d).replace("-", "") for d in list(rows))

    # ------------------------------------------------------------------
    _PANEL_DTYPES = {"trade_date": str, "is_paused": "int8", "is_st": "int8"}
    # 数值列合理精度(消除 float 长尾, 显著提升 parquet 压缩率):
    # 价格/涨跌幅/换手 4 位小数足够; 复权因子 8 位有效; 成交额/量取整到分/股
    _ROUND = {
        "open": 4, "high": 4, "low": 4, "close": 4, "pre_close": 4,
        "pct": 4, "turnover_rate": 4, "factor": 8, "amount": 2, "vol": 2,
    }

    def _load_panel(self, date: str) -> pd.DataFrame | None:
        """读取持久缓存(data/panel/{date}.csv.gz); 不存在返回 None。"""
        path = self.panel_dir / f"{date}.csv.gz"
        if not path.exists():
            return None
        return pd.read_csv(path, compression="gzip", dtype=self._PANEL_DTYPES)

    def _save_panel(self, date: str, df: pd.DataFrame):
        """写入持久缓存(每交易日一个 csv.gz, 随仓库版本化)。"""
        df = df.copy()
        for c, nd in self._ROUND.items():  # 统一精度, 提升压缩率
            if c in df.columns:
                df[c] = df[c].round(nd)
        self.panel_dir.mkdir(parents=True, exist_ok=True)
        self.panel_dir.joinpath(f"{date}.csv.gz").unlink(missing_ok=True)
        df.to_csv(self.panel_dir / f"{date}.csv.gz",
                  index=False, compression="gzip")

    def market_day(self, date: str, use_cache: bool = True) -> pd.DataFrame:
        """全市场某交易日快照(fields=all, 自动分页, 持久缓存)。

        缓存优先级: data/panel/{date}.csv.gz(仓库持久层, 随 git 版本化)
                  -> data/cache/market_{date}.pkl(旧缓存, 命中自动迁移)
                  -> 接口请求
        """
        if use_cache:
            cached = self._load_panel(date)
            if cached is not None:
                return cached
            legacy = self.cache_dir / f"market_{date}.pkl"
            if legacy.exists():
                df = pd.read_pickle(legacy)
                self._save_panel(date, df)  # 旧缓存迁移为 csv.gz
                return df
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
        if use_cache and len(df):
            self._save_panel(date, df)  # 空数据(未来日期/非交易日)不落盘
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

    def lhb_day(self, date: str, use_cache: bool = True) -> pd.DataFrame:
        """龙虎榜每日上榜股票概览(lhb_list)。

        注意(实测): date1 参数必须为 YYYY-MM-DD 横杠格式; 数据深度自 2025-05 起
        (更早日期返回空); 收盘后榜单发布有时滞, 当日 15:30 运行可能为空。
        持久缓存 data/lhb/{date}.csv.gz; 空结果不落盘(下次运行自动重试补齐)。
        stock_code 为 6 位纯数字, 此处统一补 ts_code 后缀(.SH/.SZ/.BJ)。
        """
        d = str(date).replace("-", "")
        path = ROOT / "data" / "lhb" / f"{d}.csv.gz"
        if use_cache and path.exists():
            return pd.read_csv(path, compression="gzip", dtype={"stock_code": str})
        ds = f"{d[:4]}-{d[4:6]}-{d[6:]}"
        res = self._call(self.api.lhb_list, date1=ds)
        df = pd.DataFrame(res) if isinstance(res, list) else res
        if df is None or not len(df):
            return pd.DataFrame()

        def _suf(c) -> str:
            c = str(c).split(".")[0].zfill(6)
            if c[0] == "6":
                return c + ".SH"
            if c[0] in ("0", "3"):
                return c + ".SZ"
            return c + ".BJ"
        df = df.copy()
        df["ts_code"] = df["stock_code"].map(_suf)
        df["trade_date"] = d
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False, compression="gzip")
        return df

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
