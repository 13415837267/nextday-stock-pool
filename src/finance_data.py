# -*- coding: utf-8 -*-
"""PIT财务数据层(第四十一轮, 2026-10-24): 东财业绩报表按报告期落盘。

数据源(akshare, 免凭据): stock_yjbb_em(date=报告期)——该期全市场业绩快照
(~1.1万行/期), 含"最新公告日期"(PIT锚点)、营收同比、净利同比、ROE、毛利率。

落盘: data/finance/{报告期YYYYMMDD}.csv.gz 每期一文件; 全量拉取+已存在跳过(幂等)。

PIT口径: T日选股可用 = 报告期季末<=T 且 公告日<=T 的最新一期。
注意: 东财"最新公告日期"为公司对应该期的公告日(非任意公告), 用于PIT近似可靠;
    若公告日缺失回退用报告期季末+45天(法定披露截止)兜底, 宁晚勿早。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import ROOT

FINANCE_DIR = ROOT / "data" / "finance"

_COLS = {  # 中文列名 -> 规范化英文列名
    "股票代码": "code6", "最新公告日期": "ann_date",
    "营业总收入-同比增长": "rev_yoy", "净利润-同比增长": "np_yoy",
    "净资产收益率": "roe", "销售毛利率": "gross_margin",
}


def _to_ts_code(c6: str) -> str:
    c = str(c6).zfill(6)
    if c.startswith(("6", "9")):
        return f"{c}.SH"
    if c.startswith(("0", "2", "3")):
        return f"{c}.SZ"
    return f"{c}.BJ"  # 4/8/920开头北交所


def sync_finance(periods: list[str] | None = None) -> int:
    """按报告期全量拉取+落盘, 已存在跳过, 返回新增期数。

    periods 缺省回填 2016Q4 至今(覆盖十年回测窗口), 每季末。
    """
    import akshare as ak
    if periods is None:
        periods = []
        p = "20161231"
        end = pd.Timestamp.today().strftime("%Y%m%d")
        while p <= end:
            periods.append(p)
            y, m = int(p[:4]), p[4:6]
            y, m = (y + 1, "03") if m == "12" else (y, {"03": "06", "06": "09", "09": "12"}[m])
            p = f"{y}{m}31" if m != "06" else f"{y}0630"
    FINANCE_DIR.mkdir(parents=True, exist_ok=True)
    added = 0
    for per in periods:
        out = FINANCE_DIR / f"{per}.csv.gz"
        if out.exists():
            continue
        try:
            df = ak.stock_yjbb_em(date=per)
        except Exception:  # noqa: BLE001 某期不可得(接口未覆盖该年/空)则跳过
            continue
        df = df.rename(columns=_COLS)
        df = df[[c for c in _COLS.values() if c in df.columns]]
        df["ts_code"] = df["code6"].map(_to_ts_code)
        df["period"] = per
        df.to_csv(out, index=False, compression="gzip")
        added += 1
    return added


def load_finance() -> pd.DataFrame:
    """全部报告期长表(period, ts_code, ann_date, rev_yoy, np_yoy, roe, gross_margin)。"""
    files = sorted(FINANCE_DIR.glob("*.csv.gz"))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(f, dtype={"code6": str, "period": str}) for f in files],
                   ignore_index=True)
    df["ann_date"] = pd.to_numeric(df["ann_date"], errors="coerce")
    # 公告日缺失兜底: 报告期季末+45天(法定披露截止), 宁晚勿早
    df["ann_date"] = df["ann_date"].fillna(
        (pd.to_datetime(df["period"], format="%Y%m%d") + pd.Timedelta(days=45))
        .dt.strftime("%Y%m%d").astype(float))
    return df.sort_values(["ts_code", "period"]).reset_index(drop=True)


def pit_lookup(df_fin: pd.DataFrame) -> pd.DataFrame:
    """生成PIT查找表: (ts_code, as_of_date) -> 最新一期财务。

    返回以 ann_date 生效的长表(同票多期按公告日排序, 取 as_of>=ann_date 的最后一期)。
    """
    out = df_fin.dropna(subset=["ann_date"]).copy()
    out["ann_date"] = out["ann_date"].astype(int).astype(str)
    return out.sort_values(["ts_code", "ann_date", "period"]).reset_index(drop=True)


def pit_row(pit: pd.DataFrame, ts_code: str, as_of: str) -> pd.Series | None:
    """as_of日(YYYYMMDD)该票已公告的最新一期财务行; 无则None。"""
    sub = pit[(pit["ts_code"] == ts_code) & (pit["ann_date"] <= str(as_of))]
    return sub.iloc[-1] if len(sub) else None
