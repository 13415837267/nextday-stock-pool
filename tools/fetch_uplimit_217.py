"""第二百一十七轮(续): 回补 2020 起的全市场涨停明细

数据源: zzshare review_uplimit_reason(date1) —— 一次请求返回当日**全部**题材组,
每组带 stocks 明细。实测 2023-01-03 一次拿到 26 组 / 241 只, 不需要分页。

字段(每只股票):
  plate_code/plate_name/plate_score     题材板块 + 题材热度分(越大越热)
  stock_code/stock_name/stock_price
  up_limit_keep_times / up_limit_desc   连板数 / "7天6板"
  up_limit_type                         板型(一=一字板 等)
  up_limit_time                         首次封板时刻(09:30 等) ★
  reason                                涨停原因文本 ★
  fengdan_volumn/fengdan_money/fengdan_rate   封单量/额/占成交比 ★
  feng_circulation_rate                 封单/流通市值
  actualcirculation_value               实际流通市值
  turnover_ration_real                  真实换手
  market_type                           资金类型标签(游资/机构 等) ★

口径: 全部是 T 日盘后公布, 与主策略"T 日收盘后决策、T+1 开盘买"同口径, 无未来信息。

落盘: data/uplimit/{YYYYMMDD}.csv.gz, 支持断点续跑(已存在则跳过)。
用法: python3 tools/fetch_uplimit_217.py [--start 2020-01-02] [--limit N]
"""
from __future__ import annotations

import argparse
import gzip
import io
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "uplimit"
OUT.mkdir(parents=True, exist_ok=True)
PANEL = ROOT / "data" / "panel"

COLS = [
    "date1", "plate_code", "plate_name", "plate_score",
    "stock_code", "stock_name", "stock_price",
    "up_limit_keep_times", "up_limit_desc", "up_limit_type", "up_limit_time",
    "reason", "fengdan_volumn", "fengdan_money", "fengdan_rate",
    "feng_circulation_rate", "actualcirculation_value", "turnover_ration_real",
    "market_type",
]


def trade_days(start: str) -> list[str]:
    """直接用行情面板的文件名当交易日历 —— 与回测口径严格一致。"""
    days = sorted(p.name[:8] for p in PANEL.glob("*.csv.gz"))
    return [d for d in days if d >= start.replace("-", "")]


def flatten(groups, d: str) -> pd.DataFrame:
    rows = []
    for g in groups or []:
        for s in (g.get("stocks") or []):
            r = dict(s)
            # 组级字段补到个股行(个别行自带, 以组级为准)
            r["plate_code"] = g.get("plate_code")
            r["plate_name"] = g.get("plate_name")
            r["plate_score"] = g.get("plate_score")
            rows.append(r)
    df = pd.DataFrame(rows)
    for c in COLS:
        if c not in df.columns:
            df[c] = None
    df["date1"] = d
    return df[COLS]


def fetch_one(api, d: str, retry: int = 3):
    ds = f"{d[:4]}-{d[4:6]}-{d[6:]}"
    last = None
    for i in range(retry):
        try:
            return api.review_uplimit_reason(date1=ds) or []
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.5 + 2 * i)
    print(f"  !! {d} 失败: {type(last).__name__} {last}")
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2020-01-02")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 天(测速用)")
    ap.add_argument("--sleep", type=float, default=0.28)
    args = ap.parse_args()

    from zzshare.client import DataApi
    api = DataApi()

    days = trade_days(args.start)
    if args.limit:
        days = days[: args.limit]
    todo = [d for d in days if not (OUT / f"{d}.csv.gz").exists()]
    print(f"交易日 {len(days)} 天, 已有 {len(days)-len(todo)} 天, 待补 {len(todo)} 天")

    t0 = time.time()
    fails, empty = [], []
    for i, d in enumerate(todo, 1):
        groups = fetch_one(api, d)
        if groups is None:
            fails.append(d)
            continue
        df = flatten(groups, d)
        if len(df) == 0:
            empty.append(d)
        buf = io.BytesIO()
        with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=6) as f:
            df.to_csv(f, index=False, float_format="%.4f")
        (OUT / f"{d}.csv.gz").write_bytes(buf.getvalue())
        if i % 50 == 0 or i == len(todo):
            el = time.time() - t0
            print(f"  {i}/{len(todo)}  {d}  {el:.0f}s  ({el/max(i,1):.2f}s/天)")
        time.sleep(args.sleep)

    print(f"完成: 成功 {len(todo)-len(fails)} 天, 空 {len(empty)} 天, 失败 {len(fails)} 天")
    if fails:
        print("失败日期:", fails[:20])
    if empty:
        print("空日期(可能休市/接口缺):", empty[:20])


if __name__ == "__main__":
    main()
