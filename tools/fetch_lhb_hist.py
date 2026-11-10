"""补齐龙虎榜历史数据(2016-01 ~ 2025-05), 落盘 data/lhb_hist/{YYYYMMDD}.csv.gz

背景: 现有 data/lhb 来自 tushare lhb_list, 只有 2025-05 起约 1.5 年, 不足以做跨周期验证。
东财 stock_lhb_detail_em 支持任意历史区间, 字段虽无席位明细, 但有净买额/成交额占比/
换手/流通市值/上榜原因, 足够构造强度类因子。

合规: 上榜日是 T 日盘后公布的信息, 与主策略 dip 同口径 (T 日收盘后决策, T+1 开盘买)。
      原始表里的"上榜后1/2/5/10日"是未来收益, 一律不落盘, 避免泄漏。

用法:
  python3 tools/fetch_lhb_hist.py              # 全量补齐(按日落盘)
  python3 tools/fetch_lhb_hist.py 2019-01 2020-01
  python3 tools/fetch_lhb_hist.py --pack       # 日文件合并成 data/lhb_hist.parquet
"""
from __future__ import annotations

import glob
import io
import sys
import time
import gzip
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "lhb_hist"
OUT.mkdir(parents=True, exist_ok=True)

KEEP = {
    "代码": "code",
    "名称": "name",
    "上榜日": "date",
    "收盘价": "close",
    "涨跌幅": "pct",
    "龙虎榜净买额": "net_buy",
    "龙虎榜买入额": "buy_amt",
    "龙虎榜卖出额": "sell_amt",
    "龙虎榜成交额": "lhb_amt",
    "市场总成交额": "mkt_amt",
    "净买额占总成交比": "net_ratio",
    "成交额占总成交比": "amt_ratio",
    "换手率": "turn",
    "流通市值": "circ_mv",
    "上榜原因": "reason",
}


def fetch(start: str, end: str, retry: int = 3) -> pd.DataFrame:
    import akshare as ak

    last = None
    for i in range(retry):
        try:
            df = ak.stock_lhb_detail_em(start_date=start, end_date=end)
            if df is None or len(df) == 0:
                return pd.DataFrame(columns=list(KEEP.values()))
            return df
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 + 2 * i)
    print(f"  !! {start}~{end} 失败: {type(last).__name__} {last}")
    return pd.DataFrame(columns=list(KEEP.values()))


def save_day(df: pd.DataFrame, d: str) -> int:
    """按 code+date 去重(同股多原因只留成交额最大的一条), 写压缩文件。"""
    p = OUT / f"{d}.csv.gz"
    if p.exists() and p.stat().st_size > 100:
        return -1
    sub = df[df["date"] == d].copy()
    if len(sub) == 0:
        return 0
    sub = sub.sort_values("lhb_amt", ascending=False).drop_duplicates("code")
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=6) as f:
        sub.to_csv(f, index=False, float_format="%.4f")
    p.write_bytes(buf.getvalue())
    return len(sub)


def pack() -> None:
    """把 2304 个日文件合并成单个 parquet 入库。

    2304 个小文件逐个 push 到 GitHub 会超时(实测 15 分钟后 connection timed out),
    单个 parquet 7.4MB 一次推完。目录本身进 .gitignore, 只提交 parquet。
    """
    fs = sorted(glob.glob(str(OUT / "*.csv.gz")))
    if not fs:
        print("没有日文件可打包")
        return
    df = pd.concat([pd.read_csv(f, dtype={"code": str}) for f in fs], ignore_index=True)
    df["code"] = df["code"].astype(str).str.zfill(6)
    for c in ("pct", "net_buy", "buy_amt", "sell_amt", "lhb_amt", "mkt_amt",
              "net_ratio", "amt_ratio", "turn", "circ_mv", "close"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    df["date"] = df["date"].astype(str).str[:8]
    df = df.sort_values(["date", "code"]).reset_index(drop=True)
    p = ROOT / "data" / "lhb_hist.parquet"
    df.to_parquet(p, compression="zstd", index=False)
    print(f"已打包 {len(fs)} 个日文件 → {p} ({len(df)} 行, {p.stat().st_size / 1e6:.1f}MB)")


def main() -> None:
    if "--pack" in sys.argv:
        pack()
        return
    months: list[tuple[str, str]] = []
    if len(sys.argv) >= 3:
        a = pd.Period(sys.argv[1], freq="M")
        b = pd.Period(sys.argv[2], freq="M")
    else:
        a = pd.Period("2016-01", freq="M")
        b = pd.Period("2025-06", freq="M")
    for m in pd.period_range(a, b, freq="M"):
        months.append((m.start_time.strftime("%Y%m%d"), m.end_time.strftime("%Y%m%d")))

    tot = 0
    got_days = 0
    for i, (s, e) in enumerate(months, 1):
        df = fetch(s, e)
        if len(df):
            df = df[list(KEEP)].rename(columns=KEEP)
            for c in ("net_buy", "buy_amt", "sell_amt", "lhb_amt", "mkt_amt",
                      "circ_mv", "close", "pct", "net_ratio", "amt_ratio", "turn"):
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df["code"] = df["code"].astype(str).str.zfill(6)
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y%m%d")
            for d, g in df.groupby("date"):
                n = save_day(g, d)
                if n > 0:
                    tot += n
                    got_days += 1
        print(f"[{i}/{len(months)}] {s}~{e} rows={len(df)} 累计{tot}行/{got_days}天", flush=True)
        time.sleep(0.4)

    files = sorted(OUT.glob("*.csv.gz"))
    print(f"完成: {len(files)} 个交易日文件, 区间 {files[0].stem[:8] if files else '-'} ~ "
          f"{files[-1].stem[:8] if files else '-'}")


if __name__ == "__main__":
    main()
