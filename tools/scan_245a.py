"""第245轮 A: 破位日, 除了抄个股, 还能买什么? —— ETF 的第一手账

244 轮定稿的补位是"破位日单吊跌得最透的主板个股, T+1 开盘买 T+2 收盘卖"。
但它有两个硬约束: ① 占两天资金 ② 每次只能押一只。
ETF 数据(data/etf, 927 只 / 135 万行 / 2012 起)一直没被用上。本轮先算清楚:
  在市场破位日(等权指数距 20 日线 <= -3.51%), 各类 ETF 的
    A. 当日开->收   (T+0 品种可当天买卖, 不占次日资金)
    B. 开->次日收   (与个股补位同口径, 可直接对比)
    C. 收->次日收   (T+1 制度下最宽松的进场)
分别是什么水平, 胜率多少, 分年是否稳定。

若 A 类里债券/黄金/QDII 的日内收益显著为正, 那就是一条与补位**不抢资金**的新腿
(当天平仓, 明天钱还在)。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
FEAT = OUT / "feat234"
ETFD = ROOT / "data" / "etf"

IDX_DIST_MAX = -3.51


def main() -> None:
    # ---- 破位日: 从特征表拿每日 idx_dist ----
    parts = []
    for p in sorted(FEAT.glob("f*.parquet")):
        d = pd.read_parquet(p, columns=["trade_date", "idx_dist"])
        parts.append(d.drop_duplicates("trade_date"))
    idx = pd.concat(parts).drop_duplicates("trade_date").sort_values("trade_date")
    idx["trade_date"] = idx["trade_date"].astype("int64")
    print(f"指数序列 {len(idx)} 天  {idx.trade_date.min()}~{idx.trade_date.max()}")
    brk = idx[idx.idx_dist <= IDX_DIST_MAX]
    print(f"破位日 {len(brk)} 天  (含主策略占用日)")

    # ---- ETF 日线 ----
    e = pd.read_parquet(ETFD / "daily.parquet")
    e["d"] = pd.to_datetime(e["date"]).dt.strftime("%Y%m%d").astype("int64")
    e = e.sort_values(["code", "d"]).reset_index(drop=True)
    u = pd.read_csv(ETFD / "universe.csv")
    u["code"] = u["code"].astype(str)
    e = e.drop(columns=["cls"], errors="ignore")     # 用 universe 的 cls(带名称)
    e["code"] = e["code"].astype(str)
    print(f"ETF {e.code.nunique()} 只 {len(e):,} 行")

    g = e.groupby("code", sort=False)
    e["n_o"] = g["open"].shift(-1)
    e["n2_c"] = g["close"].shift(-2)
    e["n_c"] = g["close"].shift(-1)
    e["y1"] = (e["close"] / e["open"] - 1) * 100          # A 当日开->收
    e["y2"] = (e["n2_c"] / e["n_o"] - 1) * 100            # B 开->次日收
    e["y3"] = (e["n_c"] / e["close"] - 1) * 100           # C 收->次日收
    # 可交易性: 成交额 >= 500 万元(3 万本金进得去出得来)
    e["ok"] = (e["amount"] >= 5e6) & (e["volume"] > 0)

    m = e.merge(u[["code", "name", "cls", "t0"]], on="code", how="left")
    m = m[m["d"].isin(set(brk.trade_date))]
    print(f"破位日 ETF 记录 {len(m):,} 行\n")

    def stat(sub: pd.DataFrame, col: str) -> tuple:
        x = sub[col].dropna()
        if len(x) < 50:
            return len(x), np.nan, np.nan
        return len(x), round(x.mean(), 3), round((x > 0).mean() * 100, 1)

    print(f"{'类型':<10}{'样本口径':<18}{'n':>7}{'均值%':>8}{'胜率%':>7}")
    rows = []
    for cls, cn in (("stock", "股票ETF"), ("qdii", "QDII"), ("bond", "债券ETF"),
                    ("gold", "黄金ETF"), ("money", "货币ETF")):
        sub = m[(m.cls == cls) & m.ok]
        for col, lab in (("y1", "A 当日开->收"), ("y2", "B 开->次日收"), ("y3", "C 收->次日收")):
            n, mu, wr = stat(sub, col)
            print(f"{cn:<10}{lab:<18}{n:>7}{mu:>8.3f}{wr:>7.1f}")
            rows.append({"cls": cn, "口径": lab, "n": n, "均值": mu, "胜率": wr})
    pd.DataFrame(rows).to_csv(OUT / "w245_etf_cls.csv", index=False, encoding="utf-8-sig")

    # ---- 宽基指数 ETF 单独看 ----
    broad = u[u.name.str.contains("沪深300|中证500|中证1000|上证50|创业板ETF|科创50", regex=True)]
    broad = broad[(broad.cls == "stock") & (~broad.t0)]
    bb = m[m.code.isin(set(broad.code)) & m.ok]
    print(f"\n宽基 ETF({broad.code.nunique()} 只) 破位日 {bb.d.nunique()} 天:")
    for col, lab in (("y1", "A 当日开->收"), ("y2", "B 开->次日收"), ("y3", "C 收->次日收")):
        n, mu, wr = stat(bb, col)
        print(f"  {lab:<16}{n:>7}  {mu:>7.3f}  {wr:>5.1f}%")

    # ---- 分年稳定性(只看 B 口径, 与个股补位同口径) ----
    print("\n分年 (B 开->次日收, 宽基 ETF):")
    bb = bb.copy()
    bb["yr"] = bb["d"] // 10000
    t = bb.groupby("yr")["y2"].agg(["size", "mean"]).round(2)
    t["win%"] = bb.groupby("yr")["y2"].apply(lambda x: (x > 0).mean() * 100).round(0)
    print(t.to_string())


if __name__ == "__main__":
    main()
