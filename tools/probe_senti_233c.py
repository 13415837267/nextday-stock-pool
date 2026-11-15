"""第233轮 · 市场情绪指标（涨停/炸板/连板高度）与这套策略的关系

data/limit_hist.parquet 覆盖 4109 个交易日（2009-12 起），是全史口径的
短线情绪表：涨停家数、炸板家数、最高连板、2板以上家数、炸板率。
这套策略本质是"普跌后修复"，与情绪周期（冰点/启动/高潮/退潮）理论上强相关，
但前 232 轮从未接过这张表。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "w233_senti.csv"
VARS = ["zt_cnt", "zb_cnt", "yz_cnt", "max_board", "avg_board", "zt_n2",
        "zb_ratio", "yz_ratio"]


def q5(x: pd.Series, y: pd.Series) -> tuple[float, float, list[float]]:
    ok = x.notna() & y.notna()
    if ok.sum() < 50:
        return np.nan, np.nan, []
    a, b = x[ok].astype("float64"), y[ok].astype("float64")
    pr = float(np.corrcoef(a, b)[0, 1])
    t = pr * np.sqrt((len(a) - 2) / max(1e-12, 1 - pr ** 2))
    bins = pd.qcut(a, 5, labels=False, duplicates="drop")
    return pr, t, [round(float(b[bins == i].mean()), 2) for i in range(bins.max() + 1)]


def main() -> None:
    lim = pd.read_parquet(ROOT / "data" / "limit_hist.parquet")
    lim["date"] = lim["date"].astype(str)
    lim = lim.sort_values("date").reset_index(drop=True)
    # 滚动均值：情绪是状态量，单日噪声大
    for c in ("zt_cnt", "zb_ratio", "max_board"):
        lim[f"{c}_m5"] = lim[c].rolling(5).mean()
    tr = pd.read_csv(ROOT / "output" / "w232_trades.csv", dtype={"signal": str})
    d = tr.merge(lim, left_on="signal", right_on="date", how="left")
    print(f"559 笔匹配情绪表 {d['zt_cnt'].notna().sum()} 笔 "
          f"({lim['date'].min()}~{lim['date'].max()})\n")

    rows = []
    for c in VARS + ["zt_cnt_m5", "zb_ratio_m5", "max_board_m5"]:
        pr, t, means = q5(d[c], d["ret"])
        rows.append({"变量": c, "pearson": round(pr, 3), "t": round(t, 2),
                     **{f"Q{i+1}": v for i, v in enumerate(means)},
                     "极差": round(max(means) - min(means), 2) if means else np.nan})
    r = pd.DataFrame(rows)
    print("== 情绪五分位（单笔 %）==")
    print(r.to_string(index=False))

    d["year"] = d["signal"].str[:4].astype(int)
    print("\n== 炸板率(5日均) 分年：低/中/高三档 ==")
    out = []
    for y, g in d.groupby("year"):
        if len(g) < 25:
            continue
        q = g["zb_ratio_m5"].quantile([1 / 3, 2 / 3]).values
        lo = g[g["zb_ratio_m5"] <= q[0]]["ret"]
        hi = g[g["zb_ratio_m5"] > q[1]]["ret"]
        out.append({"year": y, "n": len(g), "低炸板": round(lo.mean(), 2),
                    "高炸板": round(hi.mean(), 2), "低-高": round(lo.mean() - hi.mean(), 2)})
    print(pd.DataFrame(out).to_string(index=False))

    print("\n== 最高连板(5日均) 分年：低/高 ==")
    out = []
    for y, g in d.groupby("year"):
        if len(g) < 25:
            continue
        q = g["max_board_m5"].quantile([1 / 3, 2 / 3]).values
        lo = g[g["max_board_m5"] <= q[0]]["ret"]
        hi = g[g["max_board_m5"] > q[1]]["ret"]
        out.append({"year": y, "n": len(g), "低位": round(lo.mean(), 2),
                    "高位": round(hi.mean(), 2), "高-低": round(hi.mean() - lo.mean(), 2)})
    print(pd.DataFrame(out).to_string(index=False))

    r.to_csv(OUT, index=False, encoding="utf-8-sig")
    d.to_csv(ROOT / "output" / "w233_senti_joined.csv", index=False, encoding="utf-8-sig")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
