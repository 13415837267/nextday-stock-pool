# -*- coding: utf-8 -*-
"""第 212 轮 · 左侧"加速暴跌反转"策略回测: 深跌股里选"跌得最急最恐慌"的那只。

深跌池内因子扫描已定位: mom20极负 / bias20极负 / 高波动 / 当日大跌 的股票次日反弹最强(+2~3%)。
本脚本把"加速暴跌"画像做成可交易的"每日选1只"规则, 复利回测并分年检验。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"


def backtest(p: pd.DataFrame, cond, key, ascending, name: str) -> dict:
    sub = p[cond].copy()
    sub = sub.sort_values(["d", key], ascending=[True, ascending], kind="mergesort")
    pick = sub.groupby("d").head(1)
    r = pick.groupby("d")["ret12_net"].first()
    nav = (1 + r / 100).prod()
    years = p["d"].nunique() / 242
    cagr = nav ** (1 / years) - 1 if nav > 0 else -1
    curve = (1 + r / 100).cumprod()
    dd = (curve / curve.cummax() - 1).min()
    return {"name": name, "trades": len(r), "mult": round(float(nav), 1),
            "cagr%": round(float(cagr) * 100, 1), "per%": round(float(r.mean()), 3),
            "win%": round(float((r > 0).mean()) * 100, 1),
            "maxdd%": round(float(dd) * 100, 1),
            "kama": round(float(cagr) / abs(float(dd)) if dd < 0 else 99, 2)}


def main() -> None:
    p = pd.read_parquet(OUT / "panel_212.parquet")
    p["r"] = p["ret12_net"].clip(-21, 21)

    # 组合评分: 越跌越急(负mom20、负bias20) + 越恐慌(高波动vol20、低close_pos)
    p["z_mom20"] = -p["mom20"]
    p["z_bias20"] = -p["bias20"]
    p["crash_score"] = (-p["mom20"] + -p["bias20"] + p["vol20"])

    deep = p["dist_ph250"] < -55
    results = [
        backtest(p, deep & (p["mom20"] < -30), "mom20", True, "深跌<-55·mom20<-30·选mom20最低"),
        backtest(p, deep & (p["bias20"] < -20), "bias20", True, "深跌<-55·bias20<-20·选bias20最低"),
        backtest(p, deep & (p["mom20"] < -30), "vol20", False, "深跌<-55·mom20<-30·选vol20最高"),
        backtest(p, deep & (p["mom20"] < -30), "crash_score", False, "深跌<-55·mom20<-30·选crash_score"),
        backtest(p, deep & (p["mom20"] < -20), "mom20", True, "深跌<-55·mom20<-20·选mom20最低"),
        backtest(p, deep & (p["mom20"] < -20), "crash_score", False, "深跌<-55·mom20<-20·选crash_score"),
        backtest(p, deep & (p["mom20"] < -40), "mom20", True, "深跌<-55·mom20<-40·选mom20最低"),
        backtest(p, deep & (p["mom20"] < -30) & (p["mkt_ret"] < 0), "mom20", True,
                 "深跌·mom20<-30·大盘跌·选mom20最低"),
        backtest(p, deep & (p["mom20"] < -30) & (p["amp_pct"] > 8), "mom20", True,
                 "深跌·mom20<-30·巨震·选mom20最低"),
    ]
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    df.to_csv(OUT / "crash_bt_212.csv", index=False)

    # 最佳变体分年
    print("\n==== 深跌<-55·mom20<-30·选mom20最低 · 分年 ====")
    sub = p[deep & (p["mom20"] < -30)].sort_values(["d", "mom20"])
    pick = sub.groupby("d").head(1)
    pick["year"] = pick["d"].str[:4]
    yr = pick.groupby("year").agg(n=("r", "size"), mean=("r", "mean"),
                                  win=("r", lambda s: (s > 0).mean() * 100)).round(2)
    print(yr.to_string())


if __name__ == "__main__":
    main()
