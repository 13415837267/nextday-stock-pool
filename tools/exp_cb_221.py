#!/usr/bin/env python3
"""第221轮: 可转债 T+0 日内反转 —— 严格回测。

交易设定(真实可执行, 不是纸面):
  T 日 09:30 看到开盘价 -> 算 gap = open/prev_close - 1
  选 gap 最深的一只 -> 开盘价买入 -> 当日收盘价卖出(T+0, 无隔夜风险)
  单次往返成本 0.09%(可转债无印花税, 佣金万0.5双边 + 滑点)

三个关键对照, 用来判断"净值到底来自信号还是来自运气":
  A. 同一套规则但选 gap 最浅的(反向下注) —— 若仍赚钱说明规则无意义
  B. 随机选一只 —— 零假设基准
  C. 每天固定做 vs 只在深缺口日做 —— 检验阈值是否必要

所有口径含成本, 且同时看 全史 / 近3年。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COST = 0.0009          # 单次往返 0.09%
DAYS = 244.0


def load() -> pd.DataFrame:
    d = pd.read_parquet(ROOT / "data" / "cb" / "daily.parquet")
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values(["code", "date"]).reset_index(drop=True)
    g = d.groupby("code", observed=True, sort=False)
    d["pc"] = g["close"].shift(1)
    d["gap"] = d["open"] / d["pc"] - 1
    d["intr"] = d["close"] / d["open"] - 1
    d["r1"] = d["close"] / d["pc"] - 1
    d["vol20"] = g["r1"].transform(lambda x: x.rolling(20).std())
    d["amt"] = d["close"] * d["volume"]
    d = d[(d["gap"].between(-0.35, 0.35)) & (d["open"] > 50) & (d["pc"] > 50)]
    return d


def metrics(rets: list[float], dates: list, nav_start: float = 1.0) -> dict:
    if not rets:
        return {}
    nav = nav_start
    peak, mdd = nav, 0.0
    curve = []
    for r in rets:
        nav *= 1 + r
        peak = max(peak, nav)
        mdd = min(mdd, nav / peak - 1)
        curve.append(nav)
    d0, d1 = pd.Timestamp(str(dates[0])), pd.Timestamp(str(dates[-1]))
    yrs = max(0.5, (d1 - d0).days / 365.25)
    ann = nav ** (1 / yrs) - 1
    return {
        "笔数": len(rets),
        "年化%": round(ann * 100, 1),
        "总倍": round(nav, 1),
        "回撤%": round(mdd * 100, 1),
        "卡玛": round(ann / abs(mdd), 2) if mdd else None,
        "单笔%": round(float(np.mean(rets)) * 100, 3),
        "胜率%": round(float(np.mean([x > 0 for x in rets]) * 100), 1),
        "年数": round(yrs, 2),
    }


def run(d: pd.DataFrame, mode: str, thresh: float | None, cost: float = COST,
        vol_max: float | None = None, amt_min: float = 0.0,
        rng: np.random.Generator | None = None) -> dict:
    """mode: deep(选最深) / shallow(选最浅) / random(随机)"""
    x = d.dropna(subset=["gap", "intr"])
    x = x[x["gap"].between(-0.35, 0.35)]
    if amt_min > 0:
        x = x[x["amt"] >= amt_min]
    if vol_max is not None:
        x = x[x["vol20"].notna() & (x["vol20"] <= vol_max)]
    if thresh is not None:
        x = x[x["gap"] <= thresh]

    rets, dates = [], []
    for dt, g in x.groupby("date", sort=True):
        if g.empty:
            continue
        if mode == "deep":
            row = g.loc[g["gap"].idxmin()]
        elif mode == "shallow":
            row = g.loc[g["gap"].idxmax()]
        else:
            row = g.iloc[rng.integers(len(g))] if rng is not None else g.iloc[0]
        rets.append(float(row["intr"]) - cost)
        dates.append(int(dt.strftime("%Y%m%d")))
    return metrics(rets, dates)


def main() -> None:
    d = load()
    print(f"面板 {len(d):,} 行, {d.code.nunique()} 券, {d.date.min().date()} ~ {d.date.max().date()}")
    res: dict = {"面板": {"行": int(len(d)), "券": int(d.code.nunique()),
                          "起": str(d.date.min().date()), "止": str(d.date.max().date())}}

    # 分窗口
    last = d["date"].max()
    cut3 = last - pd.Timedelta(days=365 * 3)
    windows = {"全史": d, "近3年": d[d["date"] >= cut3]}

    for wn, dw in windows.items():
        print(f"\n{'='*76}\n【{wn}】{dw.date.min().date()} ~ {dw.date.max().date()}")
        rng = np.random.default_rng(42)
        table = {}
        tests = [
            ("A 每天做/选最深", dict(mode="deep", thresh=None)),
            ("B 每天做/选最浅(反向)", dict(mode="shallow", thresh=None)),
            ("C 每天做/随机选", dict(mode="random", thresh=None)),
            ("D gap<-1% 选最深", dict(mode="deep", thresh=-0.01)),
            ("E gap<-1.5% 选最深", dict(mode="deep", thresh=-0.015)),
            ("F gap<-2% 选最深", dict(mode="deep", thresh=-0.02)),
            ("G gap<-1% 且波动<1.5%", dict(mode="deep", thresh=-0.01, vol_max=0.015)),
            ("H gap<-1% 且额>2000万", dict(mode="deep", thresh=-0.01, amt_min=2e7)),
            ("I gap<-1% 零成本(上限)", dict(mode="deep", thresh=-0.01, cost=0.0)),
        ]
        for name, kw in tests:
            table[name] = run(dw, rng=rng, **kw)
        res[wn] = table

        hdr = f"{'方案':<24}{'笔数':>6}{'年化%':>9}{'总倍':>10}{'回撤%':>8}{'卡玛':>7}{'单笔%':>8}{'胜率%':>7}"
        print(hdr)
        for k, v in table.items():
            if not v:
                continue
            print(f"{k:<24}{v['笔数']:>6}{v['年化%']:>9.1f}{v['总倍']:>10.1f}"
                  f"{v['回撤%']:>8.1f}{str(v['卡玛']):>7}{v['单笔%']:>8.3f}{v['胜率%']:>7.1f}")

    (ROOT / "output").mkdir(exist_ok=True)
    (ROOT / "output" / "exp_cb_221.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nsaved output/exp_cb_221.json")


if __name__ == "__main__":
    main()
