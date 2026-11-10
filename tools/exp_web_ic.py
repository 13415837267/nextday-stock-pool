# -*- coding: utf-8 -*-
"""第184轮·A: 全网文献因子的横截面 IC 体检

为什么先算 IC 再跑策略
----------------------
策略回测一次几十分钟, 而"这个因子当日截面能不能排序"可以用 RankIC 在几分钟内看清。
IC 低的因子直接不进策略测试, 省掉大量无效变体 —— 也避免"跑 20 个组合挑最好的"
那种变相过拟合。

本轮复现的四个文献因子(口径见 src/selector.py 第184轮注释):
  max20 / imax20 / on_abs10 / ctc20, 外加 turn20(=to_ma20) 作对照
  (低换手溢价: 陈慰等 2018、东吴金工 Turn20, 月度 IC 均值 -0.07)

指标: 逐日 Spearman RankIC 的均值 / 标准差 / ICIR / t 值 / 胜率, 对未来 1/5/10 日收益。
  符号约定: IC<0 表示"因子值越小, 未来收益越高" —— 对我们要选的排序键而言,
  IC 为负的因子用升序取第 1 名, IC 为正的用降序。

用法: python tools/exp_web_ic.py --days 1500
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context  # noqa: E402
from src.config import load_config  # noqa: E402

FACTORS = ["ivol20", "mom20", "max20", "imax20", "on_abs10", "ctc20", "to_ma20",
           "turnover_cool", "vwap_dev", "dist_high20"]
HORIZONS = [1, 5, 10]


def _rank_ic(df: pd.DataFrame, fac: str, fwd: str) -> pd.Series:
    """逐日 RankIC(pct rank 后的 Pearson, 等价 Spearman), 全向量化。"""
    d = df["trade_date"]
    a = df.groupby(d)[fac].rank(pct=True)
    b = df.groupby(d)[fwd].rank(pct=True)
    ok = a.notna() & b.notna()
    if not ok.any():
        return pd.Series(dtype=float)
    a, b, d = a[ok], b[ok], d[ok]
    m_a = a.groupby(d).transform("mean")
    m_b = b.groupby(d).transform("mean")
    cov = (a * b).groupby(d).transform("mean") - m_a * m_b
    va = (a * a).groupby(d).transform("mean") - m_a ** 2
    vb = (b * b).groupby(d).transform("mean") - m_b ** 2
    ic = cov / np.sqrt((va * vb).clip(lower=1e-12))
    return ic.groupby(d).first()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=1500)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cfg = load_config()
    ctx = prepare_context(cfg, args.days)
    by_date = ctx["by_date"]
    print(f"[ic] 面板 {len(by_date)} 日")

    df = pd.concat([v for v in by_date.values()], ignore_index=True)
    df = df[(df["is_paused"] == 0) & (df["close"] > 0)]
    codes = df["ts_code"]
    ca = df["close_adj"] if "close_adj" in df.columns else df["close"] * df["factor"]
    for h in HORIZONS:
        fwd = ca.groupby(codes, sort=False).shift(-h) / ca - 1
        df[f"fwd{h}"] = fwd.where(fwd.abs() < 0.8) * 100   # 剔除复权脏点
    df["trade_date"] = df["trade_date"].astype(str)

    out = {}
    for fac in FACTORS:
        if fac not in df.columns:
            print(f"[ic] 缺列 {fac}, 跳过")
            continue
        row = {}
        for h in HORIZONS:
            ic = _rank_ic(df, fac, f"fwd{h}")
            if not len(ic):
                continue
            mu, sd = float(ic.mean()), float(ic.std())
            row[f"h{h}"] = {"ic": round(mu, 4), "icir": round(mu / sd * np.sqrt(252), 2)
                            if sd > 0 else None,
                            "t": round(mu / sd * np.sqrt(len(ic)), 2) if sd > 0 else None,
                            "win": round(float((ic < 0).mean()) * 100, 1), "n": int(len(ic))}
        out[fac] = row
        h10 = row.get("h10", {})
        print(f"{fac:14s} IC10={h10.get('ic'):+.4f} ICIR10={h10.get('icir'):+.2f} "
              f"t={h10.get('t'):+.1f} 负向占比={h10.get('win')}%")

    path = args.out or f"output/exp_web_ic_{args.days}.json"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"days": args.days, "factors": out,
                                      "sample_rows": int(len(df))}, ensure_ascii=False, indent=1))
    print(f"[ic] 写出 {path}")


if __name__ == "__main__":
    main()
