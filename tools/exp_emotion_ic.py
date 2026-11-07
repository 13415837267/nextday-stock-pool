# -*- coding: utf-8 -*-
"""第一百六十七轮: 市场情绪指标对"次日可交易收益"的预测力检验。

口径与主策略完全一致:
  T 日收盘算情绪指标(只用 T 日及以前) -> T+1 开盘买 -> T+2 收盘卖, 扣 0.13% 双边费。
  停牌/一字涨停无法买入的样本剔除(与主引擎同规则), 但等权口径下影响极小, 此处仅剔除停牌。

两层检验:
  A. 全市场(1500 日, 统计力强): 情绪分档 -> 次日全市场等权收益, 看单调性与 t 值
  B. 主策略(353 笔): 情绪分档 -> 每笔收益, 看是否只是 A 的影子

判据(先定后算, 防事后挑参数):
  1) A 层 Spearman |rho| >= 0.10 且 分档首尾差 >= 0.3pp 才算有信号;
  2) 单调性: 5 档里至少 4 档同向;
  3) 分年: 正号年份 >= 4/7。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CACHE = os.path.join(ROOT, "data", "cache")


def load_fwd() -> pd.DataFrame:
    """按日聚合全市场等权 T+1 开盘 -> T+2 收盘 收益。

    用 日期x股票 矩阵做 shift(行序即交易日序), 不用 groupby.shift ——
    后者在 8.4M 行切片上出现过 r2 写入后按日 count 只剩 89 的对齐异常(第167轮踩过)。
    """
    p = pd.read_pickle(os.path.join(CACHE, "panel_all.pkl"))
    p = p.loc[p["is_paused"] == 0, ["ts_code", "d", "open", "close", "factor"]]
    p["px_o"] = (p["open"].astype("float64") * p["factor"]).astype("float32")
    p["px_c"] = (p["close"].astype("float64") * p["factor"]).astype("float32")
    mo = p.pivot_table(index="d", columns="ts_code", values="px_o", aggfunc="first")
    mc = p.pivot_table(index="d", columns="ts_code", values="px_c", aggfunc="first")
    r = mc.shift(-2) / mo.shift(-1) - 1.0
    r = r.where((r.abs() <= 0.6) & (mo.shift(-1) > 0) & (mc.shift(-2) > 0))
    daily = pd.DataFrame({"mkt": r.mean(axis=1), "n": r.count(axis=1)}).reset_index()
    print(f"  [fwd] 矩阵 {mo.shape}, 日均 {daily['n'].mean():.0f} 只", flush=True)
    return daily


def main() -> None:
    fwd = load_fwd()
    em = pd.read_pickle(os.path.join(CACHE, "emotion.pkl"))
    df = em.merge(fwd, on="d", how="inner").sort_values("d").reset_index(drop=True)
    df["mkt_net"] = (df["mkt"] - 0.0013) * 100      # 扣双边 0.13%
    print(f"样本 {len(df)} 日  {df['d'].iloc[0]} ~ {df['d'].iloc[-1]}  "
          f"日均 {df['n'].mean():.0f} 只", flush=True)
    print(f"全市场 T+1开盘->T+2收盘: 均值 {df['mkt'].mean()*100:+.3f}%  "
          f"胜率 {(df['mkt']>0).mean()*100:.1f}%  标准差 {df['mkt'].std()*100:.2f}%", flush=True)

    feats = ["zt_n", "zb_rate", "dt_n", "max_board", "b2_n", "zt_prem",
             "up_ratio", "amt_yi", "zt_ratio5"]
    derived = [c for c in df.columns if c.endswith(("_z20", "_ma5", "_ma20"))]
    feats += derived

    print("\n=== A 层: 情绪指标 vs 次日全市场收益(Spearman) ===", flush=True)
    rows = []
    for c in feats:
        s = df[[c, "mkt"]].dropna()
        if len(s) < 200 or s[c].nunique() < 5:
            continue
        rho, p = stats.spearmanr(s[c], s["mkt"])
        # 分 5 档
        q = pd.qcut(s[c], 5, labels=False, duplicates="drop")
        g = s.groupby(q)["mkt"].agg(["mean", "size"])
        spread = (g["mean"].iloc[-1] - g["mean"].iloc[0]) * 100
        mono = np.corrcoef(np.arange(len(g)), g["mean"])[0, 1]
        rows.append(dict(feat=c, rho=round(rho, 3), p=round(p, 4), n=len(s),
                         q1=round(g["mean"].iloc[0] * 100, 3),
                         q5=round(g["mean"].iloc[-1] * 100, 3),
                         spread=round(spread, 3), mono=round(mono, 2)))
    a = pd.DataFrame(rows).sort_values("rho", key=lambda x: -x.abs())
    print(a.to_string(index=False))

    print("\n=== B 层: 情绪指标 vs 主策略每笔收益 ===", flush=True)
    tr = pd.DataFrame(pd.read_pickle(os.path.join(ROOT, "output", "main_trades_latest.pkl"))) \
        if os.path.exists(os.path.join(ROOT, "output", "main_trades_latest.pkl")) else None
    if tr is None:
        print("  (缺少主策略成交明细, 跳过 B 层)")
        return
    tr = tr[tr["ret_pct"].notna()].copy()
    tr["d"] = tr["signal_date"].astype(str).str.replace("-", "").str[:8]
    tb = tr.merge(df, on="d", how="left")
    print(f"  匹配 {tb['mkt'].notna().sum()}/{len(tb)} 笔")
    rows = []
    for c in feats:
        s = tb[[c, "ret_pct"]].dropna()
        if len(s) < 100:
            continue
        rho, p = stats.spearmanr(s[c], s["ret_pct"])
        q = pd.qcut(s[c], 5, labels=False, duplicates="drop")
        g = s.groupby(q)["ret_pct"].agg(["mean", "size"])
        rows.append(dict(feat=c, rho=round(rho, 3), p=round(p, 3), n=len(s),
                         q1=round(g["mean"].iloc[0], 3), q5=round(g["mean"].iloc[-1], 3),
                         spread=round(g["mean"].iloc[-1] - g["mean"].iloc[0], 3)))
    print(pd.DataFrame(rows).sort_values("rho", key=lambda x: -x.abs()).to_string(index=False))

    df.to_pickle(os.path.join(CACHE, "emotion_fwd.pkl"))
    print(f"\n保存 emotion_fwd.pkl  {df.shape}")


if __name__ == "__main__":
    main()
