"""第254轮 B: 尾部风险能不能事前识别 —— 给每笔交易配上"信号日的波动率"

253 的结论是回撤由少数几笔决定: 22 笔单笔 <=-5% 合计 -162pp, 其余 473 笔
加起来才是正收益。既然"哪一笔会暴雷"不可预测(许继电气 -10.4%、思美传媒 -11.5%
都是两天内的跌停级下跌), 唯一可控的是**仓位**。

那么问题变成: 仓位该不该与"这笔的预期波动"挂钩? 单吊满仓的隐含假设是
"所有信号的赔率相同", 但高波动票的收益分布更肥尾 —— 均值可能一样, 方差差一倍。

本脚本给 495 笔主策略交易配上信号日的三个波动度量, 看:
  a) 波动与收益均值的关系(若无关, 降仓只减波动不减收益 = 免费提升卡玛)
  b) 波动与尾部的关系(<=-5% 的笔是不是集中在高波动组)
  c) 波动与缺口的交互

波动度量(信号日 T 收盘可知, 无未来函数):
  amp20  = 过去 20 日 (high-low)/pre_close 的均值
  std20  = 过去 20 日日收益标准差
  atr14  = 过去 14 日真实波幅 / close
"""
from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context  # noqa: E402
from src.config import load_config  # noqa: E402


def main() -> None:
    cfg = load_config()
    ctx = prepare_context(cfg, 2500)
    per_code = ctx["per_code"]

    d = {o["name"]: o for o in pickle.load(open(ROOT / "output/w253a_pooltier_2500.pkl", "rb"))}
    tr = pd.DataFrame(d["B 1.0/1.0/0"]["trades"])
    tr = tr[tr.ret_pct.notna()]
    m = tr[~tr._fillin.fillna(False).astype(bool)].copy()

    # _PerCodeView 不支持切片, 逐行取后拼成小表(每笔最多 20 行, 495 笔 -> 万次查询)
    rows = []
    for k, r in enumerate(m.itertuples()):
        v = per_code.get(r.code)
        if v is None or r.signal_date not in v.index:
            rows.append((np.nan, np.nan, np.nan))
            continue
        i = v.index.get_loc(r.signal_date)
        lo = max(0, i - 19)
        recs = [v.iloc[j] for j in range(lo, i + 1)]
        w = pd.DataFrame(recs)
        pc = w["pre_close"].replace(0, np.nan).astype(float)
        amp = float((((w["high"] - w["low"]).astype(float)) / pc).mean()) * 100
        px = w["close"].astype(float).pct_change().dropna()
        std = float(px.std()) * 100 if len(px) > 5 else np.nan
        w14 = w.iloc[-14:]
        pc14 = w14["pre_close"].astype(float)
        tr_ = np.maximum((w14["high"] - w14["low"]).astype(float),
                         np.maximum((w14["high"].astype(float) - pc14).abs(),
                                    (w14["low"].astype(float) - pc14).abs()))
        atr = float(tr_.mean() / max(float(w14["close"].astype(float).iloc[-1]), 1e-9)) * 100
        rows.append((amp, std, atr))
        if (k + 1) % 100 == 0:
            print(f"  {k + 1}/{len(m)}", flush=True)
    m[["amp20", "std20", "atr14"]] = pd.DataFrame(rows, index=m.index)

    out = ROOT / "output" / "w254_tail.pkl"
    with open(out, "wb") as f:
        pickle.dump(m, f)

    print(f"主策略 {len(m)} 笔, 波动缺失 {int(m.amp20.isna().sum())}")
    print("\n=== 按 amp20 五档 ===")
    m["ab"] = pd.qcut(m.amp20, 5, labels=["Q1 低", "Q2", "Q3", "Q4", "Q5 高"])
    g = m.groupby("ab", observed=True).agg(
        n=("ret_pct", "size"), avg=("ret_pct", "mean"), med=("ret_pct", "median"),
        win=("ret_pct", lambda s: (s > 0).mean() * 100),
        sd=("ret_pct", "std"), worst=("ret_pct", lambda s: s.min()),
        tail=("ret_pct", lambda s: (s <= -5).sum()))
    g["sharpe"] = (g["avg"] / g["sd"]).round(3)
    print(g.round(2).to_string())

    print("\n=== 按 atr14 五档 ===")
    m["tb"] = pd.qcut(m.atr14, 5, labels=["Q1 低", "Q2", "Q3", "Q4", "Q5 高"])
    g2 = m.groupby("tb", observed=True).agg(
        n=("ret_pct", "size"), avg=("ret_pct", "mean"),
        win=("ret_pct", lambda s: (s > 0).mean() * 100),
        sd=("ret_pct", "std"), worst=("ret_pct", lambda s: s.min()),
        tail=("ret_pct", lambda s: (s <= -5).sum()))
    g2["sharpe"] = (g2["avg"] / g2["sd"]).round(3)
    print(g2.round(2).to_string())

    print("\n=== 波动 x 缺口(均值) ===")
    m["gb"] = pd.cut(m.gap_pct, [-99, -1.5, -0.5, 0.5, 99], labels=["<=-1.5", "-1.5~-0.5", "-0.5~0.5", ">0.5"])
    print(m.pivot_table(index="ab", columns="gb", values="ret_pct", aggfunc="mean", observed=True).round(2).to_string())
    print("\n笔数:")
    print(m.pivot_table(index="ab", columns="gb", values="ret_pct", aggfunc="size", observed=True).to_string())
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
