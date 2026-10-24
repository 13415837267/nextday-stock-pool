# -*- coding: utf-8 -*-
"""择时层变量增量预测力检验(第四十轮, 事前注册)。

事前注册变量族(定义固定不调参, 五分位分层):
  margin(D日数据D+1早8:30发布, 恰好赶上D+1开盘决策, 无需额外shift):
    m_chg5      融资余额5日变化率%
    m_chg20     融资余额20日变化率%
    rzbuy_chg5  融资买入额5日变化率%
    mvr_dev     维持担保比例偏离(当日-250日均值)
  future(T日收盘即得, 显式shift到T+1):
    basis       基差率%(期指相对现货)
    basis_chg5  基差率5日变化(斜率)

因变量: 沪深300次日收益(现货close-to-close, 与全市场等权方向高度一致)。
增量口径: 现有风控=大盘趋势(000300 vs MA20代理trade_ok), 检验变量在
"趋势内"(风控放行日)是否仍有区分度——只区分趋势方向不算增量。
era一致性: early(<20240901) vs recent(>=20240901), top-bottom差须同号。
初步筛查口径, 全样本分位存在轻前视, 通过者接入前须改expanding分位。
"""
from __future__ import annotations

import sys
sys.path.insert(0, "/workspace")

import numpy as np
import pandas as pd

from src.timing_data import load_margin, load_future

ERA_SPLIT = "20240901"


def _quintile_test(x: pd.Series, y: pd.Series, era: pd.Series, label: str) -> None:
    """五分位分层: q1..q5各层次日收益 + top-bottom差(全期/两era) + 趋势内增量。"""
    ok = x.notna() & y.notna()
    x, y = x[ok], y[ok]
    q = pd.qcut(x, 5, labels=False, duplicates="drop")
    rows = [f"  {label:12s} q均值%(1→5): " + " ".join(f"{y[q==i].mean():+.3f}" for i in range(5))]
    top, bot = y[q == 4], y[q == 0]
    diff = top.mean() - bot.mean()
    se = np.sqrt(top.var() / len(top) + bot.var() / len(bot))
    t = diff / se if se > 0 else 0.0
    # era一致性
    e_early = era.reindex(y.index) == "early"
    d_early = y[e_early & (q == 4)].mean() - y[e_early & (q == 0)].mean()
    d_recent = y[~e_early & (q == 4)].mean() - y[~e_early & (q == 0)].mean()
    consistent = "一致" if (d_early > 0) == (d_recent > 0) else "不一致"
    rows.append(f"  {'':12s} top-bot={diff:+.3f}% t={t:.2f} | "
                f"early={d_early:+.3f} recent={d_recent:+.3f} era={consistent}")
    # 趋势内增量(现有风控代理: 000300>MA20 的日子)
    tr = y.index.isin(TREND_DAYS)
    top_in, bot_in = y[tr & (q == 4)], y[tr & (q == 0)]
    if len(top_in) > 30 and len(bot_in) > 30:
        d_in = top_in.mean() - bot_in.mean()
        se_in = np.sqrt(top_in.var() / len(top_in) + bot_in.var() / len(bot_in))
        rows.append(f"  {'':12s} 趋势内({int(tr.sum())}日) top-bot={d_in:+.3f}% "
                    f"t={d_in/se_in if se_in>0 else 0:.2f} n_per_q≈{int((tr&(q==4)).sum())}")
    else:
        rows.append(f"  {'':12s} 趋势内样本不足")
    print("\n".join(rows))


def build_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    m = load_margin()
    m["trade_date"] = m["trade_date"].astype(str).str.replace("-", "", regex=False)
    m = m.set_index("trade_date").sort_index()
    m["m_chg5"] = m["rzye"].pct_change(5) * 100
    m["m_chg20"] = m["rzye"].pct_change(20) * 100
    m["rzbuy_chg5"] = m["rzbuy"].pct_change(5) * 100
    m["mvr_dev"] = m["mvr"] - m["mvr"].rolling(250).mean()

    f = load_future()
    f["trade_date"] = f["trade_date"].astype(str)
    f = f.sort_values("trade_date").reset_index(drop=True).set_index("trade_date")
    f["ret"] = f["idx_close"].pct_change().shift(-1) * 100  # T+1现货日收益
    f["basis"] = f["basis_pct"]
    f["basis_chg5"] = f["basis_pct"].diff(5)
    f["ma20"] = f["idx_close"].rolling(20).mean()
    return m, f


if __name__ == "__main__":
    m, f = build_frames()

    # 趋势日集合(现有风控代理): 000300收盘>其MA20
    global TREND_DAYS
    TREND_DAYS = set(f.index[(f["idx_close"] > f["ma20"])])

    y = f["ret"].dropna()  # 因变量: 次日市场收益
    era = pd.Series(np.where(pd.to_numeric(y.index) < int(ERA_SPLIT), "early", "recent"),
                    index=y.index)
    print(f"检验样本: {len(y)}日 {y.index[0]}~{y.index[-1]} "
          f"(两融2012-09起, 期指2017-01起) | 趋势内日数={len(TREND_DAYS)}\n")

    print("== margin变量(对齐D→D+1收益, D+1早8:30发布赶得上D+1决策) ==")
    for c in ["m_chg5", "m_chg20", "rzbuy_chg5", "mvr_dev"]:
        _quintile_test(m[c].reindex(y.index), y, era, c)
    print("\n== future变量(T收盘可得, 显式shift→T+1收益) ==")
    for c in ["basis", "basis_chg5"]:
        _quintile_test(f[c].reindex(y.index), y, era, c)

    print("\n注: 全样本分位有轻前视, 仅作初筛; 通过者接入前须expanding/滚动分位+走前验证。")
