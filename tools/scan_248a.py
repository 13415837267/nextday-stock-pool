"""第248轮 A: 重排优先级 —— 破位日到底该谁做?

247 的排队规则: 破位日优先做主策略(补位 busy 排除), 主策略空仓时才做补位。
这个优先级从没被检验过。如果破位日主策略买入的信号本身就很差,
那就该反过来: 破位日专门做补位, 主策略让位。

产出 output/w248_daily_state.csv (每日状态) + 主策略按市场状态分组统计
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
COST = 0.30


def main() -> None:
    print("=" * 78)
    print("第248轮 A: 破位日谁该做")
    print("=" * 78)

    daily = pd.read_csv(OUT / "w248_daily_state.csv")
    mk = pd.read_csv(OUT / "feat234" / "market.csv",
                     usecols=["trade_date", "breadth", "med_pct", "limit_dn",
                              "amt_shrink", "eq_ret"])
    daily = daily.merge(mk, on="trade_date", how="left")
    print(f"交易日 {len(daily)}  破位日 {int(daily.broken.sum())} ({daily.broken.mean()*100:.1f}%)")

    tr = pd.read_csv(OUT / "w232_trades.csv")
    st = daily.set_index("trade_date")
    tr["idx_d"] = tr.signal.map(st.idx_ma20_dist)
    tr["top_dh60"] = tr.signal.map(st.top1_dh60)
    tr["broken"] = tr.signal.map(st.broken)
    tr["med_dh60"] = tr.signal.map(st.med_dh60)
    tr["n_pool"] = tr.signal.map(st.n_pool)
    tr["br"] = tr.signal.map(st.breadth)
    tr["ldn"] = tr.signal.map(st.limit_dn)
    tr["shr"] = tr.signal.map(st.amt_shrink)
    tr["ret_c"] = tr.ret - COST
    print(f"主策略 {len(tr)} 笔, 其中 signal 日破位 {int(tr.broken.sum())} 笔")

    def stat(s: pd.DataFrame, nm: str) -> None:
        if not len(s):
            print(f"  {nm:<28} 0 笔")
            return
        r = s.ret_c
        t = r.mean() / (r.std(ddof=1) / np.sqrt(len(r))) if len(r) > 2 else np.nan
        print(f"  {nm:<28}{len(s):>4} 笔  单笔 {r.mean():>5.2f}%  中位 {r.median():>5.2f}%  "
              f"胜率 {(r > 0).mean()*100:>3.0f}%  t={t:>5.2f}")

    print("\n【1】主策略: 破位日 vs 非破位日")
    stat(tr, "全部")
    stat(tr[tr.broken], "signal 日破位")
    stat(tr[~tr.broken], "signal 日不破位")

    print("\n【2】主策略: 按 signal 日指数距 20 日线分档")
    bins = [-99, -5, -3.51, -2, 0, 2, 99]
    lab = ["≤-5%", "-5~-3.51", "-3.51~-2", "-2~0", "0~+2", ">+2"]
    tr["ib"] = pd.cut(tr.idx_d, bins, labels=lab)
    for k in lab:
        stat(tr[tr.ib == k], k)

    print("\n【3】主策略: 按 signal 日第一名距 60 日高分档")
    bins2 = [-99, -50, -40, -35, -25, -15, 0]
    lab2 = ["≤-50%", "-50~-40", "-40~-35", "-35~-25", "-25~-15", ">-15%"]
    tr["db"] = pd.cut(tr.top_dh60, bins2, labels=lab2)
    for k in lab2:
        stat(tr[tr.db == k], k)

    print("\n【4】主策略: 按 signal 日跌停家数分档")
    bins3 = [-1, 0, 5, 15, 40, 100, 999]
    lab3 = ["0", "1-5", "6-15", "16-40", "41-100", ">100"]
    tr["lb"] = pd.cut(tr.ldn, bins3, labels=lab3)
    for k in lab3:
        stat(tr[tr.lb == k], k)

    print("\n【5】主策略: 按 signal 日市场宽度分档")
    bins4 = [-1, 0.15, 0.3, 0.45, 0.6, 1.01]
    lab4 = ["<15%", "15-30%", "30-45%", "45-60%", ">60%"]
    tr["bb"] = pd.cut(tr.br, bins4, labels=lab4)
    for k in lab4:
        stat(tr[tr.bb == k], k)

    print("\n【6】破位日主策略与「同一天补位候选」对比")
    pk = pd.read_csv(OUT / "w247_picks.csv")
    pk["ret_c"] = pk.y - COST
    pkb = pk[pk.brk]
    print(f"  破位日补位候选  {len(pkb):>4} 笔  单笔 {pkb.ret_c.mean():>5.2f}%  胜率 {(pkb.ret_c>0).mean()*100:.0f}%")
    tb = tr[tr.broken]
    print(f"  破位日主策略    {len(tb):>4} 笔  单笔 {tb.ret_c.mean():>5.2f}%  胜率 {(tb.ret_c>0).mean()*100:.0f}%")
    # 交集: 破位日既发主策略信号、又有补位候选
    both = set(tb.signal) & set(pkb.trade_date)
    print(f"  同日都有的 {len(both)} 天")

    tr.to_csv(OUT / "w248_trades_tag.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
