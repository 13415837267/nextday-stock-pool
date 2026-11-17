"""第246轮 D: 把 245 的玩法铺到「全部交易日」—— 补位到底能不能不只做破位日

245 只在 198 个「主策略空仓 + 指数距 20 日线 <= -3.51%」日交易。11 年里
资金利用率约 30%, 1740 个空仓日在空转。这轮用全量面板(w246b_all.parquet)
回答: 同一套「低开票里挑跌最透的」, 在指数不破位的时候还赚不赚?

若赚 → 资金利用率可以翻几倍, 这是比调参数大得多的杠杆。
若不赚 → 说明这条路的天花板就是「极端超跌」, 245 已到顶, 该换方向。

用单窗口 dh250 代替 9 窗口平均(245 里两者差 0.1 卡玛), 换速度。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
COST = 0.10


def metrics(r: pd.Series) -> tuple[float, float, float]:
    nav = (1 + r / 100).cumprod()
    ann = nav.iloc[-1] ** (244 / len(r)) - 1
    dd = (nav / nav.cummax() - 1).min()
    return round(ann * 100, 2), round(dd * 100, 2), round(ann * 100 / abs(dd * 100), 2)


def main() -> None:
    d = pd.read_parquet(OUT / "w246b_all.parquet",
                        columns=["ts_code", "trade_date", "dh250", "n1_gap",
                                 "y", "amt_yi", "circ_mv", "pct"])
    d = d.dropna(subset=["dh250", "n1_gap", "y"])
    print(f"可用 {len(d):,} 行  {d.trade_date.nunique()} 天")

    mk = pd.read_csv(OUT / "feat234" / "market.csv",
                     usecols=["trade_date", "idx_ma20_dist", "breadth", "limit_dn",
                              "med_pct", "eq_ret"])
    d = d.merge(mk, on="trade_date", how="inner")
    d = d.dropna(subset=["idx_ma20_dist"])
    print(f"合并市场状态后 {len(d):,} 行  {d.trade_date.nunique()} 天")

    # ---- 每日: 低开组里 dh250 最小的一只（跌离 250 日高最远） ----
    lo = d[d.n1_gap <= 0.05]
    lo = lo.sort_values(["trade_date", "dh250", "amt_yi"],
                        ascending=[True, True, False])
    pk = lo.groupby("trade_date").head(1).copy()
    pk["y_net"] = pk["y"] - COST
    print(f"每日一只: {len(pk)} 天  单笔 {pk.y_net.mean():.3f}%")

    print("\n【1】全部交易日, 按指数距 20 日线分档（这是关键一张表）")
    bins = [-99, -8, -6, -5, -4.5, -4, -3.5, -3, -2, -1, 0, 1, 99]
    pk["ib"] = pd.cut(pk.idx_ma20_dist, bins)
    g = pk.groupby("ib", observed=True).agg(
        n=("y_net", "size"), 单笔=("y_net", "mean"),
        中位=("y_net", "median"),
        胜率=("y_net", lambda x: (x > 0).mean() * 100))
    g["年化贡献"] = g["单笔"] * g["n"] / 11
    g["分年正%"] = pk.groupby("ib", observed=True).apply(
        lambda x: (x.groupby(x.trade_date // 10000).y_net.mean() > 0).mean() * 100,
        include_groups=False).round(0)
    print(g.round(2).to_string())

    print("\n【2】只看非破位日(指数距 20 日线 > -3.5%), 单笔随低开深度")
    s = pk[pk.idx_ma20_dist > -3.5].copy()
    print(f"  共 {len(s)} 天, 单笔 {s.y_net.mean():.3f}%, 胜率 {(s.y_net>0).mean()*100:.0f}%")
    s["gb"] = pd.cut(s.n1_gap, [-11, -5, -3, -2, -1, -0.5, 0.06])
    g2 = s.groupby("gb", observed=True).agg(
        n=("y_net", "size"), 单笔=("y_net", "mean"),
        胜率=("y_net", lambda x: (x > 0).mean() * 100))
    print(g2.round(2).to_string())

    print("\n【3】破位日(<= -3.5%) 同样分档, 作为对照")
    p = pk[pk.idx_ma20_dist <= -3.5].copy()
    print(f"  共 {len(p)} 天, 单笔 {p.y_net.mean():.3f}%, 胜率 {(p.y_net>0).mean()*100:.0f}%")
    p["gb"] = pd.cut(p.n1_gap, [-11, -5, -3, -2, -1, -0.5, 0.06])
    g3 = p.groupby("gb", observed=True).agg(
        n=("y_net", "size"), 单笔=("y_net", "mean"),
        胜率=("y_net", lambda x: (x > 0).mean() * 100))
    print(g3.round(2).to_string())

    print("\n【4】分年: 破位日 vs 非破位日")
    pk["grp"] = np.where(pk.idx_ma20_dist <= -3.5, "破位", "非破位")
    g4 = pk.pivot_table(index=pk.trade_date // 10000, columns="grp",
                        values="y_net", aggfunc=["size", "mean"])
    print(g4.round(2).to_string())

    pk.to_csv(OUT / "w246d_allday.csv", index=False, encoding="utf-8-sig")
    g.reset_index().to_csv(OUT / "w246d_idx_band.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
