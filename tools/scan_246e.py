"""第246轮 E: 非破位日里那个「个股自己崩了」的信号 —— 能不能并进来

246d 挖到一件 245 完全没利用的事:
    指数不破位的 2704 天里, 有 51 天「低开组里跌最透的那只」自己低开了 5% 以上
    这 51 笔单笔 4.30%、胜率 69% —— 比破位日整体的 1.35% 高 3 倍
而破位日里同样的「低开 <= -5%」19 笔是 8.30%、胜率 84%。

也就是说: 真正起作用的可能不是「大盘破位」, 而是「这只票自己被砸出缺口」。
大盘破位只是「大概率能找到这样一只票」的一种方式。

于是规则改成并列触发:
    每日照样算「低开组里 dh250 最小」的那只
    满足其一就买:  (a) 指数距 20 日线 <= -3.5%   (b) 这只票低开 <= -5%
两个条件 T 日收盘 / T+1 9:25 都完全可知。

风险: 51 笔可能全挤在 2015-2016 股灾, 那就不可复制。必须先看分年。
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
    mk = pd.read_csv(OUT / "feat234" / "market.csv",
                     usecols=["trade_date", "idx_ma20_dist", "breadth", "limit_dn",
                              "med_pct", "eq_ret"])
    d = d.merge(mk, on="trade_date", how="inner").dropna(subset=["idx_ma20_dist"])

    lo = d[d.n1_gap <= 0.05].sort_values(
        ["trade_date", "dh250", "amt_yi"], ascending=[True, True, False])
    pk = lo.groupby("trade_date").head(1).copy()
    pk["y_net"] = pk["y"] - COST
    pk["yr"] = pk.trade_date // 10000

    print(f"每日一只 {len(pk)} 天  单笔 {pk.y_net.mean():.3f}%")
    deep = pk[pk.n1_gap <= -5]
    print(f"其中低开<=-5% {len(deep)} 天  单笔 {deep.y_net.mean():.3f}%  "
          f"胜率 {(deep.y_net>0).mean()*100:.0f}%")

    print("\n【1】「低开<=-5%」这 70 天的分年（决定可不可复制）")
    g = deep.groupby("yr").agg(n=("y_net", "size"), 单笔=("y_net", "mean"),
                               合计=("y_net", "sum"))
    g["胜率"] = deep.groupby("yr").y_net.apply(lambda x: (x > 0).mean() * 100).round(0)
    print(g.round(2).to_string())
    print(f"  合计 {deep.y_net.sum():.0f}pp, 前 3 笔占 {deep.y_net.nlargest(3).sum():.0f}pp")

    print("\n【2】分年: 三个触发源各自多少天、单笔多少")
    pk["src"] = np.where(pk.idx_ma20_dist <= -3.5,
                         np.where(pk.n1_gap <= -5, "破位+深低开", "破位"),
                         np.where(pk.n1_gap <= -5, "仅深低开", "都不满足"))
    pv = pk.pivot_table(index="yr", columns="src", values="y_net",
                        aggfunc=["size", "mean"])
    print(pv.round(2).to_string())

    print("\n【3】并进来之后: 净值对比（主策略为底, 补位仓位 50%, 滑点 0.2）")
    tr = pd.read_csv(OUT / "w232_trades.csv")
    all_dates = sorted(int(p.stem.replace(".csv", "")) for p in
                       (ROOT / "data" / "panel").glob("*.csv.gz"))
    cal = np.array([d_ for d_ in all_dates if d_ >= 2016])
    base = pd.Series(0.0, index=cal)
    for dt, gg in tr.groupby("signal"):
        base[int(dt)] = gg["ret"].mean()

    def run(sel: pd.Series, P: float, slip: float) -> tuple[float, float, float]:
        s = pd.Series(sel.values * P, index=sel.index.values).sort_index()
        s = s - slip * P
        sim = base.copy()
        sim[s.index] += s.values
        return metrics(sim)

    a0, d0, c0 = metrics(base)
    print(f"  仅主策略                     年化{a0:>7.2f}  回撤{d0:>7.2f}  卡玛{c0:>5.2f}")

    pk = pk.set_index("trade_date")
    old = pk[pk.idx_ma20_dist <= -3.5]["y_net"]
    new = pk[(pk.idx_ma20_dist <= -3.5) | (pk.n1_gap <= -5)]["y_net"]
    only_deep = pk[(pk.idx_ma20_dist > -3.5) & (pk.n1_gap <= -5)]["y_net"]

    print(f"\n  {'':<26}{'笔数':>6}{'单笔':>8}{'年化':>9}{'回撤':>9}{'卡玛':>7}")
    for tag, s in (("245 口径 (仅破位)", old), ("246 并列 (破位 or 深低开)", new),
                   ("  其中新增的仅深低开", only_deep)):
        for P in (0.5, 0.7):
            a, dd, cm = run(s, P, 0.2)
            print(f"  {tag:<26}{len(s):>6}{s.mean():>8.2f}{a:>9.2f}{dd:>9.2f}{cm:>7.2f}"
                  + ("" if P == 0.5 else "   (70%仓)"))

    print("\n【4】阈值敏感性: 「低开<=-X%」用来扩触发, X 取多少")
    for X in (-4.0, -4.5, -5.0, -5.5, -6.0, -7.0):
        s = pk[(pk.idx_ma20_dist <= -3.5) | (pk.n1_gap <= X)]["y_net"]
        a, dd, cm = run(s, 0.5, 0.2)
        n_new = len(pk[(pk.idx_ma20_dist > -3.5) & (pk.n1_gap <= X)])
        print(f"  X={X:>5.1f}  n={len(s):>4} (新增{n_new:>3})  单笔{s.mean():>6.2f}%  "
              f"年化{a:>7.2f}  回撤{dd:>7.2f}  卡玛{cm:>6.2f}")

    pk.reset_index().to_csv(OUT / "w246e_trigger.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
