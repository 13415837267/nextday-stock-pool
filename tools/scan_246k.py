"""第246轮 K: 定稿 —— 修掉一个跨了十几轮的数据 bug, 重新算基线

## 发现的 bug

build_features_234.py 的 main() 是「按年分片」跑的:
    for y in years:
        yf = [f for f in files if f.stem[:4] == str(y)]
        d = build_year(yf, ...)      # ← rolling 只看得到当年的数据
所有 rolling 特征(bias5/10/20/60、dist_high60、ret20、vol20、maxdd20、amt_ratio…)
在每年 1 月的前 20~30 个交易日都是 NaN 或严重有偏。

后果: 244g 用 `dist_high60 <= -24.46` 建候选池, 每年 1 月恒为空。
  245 的 198 笔按月分布:  1月 2 笔, 其余月份正常
  正确口径 232 笔按月:    1月 31 笔,
其余月份两版几乎逐月相同(2月28/29, 3月17/17, 4月34/34, 5月19/19, 6月24/24 …
12月 16/21)。也就是说 243/244/245 三轮的结论, 全都建立在「每年 1 月不做」的样本上。

1 月的破位日单笔只有 1.18%(其他月份 2.33%) —— 因为 1 月的下跌多由年报预告暴雷驱动,
是个股基本面事件, 不是流动性恐慌, 超跌反弹的逻辑不成立。这不是玄学, 有解释。
但「排除 1 月」是事后归因, 本轮不做, 全时段照做。

## 本轮的方案

A   245 口径: 指数 <= -3.5% 且 个股距60日高 <= -24.46%
B'  A 或 (第一名低开在 -9.5% ~ -5% 之间)   ← 排除跌停开
    "跌停开"单列: 19 笔里 11 笔是 gap <= -9.5%, 那是跌停板博弈(买得到、卖不确定),
    不是超跌反弹, 不并进来。
两者都加「单票最多买 8 次」替代 243 的冷却 20 —— 冷却换票会把好票换掉(卡玛掉 2.4),
限次则不换掉当天最好的票, 只是不再加。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
COST = 0.10
SLIP = 0.20
WINS = (60, 90, 120, 150, 180, 200, 250, 300, 360)
DHC = [f"dh{W}" for W in WINS]


def metrics(r: pd.Series) -> tuple[float, float, float]:
    nav = (1 + r / 100).cumprod()
    ann = nav.iloc[-1] ** (244 / len(r)) - 1
    dd = (nav / nav.cummax() - 1).min()
    return round(ann * 100, 2), round(dd * 100, 2), round(ann * 100 / abs(dd * 100), 2)


def main() -> None:
    tr = pd.read_csv(OUT / "w232_trades.csv")
    all_dates = sorted(int(p.stem.replace(".csv", "")) for p in
                       (ROOT / "data" / "panel").glob("*.csv.gz"))
    cal = np.array([x for x in all_dates if x >= int(tr.signal.min())])
    base = pd.Series(0.0, index=cal)
    busy = set(int(x) for x in tr.signal)
    for _, r in tr.iterrows():
        busy.add(int(r["buy"]))
        busy.add(int(r["sell"]))
    for dt, g in tr.groupby("signal"):
        base[int(dt)] = g["ret"].mean()

    d = pd.read_parquet(OUT / "w246b_all.parquet",
                        columns=["ts_code", "trade_date", "listed", "n1_gap", "y",
                                 "amt_yi", "circ_mv"] + DHC)
    d = d[d.listed >= 250]
    d["trade_date"] = d["trade_date"].astype("int64")
    d = d[(d.trade_date >= cal[0]) & (~d.trade_date.isin(busy))]
    mk = pd.read_csv(OUT / "feat234" / "market.csv",
                     usecols=["trade_date", "idx_ma20_dist"])
    d = d.merge(mk, on="trade_date", how="inner")
    d = d.dropna(subset=DHC + ["n1_gap", "y", "idx_ma20_dist"])

    parts = []
    for _yr, g in d.groupby(d.trade_date // 10000, sort=True):
        den = g[DHC].notna().sum()
        mr = (g[DHC].rank(ascending=True) / den).mean(axis=1, skipna=True)
        g = g.assign(mr=mr)
        lo = g[g.n1_gap <= 0.05].sort_values(
            ["trade_date", "mr", "amt_yi"], ascending=[True, True, False])
        parts.append(lo)
    lo = pd.concat(parts, ignore_index=True)
    del parts, d
    gb = {int(dt): g for dt, g in lo.groupby("trade_date")}

    def sel(rule: str, cap: int | None = 8) -> pd.DataFrame:
        n: dict[str, int] = {}
        out = []
        for dt in sorted(gb):
            g = gb[dt]
            top = g.iloc[0]
            broke = (top.idx_ma20_dist <= -3.5) and (top.dh60 <= -24.46)
            gap = top.n1_gap
            if rule == "A":
                trig = broke
            elif rule == "B1":
                trig = broke or (-9.5 < gap <= -5)
            else:                                        # B: 含跌停开
                trig = broke or (gap <= -5)
            if not trig:
                continue
            pick = top
            if cap:
                for j in range(len(g)):
                    rr = g.iloc[j]
                    if n.get(rr.ts_code, 0) < cap:
                        pick = rr
                        break
            n[pick.ts_code] = n.get(pick.ts_code, 0) + 1
            out.append({"trade_date": dt, "ts_code": pick.ts_code, "y": pick.y,
                        "n1_gap": pick.n1_gap, "brk": broke, "idx": top.idx_ma20_dist,
                        "yr": dt // 10000, "m": dt // 100 % 100})
        return pd.DataFrame(out)

    def net(c: pd.DataFrame, P: float) -> tuple[float, float, float, pd.Series]:
        s = pd.Series((c["y"].values - COST - SLIP) * P, index=c.trade_date.values)
        s = s.sort_index()
        sim = base.copy()
        sim[s.index] = sim[s.index] + s.values
        return (*metrics(sim), s)

    A = sel("A")
    B1 = sel("B1")
    Bc = sel("B")
    print(f"仅主策略     {metrics(base)}")
    print(f"A   {len(A)} 笔 (1月 {int((A.m==1).sum())})   B'  {len(B1)} 笔  "
          f"B(含跌停开) {len(Bc)} 笔")

    print("\n【1】仓位细扫 (50% 以下这段才是平台)")
    print(f"{'仓位':>6}{'A年化':>9}{'A回撤':>9}{'A卡玛':>7}   "
          f"{'B9年化':>9}{'B9回撤':>9}{'B9卡玛':>7}")
    rows = []
    for P in (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70):
        aa, add, ac, _ = net(A, P)
        ba, bdd, bc, _ = net(B1, P)
        print(f"{P:>6.0%}{aa:>9.2f}{add:>9.2f}{ac:>7.2f}   {ba:>9.2f}{bdd:>9.2f}{bc:>7.2f}")
        rows.append({"仓位": P, "A年化": aa, "A回撤": add, "A卡玛": ac,
                     "B年化": ba, "B回撤": bdd, "B卡玛": bc})

    print("\n【2】B' 分年 (40% 仓)")
    B1 = B1.copy()
    B1["y_net"] = B1.y - COST - SLIP
    g = B1.groupby("yr").agg(n=("y_net", "size"), 单笔=("y_net", "mean"),
                             合计=("y_net", "sum"))
    g["胜率"] = B1.groupby("yr").y_net.apply(lambda x: (x > 0).mean() * 100).round(0)
    print(g.round(2).to_string())

    print("\n【3】B' 三段 (40% 仓)")
    for lo_y, hi_y, tag in ((2016, 2019, "2016-2019"), (2020, 2022, "2020-2022"),
                            (2023, 2026, "2023-2026")):
        for nm, c in (("A", A), ("B'", B1)):
            cc = c[(c.yr >= lo_y) & (c.yr <= hi_y)]
            a, dd, cm, s = net(cc, 0.4)
            print(f"  {tag} {nm:<3} n={len(cc):>3} 单笔{s.mean()/0.4:>6.2f}% "
                  f"卡玛{cm:>5.2f}", end="")
        print()

    print("\n【4】1 月 vs 其他月份 (B', 40% 仓)")
    for tag, m in (("1月", B1.m == 1), ("其他月", B1.m != 1)):
        c = B1[m]
        a, dd, cm, s = net(c, 0.4)
        print(f"  {tag:<6} n={len(c):>3} 单笔{s.mean()/0.4:>6.2f}% 胜率"
              f"{(s>0).mean()*100:>4.0f}%  该段卡玛{cm:>5.2f}")

    print("\n【5】集中度与极端依赖 (B')")
    cnt = B1.ts_code.value_counts()
    y = B1.y_net
    print(f"  {len(B1)} 笔 / {B1.ts_code.nunique()} 只票  最长连买 {cnt.max()}  "
          f"前5票占 {cnt.head(5).sum()/len(B1)*100:.0f}%  最差单笔 {y.min():.1f}%")
    print(f"  合计 {y.sum():.0f}pp  最大5笔 {y.nlargest(5).sum():.0f}pp "
          f"({y.nlargest(5).sum()/y.sum()*100:.0f}%)  剔5笔后单笔 "
          f"{y.drop(y.nlargest(5).index).mean():.2f}%")

    print("\n【6】与 245 报的数字对照 (245: 70%仓 117.2/-23.75/4.93, 198 笔)")
    for nm, c in (("A", A), ("B'", B1)):
        for P in (0.4, 0.7):
            a, dd, cm, s = net(c, P)
            print(f"  {nm} {P:.0%}仓  {a:>7.2f}%/{dd:>7.2f}%/{cm:>5.2f}  ({len(c)} 笔)")

    pd.DataFrame(rows).to_csv(OUT / "w246k_pos.csv", index=False, encoding="utf-8-sig")
    B1.to_csv(OUT / "w246k_picks.csv", index=False, encoding="utf-8-sig")
    A.to_csv(OUT / "w246k_picksA.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
