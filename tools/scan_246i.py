"""第246轮 I: 把「深低开」拆开看 —— 它到底是超跌反弹, 还是跌停板博弈

246g 的 B 方案(破位 or 低开<=-5) 236 笔 单笔 2.20% 卡玛 5.80, 比 A 的 5.02 高。
但把新增的 19 笔摊开看, 11 笔是跌停开盘(gap <= -9.5):
  20250117 002693 -9.72% / 20250422 002693 -9.97% / 20250423 002693 -10.03%
  20250424 002693 -9.97% / 20250425 002693 -10.02%  (同一只票连买 5 天)
  20201214 002647 -10.03% / 20201216 002647 -9.98% / 20240918 600696 -9.98%
  20230524 000656 -10.42% / 20260116 600340 -10.13% / ...
这不是「超跌后的温和低开」, 是「跌停板上的博弈」:
  买入容易(跌停有大量卖盘), 卖出不确定(若 T+2 仍跌停, 回测按收盘价成交是虚的)
真真假假混在一起, 不能当成一个信号。

所以本轮把「深低开」按深度拆开, 分别看:
  -5 ~ -8.5   温和闪崩(还在正常交易区间)
  <= -9.5     跌停开(另一个游戏)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
COST = 0.10
SLIP = 0.20


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
    dayrank = {int(d): i for i, d in enumerate(cal)}
    base = pd.Series(0.0, index=cal)
    busy = set(int(x) for x in tr.signal)
    for _, r in tr.iterrows():
        busy.add(int(r["buy"]))
        busy.add(int(r["sell"]))
    for dt, g in tr.groupby("signal"):
        base[int(dt)] = g["ret"].mean()

    d = pd.read_parquet(OUT / "w246b_all.parquet",
                        columns=["ts_code", "trade_date", "dh250", "dh60", "listed",
                                 "n1_gap", "y", "amt_yi", "circ_mv"])
    d = d.dropna(subset=["dh250", "dh60", "n1_gap", "y"])
    d = d[d.listed >= 250]
    d["trade_date"] = d["trade_date"].astype("int64")
    d = d[(d.trade_date >= cal[0]) & (~d.trade_date.isin(busy))]
    mk = pd.read_csv(OUT / "feat234" / "market.csv",
                     usecols=["trade_date", "idx_ma20_dist"])
    d = d.merge(mk, on="trade_date", how="inner").dropna(subset=["idx_ma20_dist"])
    lo = d[d.n1_gap <= 0.05].sort_values(
        ["trade_date", "dh250", "amt_yi"], ascending=[True, True, False])
    lo = lo[["ts_code", "trade_date", "dh250", "dh60", "n1_gap", "y", "amt_yi",
             "circ_mv", "idx_ma20_dist"]].reset_index(drop=True)
    gb = {int(dt): g for dt, g in lo.groupby("trade_date")}

    def sel(thr: float, hi: float, cap: int | None) -> pd.DataFrame:
        """thr: 深低开下限; hi: 深低开上限(排除跌停开); cap: 单票最多买几次"""
        last: dict[str, int] = {}
        n: dict[str, int] = {}
        out = []
        for dt in sorted(gb):
            g = gb[dt]
            top = g.iloc[0]
            broke = (top.idx_ma20_dist <= -3.5) and (top.dh60 <= -24.46)
            deep = (top.n1_gap <= thr) and (top.n1_gap > hi)
            if not (broke or deep):
                continue
            i = dayrank[dt]
            pick = top
            for j in range(len(g)):
                rr = g.iloc[j]
                if cap and n.get(rr.ts_code, 0) >= cap:
                    continue
                pick = rr
                break
            last[pick.ts_code] = i
            n[pick.ts_code] = n.get(pick.ts_code, 0) + 1
            out.append({"trade_date": dt, "ts_code": pick.ts_code, "y": pick.y,
                        "n1_gap": pick.n1_gap, "brk": broke, "yr": dt // 10000})
        return pd.DataFrame(out)

    def net(c: pd.DataFrame, P: float) -> tuple[float, float, float, pd.Series]:
        s = pd.Series((c["y"].values - COST - SLIP) * P, index=c.trade_date.values)
        s = s.sort_index()
        sim = base.copy()
        sim[s.index] = sim[s.index] + s.values
        return (*metrics(sim), s)

    print(f"仅主策略 50%   {metrics(base)}")
    print(f"\n{'方案':<34}{'笔数':>5}{'票数':>5}{'单笔':>8}{'年化':>9}{'回撤':>9}{'卡玛':>7}")
    cands = [
        ("A 245口径 无限制", sel(-99, -99, None)),
        ("A + 单票<=12", sel(-99, -99, 12)),
        ("A + 单票<=10", sel(-99, -99, 10)),
        ("A + 单票<=8", sel(-99, -99, 8)),
        ("A + 单票<=6", sel(-99, -99, 6)),
        ("A + 单票<=4", sel(-99, -99, 4)),
        ("B 深低开<=-5 全收", sel(-5, -99, None)),
        ("B' 只 -5~-8.5(排跌停开)", sel(-5, -8.5, None)),
        ("B'' 只 -5~-9.5", sel(-5, -9.5, None)),
        ("B' + 单票<=8", sel(-5, -8.5, 8)),
        ("B'' + 单票<=8", sel(-5, -9.5, 8)),
        ("B + 单票<=8", sel(-5, -99, 8)),
    ]
    res = []
    for tag, c in cands:
        a, dd, cm, s = net(c, 0.5)
        print(f"{tag:<34}{len(c):>5}{c.ts_code.nunique():>5}{(s.mean()/0.5):>8.2f}"
              f"{a:>9.2f}{dd:>9.2f}{cm:>7.2f}")
        res.append({"方案": tag, "笔数": len(c), "票数": c.ts_code.nunique(),
                    "单笔": round(s.mean() / 0.5, 2), "年化": a, "回撤": dd, "卡玛": cm})

    best = sel(-99, -99, 8)
    print("\n【A + 单票<=8 的仓位扫描】")
    for P in (0.40, 0.50, 0.60, 0.70, 0.85, 1.00):
        a, dd, cm, s = net(best, P)
        print(f"  {P:>5.0%}  年化{a:>7.2f}  回撤{dd:>7.2f}  卡玛{cm:>6.2f}")

    print("\n【A + 单票<=8 分年】")
    best["y_net"] = best.y - COST - SLIP
    g = best.groupby("yr").agg(n=("y_net", "size"), 单笔=("y_net", "mean"),
                               合计=("y_net", "sum"))
    g["胜率"] = best.groupby("yr").y_net.apply(lambda x: (x > 0).mean() * 100).round(0)
    print(g.round(2).to_string())

    cnt = best.ts_code.value_counts()
    print(f"\n  集中度: {len(best)} 笔 / {best.ts_code.nunique()} 只票  "
          f"最长连买 {cnt.max()} 最差票 {(best.groupby('ts_code').y.mean()-0.3).min():.1f}%")

    print("\n【A + 单票<=8 三段】")
    for lo_y, hi_y, tag in ((2016, 2019, "2016-2019"), (2020, 2022, "2020-2022"),
                            (2023, 2026, "2023-2026")):
        cc = best[(best.yr >= lo_y) & (best.yr <= hi_y)]
        a, dd, cm, s = net(cc, 0.5)
        print(f"  {tag}  n={len(cc):>3}  单笔{s.mean()/0.5:>6.2f}%  卡玛{cm:>5.2f}  回撤{dd:>6.2f}")

    print("\n【B' (只温和闪崩) 明细】")
    bp = sel(-5, -8.5, None)
    ex = bp[~bp.brk].sort_values("trade_date")
    print(f"  新增 {len(ex)} 笔: 单笔 {(ex.y-COST-SLIP).mean():.2f}%  "
          f"胜率 {((ex.y-COST-SLIP)>0).mean()*100:.0f}%")
    print(ex[["trade_date", "ts_code", "n1_gap", "y"]].to_string(index=False))

    pd.DataFrame(res).to_csv(OUT / "w246i_compare.csv", index=False, encoding="utf-8-sig")
    best.to_csv(OUT / "w246i_picks.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
