"""第246轮 H: 不要冷却了 —— 用「单票限次」管集中度

246g 的结论:
  ① B(并列深低开) 三段分年 全部 >= A, 后两段明显更好(2023-2026: 4.06 -> 4.52)
  ② 剔除最大 5 笔后 B 仍赢 0.57, 不是靠一两笔
  ③ B 最差单票 -12.2% vs A 的 -22.3% —— 更安全
  ④ 深低开阈值 -4 ~ -7 卡玛 5.72~5.82, 不敏感
  ⑤ 但两套最长连买都是 20 次, 集中度硬伤

243 轮用「冷却 20 日」解决集中度, 代价是卡玛掉 0.87。本轮发现冷却在这套里掉 2.4 ——
因为冷却换票换到的第 2 名, 在单窗口 dh250 下比第 1 名差很多。

换一种管集中度的办法: 同一只票最多买 N 次。好处是「不换掉最好的票, 只是不再加」。
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
                     usecols=["trade_date", "idx_ma20_dist", "breadth", "limit_dn",
                              "med_pct", "eq_ret"])
    d = d.merge(mk, on="trade_date", how="inner").dropna(subset=["idx_ma20_dist"])
    lo = d[d.n1_gap <= 0.05].sort_values(
        ["trade_date", "dh250", "amt_yi"], ascending=[True, True, False])
    lo = lo[["ts_code", "trade_date", "dh250", "dh60", "n1_gap", "y", "amt_yi",
             "circ_mv", "idx_ma20_dist"]].reset_index(drop=True)
    gb = {int(dt): g for dt, g in lo.groupby("trade_date")}

    def sel(cap: int | None, cd: int = 0) -> pd.DataFrame:
        """cap=单票最多买几次(超出则换当天下一只); cd=冷却天数(0=不用)"""
        last: dict[str, int] = {}
        n: dict[str, int] = {}
        out = []
        for dt in sorted(gb):
            g = gb[dt]
            top = g.iloc[0]
            broke = (top.idx_ma20_dist <= -3.5) and (top.dh60 <= -24.46)
            if not (broke or (top.n1_gap <= -5)):
                continue
            i = dayrank[dt]
            pick = top
            for j in range(len(g)):
                rr = g.iloc[j]
                if cd and rr.ts_code in last and i - last[rr.ts_code] < cd:
                    continue
                if cap and n.get(rr.ts_code, 0) >= cap:
                    continue
                pick = rr
                break
            last[pick.ts_code] = i
            n[pick.ts_code] = n.get(pick.ts_code, 0) + 1
            out.append({"trade_date": dt, "ts_code": pick.ts_code, "y": pick.y,
                        "n1_gap": pick.n1_gap, "dh60": pick.dh60,
                        "idx": top.idx_ma20_dist, "brk": broke,
                        "yr": dt // 10000})
        return pd.DataFrame(out)

    def net(c: pd.DataFrame, P: float) -> tuple[float, float, float, pd.Series]:
        s = pd.Series((c["y"].values - COST - SLIP) * P, index=c.trade_date.values)
        s = s.sort_index()
        sim = base.copy()
        sim[s.index] = sim[s.index] + s.values
        return (*metrics(sim), s)

    print(f"仅主策略 {metrics(base)}")
    print(f"\n{'方案':<22}{'笔数':>5}{'票数':>5}{'最长连':>6}{'前5占':>6}"
          f"{'单笔':>8}{'年化':>9}{'回撤':>9}{'卡玛':>7}")
    rows = []
    for tag, c in (
        ("无限制", sel(None)),
        ("单票<=12次", sel(12)),
        ("单票<=8次", sel(8)),
        ("单票<=5次", sel(5)),
        ("单票<=3次", sel(3)),
        ("单票<=2次", sel(2)),
        ("冷却20日", sel(None, 20)),
        ("单票<=5 & 冷却20", sel(5, 20)),
    ):
        cnt = c.ts_code.value_counts()
        a, dd, cm, s = net(c, 0.5)
        print(f"{tag:<22}{len(c):>5}{c.ts_code.nunique():>5}{cnt.max():>6}"
              f"{cnt.head(5).sum()/len(c)*100:>5.0f}%{s.mean()/0.5:>8.2f}"
              f"{a:>9.2f}{dd:>9.2f}{cm:>7.2f}")
        rows.append({"方案": tag, "笔数": len(c), "票数": c.ts_code.nunique(),
                     "最长连买": int(cnt.max()), "单笔": round(s.mean() / 0.5, 2),
                     "年化": a, "回撤": dd, "卡玛": cm})

    print("\n【单票<=8 的仓位扫描】")
    c = sel(8)
    for P in (0.35, 0.50, 0.60, 0.70, 0.85, 1.00):
        a, dd, cm, s = net(c, P)
        print(f"  {P:>5.0%}  年化{a:>7.2f}  回撤{dd:>7.2f}  卡玛{cm:>6.2f}")

    print("\n【单票<=8 分年】")
    c = sel(8)
    c["y_net"] = c.y - COST - SLIP
    g = c.groupby("yr").agg(n=("y_net", "size"), 单笔=("y_net", "mean"),
                            合计=("y_net", "sum"))
    g["胜率"] = c.groupby("yr").y_net.apply(lambda x: (x > 0).mean() * 100).round(0)
    print(g.round(2).to_string())

    print("\n【单票<=8 三段】")
    for lo_y, hi_y, tag in ((2016, 2019, "2016-2019"), (2020, 2022, "2020-2022"),
                            (2023, 2026, "2023-2026")):
        cc = c[(c.yr >= lo_y) & (c.yr <= hi_y)]
        a, dd, cm, s = net(cc, 0.5)
        print(f"  {tag}  n={len(cc):>3}  单笔{s.mean()/0.5:>6.2f}%  卡玛{cm:>5.2f}  "
              f"回撤{dd:>6.2f}")

    pd.DataFrame(rows).to_csv(OUT / "w246h_cap.csv", index=False, encoding="utf-8-sig")
    sel(8).to_csv(OUT / "w246h_picks.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
