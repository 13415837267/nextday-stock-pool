"""第246轮 J: 换回 9 窗口平均排名, 把所有结论重算一遍

前几步都用单窗口 dh250, 基线偏低(A 只到 5.02, 而 245 报的是 5.49)。原因是排序键不同:
245 用 9 个窗口各自的排名取平均, 单窗口会有"某只票在 250 日尺度上不算最惨但 60 日很惨"
的情况被漏掉。246b 重建时顺手把 9 个窗口都算了(宽表 rolling 只用 2 秒), 这里补上。

四个方案:
  A   245 口径: 指数 <= -3.5% 且 个股距60日高 <= -24.46%
  B   并列: A  或  第一名低开 <= -5%(含跌停开)
  B'  并列但排除跌停开: A 或 -9.5% < 低开 <= -5%
  B'' A 或 -8.5% < 低开 <= -5%
每个都跑 无限制 / 单票<=8
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
    print(f"候选 {len(d):,} 行 {d.trade_date.nunique()} 天", flush=True)

    parts = []
    for _yr, g in d.groupby(d.trade_date // 10000, sort=True):
        den = g[DHC].notna().sum()
        mr = (g[DHC].rank(ascending=True) / den).mean(axis=1, skipna=True)
        g = g.assign(mr=mr)
        lo = g[g.n1_gap <= 0.05]
        lo = lo.sort_values(["trade_date", "mr", "amt_yi"],
                            ascending=[True, True, False])
        parts.append(lo)
    lo = pd.concat(parts, ignore_index=True)
    del parts, d
    print(f"低开组 {len(lo):,} 行  {lo.trade_date.nunique()} 天", flush=True)
    gb = {int(dt): g for dt, g in lo.groupby("trade_date")}
    dayrank = {int(x): i for i, x in enumerate(cal)}

    def sel(rule: str, cap: int | None) -> pd.DataFrame:
        n: dict[str, int] = {}
        out = []
        for dt in sorted(gb):
            g = gb[dt]
            top = g.iloc[0]
            broke = (top.idx_ma20_dist <= -3.5) and (top.dh60 <= -24.46)
            gap = top.n1_gap
            deep = gap <= -5
            if rule == "A":
                trig = broke
            elif rule == "B":
                trig = broke or deep
            elif rule == "B1":       # 排除跌停开 (< -9.5 视为跌停/异常)
                trig = broke or (-9.5 < gap <= -5)
            elif rule == "B2":       # 排除 -8.5 以下
                trig = broke or (-8.5 < gap <= -5)
            else:
                trig = deep
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
                        "yr": dt // 10000})
        return pd.DataFrame(out)

    def net(c: pd.DataFrame, P: float) -> tuple[float, float, float, pd.Series]:
        s = pd.Series((c["y"].values - COST - SLIP) * P, index=c.trade_date.values)
        s = s.sort_index()
        sim = base.copy()
        sim[s.index] = sim[s.index] + s.values
        return (*metrics(sim), s)

    print(f"\n仅主策略 50%  {metrics(base)}")
    print(f"\n{'方案':<26}{'笔数':>5}{'票数':>5}{'单笔':>8}{'年化':>9}{'回撤':>9}{'卡玛':>7}")
    res = []
    for rule, nm in (("A", "A 245口径"), ("B", "B 并列(含跌停开)"),
                     ("B1", "B' 并列(排-9.5以下)"), ("B2", "B'' 并列(排-8.5以下)"),
                     ("C", "C 仅深低开")):
        for cap in (None, 8):
            c = sel(rule, cap)
            if not len(c):
                continue
            a, dd, cm, s = net(c, 0.5)
            tag = f"{nm}" + ("" if cap is None else f" +限{cap}")
            print(f"{tag:<26}{len(c):>5}{c.ts_code.nunique():>5}{(s.mean()/0.5):>8.2f}"
                  f"{a:>9.2f}{dd:>9.2f}{cm:>7.2f}")
            res.append({"方案": tag, "笔数": len(c), "票数": c.ts_code.nunique(),
                        "单笔": round(s.mean() / 0.5, 2), "年化": a, "回撤": dd, "卡玛": cm})
        print()

    for rule, nm in (("A", "A"), ("B1", "B'")):
        c = sel(rule, 8)
        print(f"\n【{nm} + 限8 仓位扫描】")
        for P in (0.40, 0.50, 0.60, 0.70, 0.85, 1.00):
            a, dd, cm, s = net(c, P)
            print(f"  {P:>5.0%}  年化{a:>7.2f}  回撤{dd:>7.2f}  卡玛{cm:>6.2f}")

    c = sel("A", 8)
    c["y_net"] = c.y - COST - SLIP
    print("\n【A + 限8 分年】")
    g = c.groupby("yr").agg(n=("y_net", "size"), 单笔=("y_net", "mean"),
                            合计=("y_net", "sum"))
    g["胜率"] = c.groupby("yr").y_net.apply(lambda x: (x > 0).mean() * 100).round(0)
    print(g.round(2).to_string())
    print("\n【A + 限8 三段 (50%)】")
    for lo_y, hi_y, tag in ((2016, 2019, "2016-2019"), (2020, 2022, "2020-2022"),
                            (2023, 2026, "2023-2026")):
        cc = c[(c.yr >= lo_y) & (c.yr <= hi_y)]
        a, dd, cm, s = net(cc, 0.5)
        print(f"  {tag}  n={len(cc):>3}  单笔{s.mean()/0.5:>6.2f}%  卡玛{cm:>5.2f}")

    # 与 245 的 198 笔是否一致
    print("\n【A+限8 与 245 的 198 笔重叠】")
    p245 = pd.read_csv(OUT / "w245_picks.csv", header=0)
    p245.columns = ["signal", "y"]
    s1 = set(c.trade_date.astype(int))
    s2 = set(p245.signal.astype(int))
    print(f"  本方案 {len(s1)} 天, 245 方案 {len(s2)} 天, 交集 {len(s1 & s2)}")

    pd.DataFrame(res).to_csv(OUT / "w246j_compare.csv", index=False, encoding="utf-8-sig")
    c.to_csv(OUT / "w246j_picks.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
