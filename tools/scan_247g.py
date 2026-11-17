"""第247轮 G: dh60 从 -24.46 收紧到 -35 的锐度检验

F 轮扫出 dh60 是倒 U 型: -24.46→5.16 / -32→5.18 / -35→5.49 / -40→5.41 /
  -50→4.68 / -60→4.60 / 删除→4.32。峰在 -35~-40, 两侧都掉, 像是真的甜点区。
但整体卡玛会掩盖"少数年份贡献"的老问题。本脚本做三件事:
  ① 分年: 每一个 dh 值下, 每年的卡玛与单笔
  ② 逐年配对: -35 相对 -24.46 在几年里更好
  ③ 三段 (2016-2020 / 2021-2023 / 2024-2026)
若 ① 显示优势集中在 1~2 年, 就是噪声, 不采用。
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
COST = 0.30
WINS = (60, 90, 120, 150, 180, 200, 250, 300, 360)
DHC = [f"dh{W}" for W in WINS]


def metrics(r: pd.Series) -> tuple[float, float, float]:
    nav = (1 + r / 100).cumprod()
    ann = nav.iloc[-1] ** (244 / len(r)) - 1
    dd = (nav / nav.cummax() - 1).min()
    return round(ann * 100, 2), round(dd * 100, 2), round(ann * 100 / abs(dd * 100), 2)


def main() -> None:
    print("=" * 78)
    print("第247轮 G: dh60 收紧的锐度")
    print("=" * 78)

    tr = pd.read_csv(OUT / "w232_trades.csv")
    tr["signal"] = tr["signal"].astype(int)
    cal = sorted(int(f.split("/")[-1].replace(".csv.gz", ""))
                 for f in glob.glob(str(ROOT / "data" / "panel" / "*.csv.gz")))
    cal = np.array([d for d in cal if d >= tr.signal.min()])
    base = pd.Series(0.0, index=cal)
    busy = set(int(x) for x in tr.signal)
    for _, r in tr.iterrows():
        busy.add(int(r["buy"]))
        busy.add(int(r["sell"]))
    for dt, g in tr.groupby("signal"):
        base[int(dt)] = g["ret"].mean()

    d = pd.read_parquet(OUT / "w246b_all.parquet",
                        columns=["ts_code", "trade_date", "listed", "n1_gap", "y",
                                 "amt_yi"] + DHC)
    d = d[d.listed >= 250]
    d["trade_date"] = d["trade_date"].astype("int64")
    d = d[(d.trade_date >= cal[0]) & (~d.trade_date.isin(busy))]
    mk = pd.read_csv(OUT / "feat234" / "market.csv", usecols=["trade_date", "idx_ma20_dist"])
    d = d.merge(mk, on="trade_date", how="inner")
    d = d.dropna(subset=DHC + ["n1_gap", "y", "idx_ma20_dist"])

    parts = []
    for _yr, g in d.groupby(d.trade_date // 10000, sort=True):
        den = g[DHC].notna().sum()
        mr = (g[DHC].rank(ascending=True) / den).mean(axis=1, skipna=True)
        g = g.assign(mr=mr)
        parts.append(g[g.n1_gap <= 0.05])
    lo = pd.concat(parts, ignore_index=True)
    del parts, d
    lo = lo.sort_values(["trade_date", "mr", "amt_yi"], ascending=[True, True, False])
    gb = {int(k): g for k, g in lo.groupby("trade_date")}

    def sel(dh: float) -> pd.DataFrame:
        n: dict[str, int] = {}
        out = []
        for dt in sorted(gb):
            g = gb[dt]
            top = g.iloc[0]
            broke = (top.idx_ma20_dist <= -3.5) and (top.dh60 <= dh)
            gap = float(top.n1_gap)
            deep = (-9.5 < gap <= -5.0)
            if not (broke or deep):
                continue
            pick = top
            for j in range(len(g)):
                rr = g.iloc[j]
                if n.get(rr.ts_code, 0) < 8:
                    pick = rr
                    break
            n[pick.ts_code] = n.get(pick.ts_code, 0) + 1
            out.append({"trade_date": dt, "ts_code": pick.ts_code, "y": pick.y})
        return pd.DataFrame(out)

    DHS = (-24.46, -30.0, -35.0, -40.0)
    picks = {dh: sel(dh) for dh in DHS}

    print("\n【1】分年单笔 (每格: 笔数/单笔%)")
    yrs = list(range(2016, 2027))
    hdr = "  年份 " + "".join(f"{('%.1f' % dh):>14}" for dh in DHS)
    print(hdr)
    tab = {}
    for yr in yrs:
        line = f"  {yr} "
        for dh in DHS:
            c = picks[dh]
            s = c[(c.trade_date // 10000) == yr]
            v = (s.y - COST).mean() if len(s) else np.nan
            line += f"{len(s):>5}/{v:>7.2f}" if len(s) else f"{'—':>14}"
            tab[(yr, dh)] = v
        print(line)

    print("\n【2】逐年配对: dh60=-35 的单笔 减 -24.46 的单笔")
    win = 0
    tot = 0
    for yr in yrs:
        a, b = tab.get((yr, -35.0)), tab.get((yr, -24.46))
        if pd.isna(a) or pd.isna(b):
            continue
        tot += 1
        d_ = a - b
        win += d_ > 0
        print(f"  {yr}: {d_:+6.2f}pp {'✅' if d_ > 0 else '❌'}")
    print(f"  → {win}/{tot} 年更好")

    print("\n【3】三段")
    for lab, lo_, hi_ in (("2016-2020", 2016, 2020), ("2021-2023", 2021, 2023),
                          ("2024-2026", 2024, 2026)):
        line = f"  {lab}: "
        for dh in DHS:
            c = picks[dh]
            s = c[(c.trade_date // 10000 >= lo_) & (c.trade_date // 10000 <= hi_)]
            line += f" dh{dh:.0f} n={len(s):>3} {(s.y - COST).mean():>5.2f}%"
        print(line)

    print("\n【4】整体与分段卡玛")
    print(f"{'dh':>8}{'笔数':>6}{'单笔':>8}{'年化':>9}{'回撤':>9}{'卡玛':>7}"
          f"{'  2016-20':>10}{'2021-23':>10}{'2024-26':>10}")
    for dh in DHS:
        c = picks[dh]
        sim = base.copy()
        v = pd.Series((c.y.values - COST) * 0.40, index=c.trade_date.values).sort_index()
        sim[v.index] = sim[v.index] + v.values
        a, dd, cm = metrics(sim)
        segs = []
        for lo_, hi_ in ((2016, 2020), (2021, 2023), (2024, 2026)):
            s = sim[(sim.index // 10000 >= lo_) & (sim.index // 10000 <= hi_)]
            segs.append(metrics(s)[2])
        print(f"{dh:>8.1f}{len(c):>6}{(c.y - COST).mean():>8.2f}{a:>9.2f}{dd:>9.2f}"
              f"{cm:>7.2f}{segs[0]:>10.2f}{segs[1]:>10.2f}{segs[2]:>10.2f}")

    for dh in DHS:
        picks[dh].to_csv(OUT / f"w247g_{abs(dh):.0f}.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
