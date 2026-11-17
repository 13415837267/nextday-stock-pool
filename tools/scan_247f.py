"""第247轮 F: 两个尾巴

① dh60 从 -24.46 放宽到 -35 让卡玛升到 5.49 —— 只筛掉 13 笔就升 0.33, 太像噪声。
   继续放到 -40/-50/-60, 若一路升 = 这条件本就该删; 若 -35 是峰 = 尖峰过拟合。
② 深低开门槛 (-9.5%, -5%] 只有 24 次。放宽下界到 -3%/-2% 能多做多少天? 单笔掉多少?
   (空闲 1175 天里还有 919 天完全没信号, 这是唯一的扩容方向)
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
    print("第247轮 F: dh60 放宽的边界 / 深低开门槛放宽")
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
    mk = pd.read_csv(OUT / "feat234" / "market.csv",
                     usecols=["trade_date", "idx_ma20_dist"])
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
    lo["rk"] = lo.groupby("trade_date").cumcount() + 1
    gb = {int(k): g for k, g in lo.groupby("trade_date")}
    print(f"低开票池 {len(lo):,} 行 / {len(gb)} 天")
    # 每天被选中的 top1 的缺口分布 —— 看深低开区间到底有多稀有
    topg = pd.Series([g.iloc[0].n1_gap for g in gb.values()])
    print(f"每日 top1 的缺口分位: " +
          "  ".join(f"P{int(q*100)}={topg.quantile(q):.2f}" for q in (0.5, 0.9, 0.95, 0.99)))
    print(f"  gap <= -5 的 top1: {int((topg <= -5).sum())} 天   "
          f"gap <= -3 的: {int((topg <= -3).sum())} 天   "
          f"gap <= -2 的: {int((topg <= -2).sum())} 天")

    def sel(th: float = -3.5, dh: float = -24.46,
            glo: float = -9.5, ghi: float = -5.0) -> pd.DataFrame:
        n: dict[str, int] = {}
        out = []
        for dt in sorted(gb):
            g = gb[dt]
            top = g.iloc[0]
            broke = (top.idx_ma20_dist <= th) and (top.dh60 <= dh)
            gap = float(top.n1_gap)
            deep = (glo < gap <= ghi)
            if not (broke or deep):
                continue
            pick = top
            for j in range(len(g)):
                rr = g.iloc[j]
                if n.get(rr.ts_code, 0) < 8:
                    pick = rr
                    break
            n[pick.ts_code] = n.get(pick.ts_code, 0) + 1
            out.append({"trade_date": dt, "ts_code": pick.ts_code, "y": pick.y,
                        "brk": broke, "path": "破位" if broke else "深低开",
                        "idx": top.idx_ma20_dist})
        return pd.DataFrame(out)

    def run(c: pd.DataFrame, P: float = 0.40) -> tuple[float, float, float]:
        sim = base.copy()
        v = pd.Series((c.y.values - COST) * P, index=c.trade_date.values).sort_index()
        sim[v.index] = sim[v.index] + v.values
        return metrics(sim)

    print("\n【1】dh60 一路放宽 (阈值 -3.5, 深低开区间不变)")
    print(f"{'dh60':>8}{'笔数':>6}{'单笔':>8}{'年化':>9}{'回撤':>9}{'卡玛':>7}")
    for dh in (-20.0, -24.46, -28.0, -32.0, -35.0, -40.0, -50.0, -60.0, -100.0):
        c = sel(dh=dh)
        a, dd, cm = run(c)
        print(f"{dh if dh > -99 else '删除':>8}{len(c):>6}{(c.y - COST).mean():>8.2f}"
              f"{a:>9.2f}{dd:>9.2f}{cm:>7.2f}")

    print("\n【2】深低开区间放宽 (无个股条件 dh60<=-24.46 只作用于破位路径)")
    print(f"{'区间':>14}{'深低开笔数':>10}{'单笔':>8}{'总笔数':>8}"
          f"{'年化':>9}{'回撤':>9}{'卡玛':>7}")
    rows = []
    for glo, ghi, tag in ((-9.5, -5.0, "-9.5~-5"), (-9.5, -4.0, "-9.5~-4"),
                          (-9.5, -3.0, "-9.5~-3"), (-9.5, -2.0, "-9.5~-2"),
                          (-7.0, -3.0, "-7~-3"), (-5.0, -3.0, "-5~-3"),
                          (-4.0, -2.0, "-4~-2"), (-9.5, -9.5, "关闭")):
        c = sel(glo=glo, ghi=ghi)
        dp = c[c.path == "深低开"]
        yy = dp.y - COST
        a, dd, cm = run(c)
        print(f"{tag:>14}{len(dp):>10}{(yy.mean() if len(dp) else np.nan):>8.2f}"
              f"{len(c):>8}{a:>9.2f}{dd:>9.2f}{cm:>7.2f}")
        rows.append({"区间": tag, "深低开笔数": len(dp),
                     "单笔": round(yy.mean(), 2) if len(dp) else 0,
                     "总笔数": len(c), "年化": a, "回撤": dd, "卡玛": cm})
        c.to_csv(OUT / f"w247f_{tag.replace('~', '_')}.csv", index=False,
                 encoding="utf-8-sig")
    pd.DataFrame(rows).to_csv(OUT / "w247f_deep.csv", index=False, encoding="utf-8-sig")

    print("\n【3】dh60=-35 时到底筛掉了哪 13 笔")
    c1 = sel(dh=-24.46)
    c2 = sel(dh=-35.0)
    gone = c1[~c1.trade_date.isin(set(c2.trade_date)) | ~c1.ts_code.isin(set(c2.ts_code))]
    gone = c1.merge(c2[["trade_date", "ts_code"]], on=["trade_date", "ts_code"],
                    how="left", indicator=True)
    g = gone[gone._merge == "left_only"]
    print(f"  消失 {len(g)} 笔  单笔 {(g.y - COST).mean():.2f}%  合计 "
          f"{(g.y - COST).sum():.0f}pp  胜率 {(g.y > COST).mean() * 100:.0f}%")
    print(g[["trade_date", "ts_code", "y", "brk"]].sort_values("trade_date").to_string(index=False))


if __name__ == "__main__":
    main()
