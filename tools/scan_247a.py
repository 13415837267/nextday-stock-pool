"""第247轮 A: 两条触发路径分开定仓位

246 定稿 256 笔 = 破位 232 笔 + 深低开 24 笔, 两边同用 40% 仓。
但两者的统计强度差一个数量级:
  破位   232 笔 单笔 1.86%  t=4.49
  深低开  24 笔 单笔 6.81%  t=1.48   剔掉最大一笔(002217 +102.6pp)后只剩 2.66%
深低开 24 笔里 2024-2025 两年占 7 笔 / 合计 +124pp, 靠极少数样本。

本脚本把仓位拆成两个独立旋钮: 破位日 P1、深低开日 P2, 看卡玛面。
若 P2 最优点远低于 P1, 说明市场早就"定价"了这条弱信号。
"""

from __future__ import annotations

import glob

import numpy as np
import pandas as pd

OUT = __import__("pathlib").Path("output")
COST = 0.30          # 双边佣金+印花税+滑点(pp)
YEARS_PER = 244


def metrics(r: pd.Series) -> tuple[float, float, float]:
    nav = (1 + r / 100).cumprod()
    ann = nav.iloc[-1] ** (YEARS_PER / len(r)) - 1
    dd = (nav / nav.cummax() - 1).min()
    return ann * 100, dd * 100, ann * 100 / abs(dd * 100)


def main() -> None:
    print("=" * 78)
    print("第247轮 A: 破位仓 P1 × 深低开仓 P2")
    print("=" * 78)

    picks = pd.read_csv(OUT / "w246k_picks.csv")
    picks["trade_date"] = picks["trade_date"].astype(int)

    tr = pd.read_csv(OUT / "w232_trades.csv")
    cal = sorted(int(f.split("/")[-1].replace(".csv.gz", ""))
                 for f in glob.glob("data/panel/*.csv.gz"))
    cal = np.array([d for d in cal if d >= tr.signal.min()])
    base = pd.Series(0.0, index=cal)
    for dt, g in tr.groupby("signal"):
        base[int(dt)] = g["ret"].mean()

    print(f"\n主策略 {len(tr)} 笔 / {tr.signal.nunique()} 个信号日, "
          f"日历 {cal[0]}~{cal[-1]} 共 {len(cal)} 天")
    a0, d0, c0 = metrics(base)
    print(f"仅主策略: 年化 {a0:.2f}%  回撤 {d0:.2f}%  卡玛 {c0:.2f}")

    br = picks[picks.brk]
    dp = picks[~picks.brk]
    print(f"破位 {len(br)} 笔  深低开 {len(dp)} 笔")

    print("\n【1】单条路径单独叠加(只用一条路径的仓位)")
    print(f"{'路径':<8}{'仓':>5}{'笔数':>6}{'年化':>9}{'回撤':>9}{'卡玛':>7}")
    for nm, d in (("破位", br), ("深低开", dp)):
        for P in (0.2, 0.3, 0.4, 0.5, 0.7, 1.0):
            s = pd.Series((d.y.values - COST) * P, index=d.trade_date.values).sort_index()
            sim = base.copy()
            sim[s.index] = sim[s.index] + s.values
            a, dd, cm = metrics(sim)
            print(f"{nm:<8}{P:>5.0%}{len(d):>6}{a:>9.2f}{dd:>9.2f}{cm:>7.2f}")

    print("\n【2】P1(破位) × P2(深低开) 卡玛面")
    p1s = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    p2s = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
    tab = pd.DataFrame(index=[f"{p:.0%}" for p in p1s],
                       columns=[f"{p:.0%}" for p in p2s], dtype=float)
    rows = []
    for P1 in p1s:
        for P2 in p2s:
            sim = base.copy()
            for d, P in ((br, P1), (dp, P2)):
                if P <= 0 or not len(d):
                    continue
                s = pd.Series((d.y.values - COST) * P,
                              index=d.trade_date.values).sort_index()
                sim[s.index] = sim[s.index] + s.values
            a, dd, cm = metrics(sim)
            tab.loc[f"{P1:.0%}", f"{P2:.0%}"] = cm
            rows.append({"P1": P1, "P2": P2, "年化": round(a, 2),
                         "回撤": round(dd, 2), "卡玛": round(cm, 2)})
    print(tab.round(2).to_string())
    pd.DataFrame(rows).to_csv(OUT / "w247a_grid.csv", index=False, encoding="utf-8-sig")

    print("\n【3】同 P1=40% 时 P2 的完整曲线(看有没有平台)")
    for P2 in p2s:
        r = next(x for x in rows if abs(x["P1"] - 0.4) < 1e-9 and abs(x["P2"] - P2) < 1e-9)
        print(f"  P2={P2:>4.0%}  年化{r['年化']:>7.2f}  回撤{r['回撤']:>7.2f}  卡玛{r['卡玛']:>5.2f}")

    print("\n【4】深低开 24 笔的稳健性")
    y = dp.y - COST
    t = y.mean() / (y.std(ddof=1) / np.sqrt(len(y)))
    print(f"  单笔 {y.mean():.2f}%  中位 {y.median():.2f}%  t={t:.2f}")
    print(f"  剔最大一笔后 {y.drop(y.idxmax()).mean():.2f}%  "
          f"剔最大三笔后 {y.nlargest(3).index.to_series().pipe(lambda s: y.drop(s.index)).mean():.2f}%")
    for lab, lo, hi in (("2016-2020", 2016, 2020), ("2021-2026", 2021, 2026)):
        e = dp[(dp.yr >= lo) & (dp.yr <= hi)]
        print(f"  {lab}: {len(e)} 笔 单笔 {(e.y - COST).mean():.2f}%  合计 {(e.y - COST).sum():.0f}pp")

    print("\n【5】P1=40% 时, 深低开如果完全不做(等价 246 的 A 方案)")
    sim = base.copy()
    s = pd.Series((br.y.values - COST) * 0.40, index=br.trade_date.values).sort_index()
    sim[s.index] = sim[s.index] + s.values
    a, dd, cm = metrics(sim)
    print(f"  年化 {a:.2f}%  回撤 {dd:.2f}%  卡玛 {cm:.2f}  (246 报的 A 是 119.1/4.67, "
          "差异来自 246 的 A 用了 232 笔但选股/日期略不同)")

    print("\n【6】定稿候选: P1=40% / P2=20% 与 P1=40%/P2=40% 的分年")
    for P2 in (0.0, 0.2, 0.4):
        sim = base.copy()
        for d, P in ((br, 0.40), (dp, P2)):
            if P <= 0:
                continue
            s = pd.Series((d.y.values - COST) * P, index=d.trade_date.values).sort_index()
            sim[s.index] = sim[s.index] + s.values
        yr = sim.groupby(sim.index // 10000).apply(lambda x: (1 + x / 100).prod() - 1) * 100
        a, dd, cm = metrics(sim)
        print(f"  P2={P2:.0%}: 年化{a:>7.2f} 回撤{dd:>7.2f} 卡玛{cm:>5.2f}  "
              f"正年 {int((yr > 0).sum())}/{len(yr)}  " +
              " ".join(f"{k}:{v:+.0f}" for k, v in yr.items()))


if __name__ == "__main__":
    main()
