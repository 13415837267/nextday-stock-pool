"""第二百三十九轮 C: 唯一活下来的样本外条件(B1 暴跌日抗跌)到底牢不牢

239B 结果: A 组(样本内挖)补位单笔 +1.52%, B 组(样本外挖) +0.70%, 差 0.81pp
—— 过拟合程度第一次被量化。但 B1 单独看不但没退化, 还比 A5 略好(2.22% vs 1.98%),
补位后卡玛 6.24 > 基准 4.95。本轮专查它:
  ① 分年 ② 剔掉 2024 ③ 事件簇(是不是全靠两三次暴跌) ④ 尾部 ⑤ 最差单笔
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "output" / "feat234"
OUT = ROOT / "output"
TRADES = OUT / "w232_trades.csv"

COST = 0.10
COLS = ["ts_code", "trade_date", "y", "n1_ok", "pct", "rel_pct", "bias20", "ret10",
        "maxdd20", "vol20", "is_paused", "is_st", "listed", "circ_mv", "amp",
        "turnover_rate", "close"]


def load() -> pd.DataFrame:
    parts = []
    for p in sorted(FEAT.glob("f*.parquet")):
        d = pd.read_parquet(p, columns=COLS)
        d = d[(d["is_paused"] == 0) & (d["is_st"] == 0) & (d["n1_ok"] == True)]   # noqa: E712
        d = d[d["pct"].between(-11.5, 11.5) & d["y"].between(-40, 60) & (d["listed"] >= 250)]
        if len(d):
            parts.append(d)
    return pd.concat(parts, ignore_index=True)


def metrics(day_ret: pd.Series) -> tuple[float, float, float]:
    nav = (1 + day_ret / 100).cumprod()
    yrs = len(day_ret) / 244.0
    ann = nav.iloc[-1] ** (1 / yrs) - 1
    dd = (nav / nav.cummax() - 1).min()
    return round(ann * 100, 2), round(dd * 100, 2), round(ann * 100 / abs(dd * 100), 2)


def main() -> None:
    d = load()
    cal = np.sort(d["trade_date"].unique())
    cal = cal[cal >= 20190101]
    nxt = {cal[i]: cal[i + 1] for i in range(len(cal) - 1)}
    nxt[np.int64(cal[-1])] = np.int64(cal[-1])
    d = d[d["trade_date"].isin(cal)].copy()

    tr = pd.read_csv(TRADES)
    tr = tr[tr["signal"].isin(cal)]
    base = pd.Series(0.0, index=cal)
    for dt, g in tr.groupby("signal"):
        base[dt] = g["ret"].mean()
    busy = set()
    for _, r in tr.iterrows():
        busy.add(r["buy"]); busy.add(r["sell"])

    m = (d["rel_pct"] > 1.26) & ((d["vol20"] * d["pct"]) < -4.37)
    s = d[m]
    print(f"B1 条件内 {len(s):,} 样本 / {s['trade_date'].nunique()} 日 / "
          f"{len(s)/s['trade_date'].nunique():.1f} 只/日")

    # 补位逐笔
    picks = []
    for dt in s["trade_date"].unique():
        if dt in busy:
            continue
        bt, sl = nxt[dt], nxt[nxt[dt]]
        if bt in busy or sl in busy:
            continue
        g = s[s["trade_date"] == dt]
        r = g.loc[g["circ_mv"].idxmin()]
        picks.append({"signal": dt, "buy": bt, "sell": sl, "code": r["ts_code"],
                      "y": r["y"] - COST, "n_pool": len(g),
                      "mv": r["circ_mv"], "pct": r["pct"], "rel": r["rel_pct"]})
    p = pd.DataFrame(picks).sort_values("signal")
    p["yr"] = (p["signal"] // 10000).astype(int)
    print(f"\n补位 {len(p)} 笔  单笔 {p['y'].mean():+.3f}%  胜率 {(p['y']>0).mean()*100:.1f}%")

    print("\n① 分年")
    g = p.groupby("yr")["y"].agg(["size", "sum", "mean", "min"])
    g["胜率"] = p.groupby("yr")["y"].apply(lambda x: (x > 0).mean() * 100).round(0)
    print(g.round(2).to_string())

    print("\n② 剔除 2024 后")
    for drop in [[], [2024]]:
        q = p[~p["yr"].isin(drop)]
        sim = base.copy()
        sim[pd.Index(q["signal"])] = q["y"].values
        print(f"  剔除 {drop or '无'}: 补位 {len(q)} 笔 单笔 {q['y'].mean():+.3f}%  "
              f"年化 {metrics(sim)[0]}%  回撤 {metrics(sim)[1]}%  卡玛 {metrics(sim)[2]}")

    print("\n③ 单笔贡献集中度")
    v = p["y"].sort_values(ascending=False).values
    tot = v.sum()
    print(f"  总和 {tot:+.0f}pp  最大 1 笔占 {v[0]/tot*100:.0f}%  前 5 笔占 {v[:5].sum()/tot*100:.0f}%  "
          f"前 10 笔占 {v[:10].sum()/tot*100:.0f}%")
    print(f"  中位 {np.median(v):+.2f}%  最差 {v[-1]:+.2f}%  最差 5 笔 {v[-5:].round(2).tolist()}")

    print("\n④ 事件簇(相邻补位日间隔 <= 5 个交易日视为同一簇)")
    idx = np.searchsorted(cal, p["signal"].values)
    cl, cur = [], [0]
    for i in range(1, len(idx)):
        if idx[i] - idx[i - 1] <= 5:
            cur.append(i)
        else:
            cl.append(cur); cur = [i]
    cl.append(cur)
    cl = sorted(cl, key=lambda x: -p.iloc[x]["y"].sum())
    print(f"  共 {len(cl)} 簇, 前 5 大簇:")
    for c in cl[:5]:
        sub = p.iloc[c]
        print(f"    {sub['signal'].min()}~{sub['signal'].max()}  {len(c)} 笔  "
              f"合计 {sub['y'].sum():+.0f}pp  单笔 {sub['y'].mean():+.2f}%")
    top3 = sum(p.iloc[c]["y"].sum() for c in cl[:3])
    print(f"  前 3 簇占总贡献 {top3/tot*100:.0f}%")

    print("\n⑤ 补位日的大盘")
    mk = pd.read_csv(FEAT / "market.csv").rename(columns={"trade_date": "date"}).set_index("date")
    z = mk.reindex(p["signal"].values)
    print(f"  当日中位涨跌 {z['med_pct'].median():+.2f}%  "
          f"跌停数中位 {z['limit_dn'].median():.0f}  "
          f"指数 20 日线上占比 {(z['idx_ma20_dist']>0).mean()*100:.0f}%")

    p.to_csv(OUT / "w239c_b1_picks.csv", index=False, encoding="utf-8-sig")
    print("\n逐笔已存 output/w239c_b1_picks.csv")

    # ⑥ 逐年净值叠加图数据
    sim = base.copy()
    sim[pd.Index(p["signal"])] = p["y"].values
    print(f"\n最终: 基准 {metrics(base)}  ->  基准+B1 {metrics(sim)}")


if __name__ == "__main__":
    main()
