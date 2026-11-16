"""第 241 轮 C: 既然 W7 是 beta 规则, 就按 beta 的方式执行

probe_241b 的结论: 破位日 7 个选股键的 y_ex 全在 ±0.2pp, alpha 等于零。
那"按市值最小挑 1 只"多出来的 0.6pp 是小市值 beta 的加杠杆, 不是选股能力
——代价是最差单笔从 -9.87%(等权) 恶化到 -16.87%。

所以执行形态应该反过来: 越接近"买整个市场", beta 越纯、尾部风险越小。
本脚本把三种买法放到主策略框架里对比 (同一资金池, 同一 2500 日窗口):
  S1 最小市值 1 只 (240 轮做法)
  S2 最小市值 10 只
  S3 流动性前 10 只等权 (最接近大盘 beta)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "output" / "feat234"
OUT = ROOT / "output"
COST = 0.10
COLS = ["ts_code", "trade_date", "y", "n1_ok", "pct", "circ_mv", "idx_dist",
        "dist_high60", "amt_yi", "is_paused", "is_st", "listed"]


def M(r: pd.Series) -> tuple[float, float, float]:
    nav = (1 + r / 100).cumprod()
    ann = nav.iloc[-1] ** (244 / len(r)) - 1
    dd = (nav / nav.cummax() - 1).min()
    return round(ann * 100, 2), round(dd * 100, 2), round(ann * 100 / abs(dd * 100), 2)


def load() -> pd.DataFrame:
    parts = []
    for p in sorted(FEAT.glob("f*.parquet")):
        d = pd.read_parquet(p, columns=COLS)
        d = d[(d["is_paused"] == 0) & (d["is_st"] == 0) & (d["n1_ok"] == True)]      # noqa: E712
        d = d[d["pct"].between(-11.5, 11.5) & d["y"].between(-40, 60) & (d["listed"] >= 250)]
        d = d[d["ts_code"].str[:2].isin(["60", "00"])]
        if len(d):
            parts.append(d)
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    d = load()
    cal = np.sort(d["trade_date"].unique())
    tr = pd.read_csv(OUT / "w232_trades.csv")
    cal = cal[cal >= tr["signal"].min()]
    tr = tr[tr["signal"].isin(cal)]

    busy, sig = set(), set(tr["signal"])
    for _, r in tr.iterrows():
        busy.add(r["buy"])
        busy.add(r["sell"])
    free = [x for x in cal if x not in busy and x not in sig]

    base = pd.Series(0.0, index=cal)
    for dt, g in tr.groupby("signal"):
        base[dt] = g["ret"].mean()
    b = M(base)
    print(f"窗口 {cal[0]}~{cal[-1]} {len(cal)}日  基准 年化{b[0]}% 回撤{b[1]}% 卡玛{b[2]}")
    print(f"主策略 {len(tr)} 笔 / 空仓 {len(free)} 日")

    nxt = {cal[i]: cal[i + 1] for i in range(len(cal) - 1)}
    nxt[cal[-1]] = cal[-1]

    def days_for(ic: float, hc: float) -> list:
        s = d[(d["idx_dist"] <= ic) & (d["dist_high60"] <= hc)]
        out = []
        for dt in s["trade_date"].unique():
            if dt not in free:
                continue
            bd = nxt.get(dt)
            sd = nxt.get(bd, bd)
            if bd in busy or sd in busy:
                continue
            out.append(dt)
        return sorted(out)

    RULES = {"(-3.51,-24.46)": (-3.51, -24.46), "(-4.0,-28)": (-4.0, -28.0),
             "(-3.0,-21)": (-3.0, -21.0)}
    PICKS = {
        "S1 最小市值1只": lambda g: g.nsmallest(1, "circ_mv")["y"].mean(),
        "S2 最小市值10只": lambda g: g.nsmallest(min(10, len(g)), "circ_mv")["y"].mean(),
        "S3 流动性前10只": lambda g: g.nlargest(min(10, len(g)), "amt_yi")["y"].mean(),
        "S4 全命中等权": lambda g: g["y"].mean(),
    }

    print("\n" + "=" * 92)
    rows = []
    picks_store = {}
    for rn, (ic, hc) in RULES.items():
        days = days_for(ic, hc)
        sub = d[d["trade_date"].isin(days)]
        v_all = {}
        for pn, fn in PICKS.items():
            v = {dt: fn(g) - COST for dt, g in sub.groupby("trade_date")}
            v = pd.Series(v)
            v_all[pn] = v
            for w in (0.3, 1.0):
                sim = base.copy()
                sim[v.index] += v.values * w
                a, dd, cm = M(sim)
                rows.append({"规则": rn, "买法": pn, "仓位": f"{w:.0%}", "次数": len(v),
                             "单笔": round(v.mean(), 3), "最差": round(v.min(), 2),
                             "年化": a, "回撤": dd, "卡玛": cm, "比基准": round(cm - b[2], 2)})
        picks_store[rn] = v_all
    r = pd.DataFrame(rows)
    print(r.to_string(index=False))
    r.to_csv(OUT / "w241c_buyform.csv", index=False, encoding="utf-8-sig")

    # 主推: 固定阈值 + 流动性前 10 等权 + 30% 仓
    v = picks_store["(-3.51,-24.46)"]["S3 流动性前10只"]
    print("\n" + "=" * 92)
    print("主推组合 (-3.51,-24.46) + 流动性前10等权, 分年:")
    p = v.reset_index()
    p.columns = ["signal", "y"]
    p["yr"] = p["signal"] // 10000
    g = p.groupby("yr")["y"].agg(["size", "sum", "mean"])
    g["win"] = p.groupby("yr")["y"].apply(lambda x: (x > 0).mean() * 100).round(0)
    print(g.round(2).to_string())
    print(f"合计 {len(p)} 次 单笔 {p.y.mean():+.3f}% 胜率 {(p.y > 0).mean() * 100:.1f}% 最差 {p.y.min():.2f}%")

    for w in (0.2, 0.3, 0.4, 0.5, 0.7, 1.0):
        sim = base.copy()
        sim[v.index] += v.values * w
        a, dd, cm = M(sim)
        print(f"  仓位 {w:.0%}: 年化{a}% 回撤{dd}% 卡玛{cm}  (基准 {b[0]}/{b[1]}/{b[2]})")

    p.to_csv(OUT / "w241c_picks.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
