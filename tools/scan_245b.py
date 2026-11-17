"""第245轮 B: 被放弃的 82 天, 能不能救回来

244 轮实盘口径: 198 次信号里 82 次 T+1 高开(41%), 只能空仓。
问题是: 高开说明市场已经在反弹了, 这天真的没得做吗?
或者反过来 —— 破位日次日高开, 该买的是"还在跌的"还是"已经反弹的"?

用不过滤 n1_ok 的池子(pool2), 把 198 天劈成两半, 各自看:
  低开组 (n1_ok=True)  -> 244 现行: 买 9 窗口平均排名最靠前(跌最透)的
  高开组 (n1_ok=False) -> 未知: 分位扫描, 看买强还是买弱, 还是干脆不买
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
POOL = OUT / "w244_pool2"
WINS = (60, 90, 120, 150, 180, 200, 250, 300, 360)
COST = 0.10


def main() -> None:
    sig = pd.read_parquet(POOL / "signals.parquet")
    sig["trade_date"] = sig["trade_date"].astype("int64")
    w = sig.copy()
    for W in WINS:
        px = pd.read_parquet(POOL / f"W{W}.parquet").rename(columns={"dh": f"dh{W}"})
        w = w.merge(px, on=["ts_code", "trade_date"], how="left")
        del px
    dhcols = [f"dh{W}" for W in WINS]
    w = w.dropna(subset=dhcols)
    w["n1_ok"] = w["n1_ok"].fillna(False)

    # 9 窗口平均排名
    def add_rank(g: pd.DataFrame) -> pd.DataFrame:
        den = {c: max(int(g[c].notna().sum()), 1) for c in dhcols}
        rk = pd.DataFrame({c: g[c].rank(ascending=True) / den[c] for c in dhcols})
        g = g.copy()
        g["mr"] = rk.mean(axis=1, skipna=True)          # 0=跌最透, 1=最强
        return g

    w = pd.concat([add_rank(g) for _, g in w.groupby("trade_date")], ignore_index=True)
    w["y_net"] = w["y"] - COST
    print(f"候选 {len(w):,} 行 / {w.trade_date.nunique()} 天")

    print("\n【1】低开 vs 高开, 全体均值")
    for k, lab in ((True, "低开 n1_ok=True"), (False, "高开 n1_ok=False")):
        s = w[w.n1_ok == k]
        x = s["y_net"].dropna()
        print(f"  {lab:<18} 票数 {len(x):>6}  单笔 {x.mean():>6.2f}%  胜率 {(x > 0).mean() * 100:>5.1f}%")

    print("\n【2】按 9 窗口平均排名分五档 (0=跌最透, 1=最强)")
    print(f"{'档':<6}{'低开 n':>8}{'低开 单笔':>10}{'低开 胜率':>10}"
          f"{'高开 n':>8}{'高开 单笔':>10}{'高开 胜率':>10}")
    w["q"] = (w.groupby(["trade_date", "n1_ok"])["mr"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 5, labels=False, duplicates="drop")))
    rows = []
    for q in range(5):
        line = f"{q:<6}"
        for k in (True, False):
            s = w[(w.q == q) & (w.n1_ok == k)]["y_net"].dropna()
            line += f"{len(s):>8}{s.mean():>10.2f}{(s > 0).mean() * 100:>9.1f}%"
            rows.append({"q": q, "n1_ok": k, "n": len(s), "mean": round(s.mean(), 3),
                         "win": round((s > 0).mean() * 100, 1)})
        print(line)
    pd.DataFrame(rows).to_csv(OUT / "w245_quintile.csv", index=False, encoding="utf-8-sig")

    print("\n【3】每天各选一只, 四种组合的逐日序列 (198 天)")
    byday = {dt: g for dt, g in w.groupby("trade_date")}
    picks = {"低开买最弱": [], "低开买最强": [], "高开买最弱": [], "高开买最强": []}
    for dt in sorted(byday):
        g = byday[dt]
        lo = g[g.n1_ok]
        hi = g[~g.n1_ok]
        if len(lo):
            picks["低开买最弱"].append(lo.nsmallest(1, "mr")["y_net"].iloc[0])
            picks["低开买最强"].append(lo.nlargest(1, "mr")["y_net"].iloc[0])
        if len(hi):
            picks["高开买最弱"].append(hi.nsmallest(1, "mr")["y_net"].iloc[0])
            picks["高开买最强"].append(hi.nlargest(1, "mr")["y_net"].iloc[0])
    for k, v in picks.items():
        a = np.array(v, dtype=float)
        a = a[~np.isnan(a)]
        print(f"  {k:<12} n={len(a):>4}  单笔 {a.mean():>6.2f}%  "
              f"中位 {np.median(a):>6.2f}%  胜率 {(a > 0).mean() * 100:>5.1f}%  "
              f"合计 {a.sum():>7.1f}pp")

    print("\n【4】高开日最弱的一档, 分年表现")
    hi = w[~w.n1_ok].copy()
    hi["yr"] = hi.trade_date // 10000
    top = hi[hi.q == 0]
    t = top.groupby("yr")["y_net"].agg(["size", "mean"]).round(2)
    print(t.to_string())


if __name__ == "__main__":
    main()
