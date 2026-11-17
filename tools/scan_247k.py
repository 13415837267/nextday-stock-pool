"""第247轮 K: 定稿 —— 正确口径 + 深低开扩池

三轮回溯:
  245 报 5.49: 破位判定干净(池子层面), 但样本漏了每年 1 月(31/256 笔)
  246 报 5.16: 补回 1 月, 但 sel() 用"明天会低开的票"里的第一名去判破位 → 未来信息
  247 修正: 破位用 T 日全池第一名(idx_dist/dh60), 深低开用 T+1 9:25 全池排名前 N 且崩 5~9.5%
定稿在 N=10 (61 笔深低开) 上, 取平台中心而不是角点。
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
DH = -35.0
N_DEEP = 10


def metrics(r: pd.Series) -> tuple[float, float, float]:
    nav = (1 + r / 100).cumprod()
    ann = nav.iloc[-1] ** (244 / len(r)) - 1
    dd = (nav / nav.cummax() - 1).min()
    return round(ann * 100, 2), round(dd * 100, 2), round(ann * 100 / abs(dd * 100), 2)


def main() -> None:
    print("=" * 78)
    print("第247轮 K: 定稿")
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
        parts.append(g)
    full = pd.concat(parts, ignore_index=True)
    del parts, d
    full = full.sort_values(["trade_date", "mr", "amt_yi"], ascending=[True, True, False])
    full["rk"] = full.groupby("trade_date").cumcount() + 1
    gb = {int(k): g for k, g in full.groupby("trade_date")}
    tops = {k: v.iloc[0] for k, v in gb.items()}

    def sel(N: int, dh: float = DH) -> pd.DataFrame:
        n: dict[str, int] = {}
        out = []
        for dt in sorted(gb):
            g = gb[dt]
            t1 = tops[dt]
            broke = (t1.idx_ma20_dist <= -3.5) and (t1.dh60 <= dh)
            if broke:
                pick, path = None, "破位"
                for j in range(len(g)):
                    rr = g.iloc[j]
                    if float(rr.n1_gap) <= 0.05 and n.get(rr.ts_code, 0) < 8:
                        pick = rr
                        break
            else:
                cc = g[(g.n1_gap > -9.5) & (g.n1_gap <= -5.0) & (g.rk <= N)]
                pick, path = None, "深低开"
                for j in range(len(cc)):
                    rr = cc.iloc[j]
                    if n.get(rr.ts_code, 0) < 8:
                        pick = rr
                        break
            if pick is None:
                continue
            n[pick.ts_code] = n.get(pick.ts_code, 0) + 1
            out.append({"trade_date": dt, "ts_code": pick.ts_code, "y": pick.y,
                        "n1_gap": float(pick.n1_gap), "brk": broke, "path": path,
                        "idx": float(t1.idx_ma20_dist), "rk": int(pick.rk),
                        "yr": dt // 10000, "m": dt // 100 % 100})
        return pd.DataFrame(out)

    c = sel(N_DEEP)
    c.to_csv(OUT / "w247_picks.csv", index=False, encoding="utf-8-sig")
    print(f"定稿 {len(c)} 笔 (破位 {int(c.brk.sum())} / 深低开 {int((~c.brk).sum())})  "
          f"单笔 {(c.y - COST).mean():.2f}%  胜率 {((c.y - COST) > 0).mean() * 100:.0f}%")

    print("\n【1】仓位 (P1 = P2)")
    print(f"{'P':>6}{'年化':>9}{'回撤':>9}{'卡玛':>7}")
    for P in (0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7):
        sim = base.copy()
        v = pd.Series((c.y.values - COST) * P, index=c.trade_date.values).sort_index()
        sim[v.index] = sim[v.index] + v.values
        print(f"{P:>6.0%}{metrics(sim)[0]:>9.2f}{metrics(sim)[1]:>9.2f}{metrics(sim)[2]:>7.2f}")

    print("\n【2】路径分仓 P1/P2 (P1=40% 固定, 扫 P2)")
    print(f"{'P2':>6}{'年化':>9}{'回撤':>9}{'卡玛':>7}")
    for P2 in (0.0, 0.2, 0.3, 0.4, 0.5, 0.7):
        sim = base.copy()
        for path, P in (("破位", 0.40), ("深低开", P2)):
            s = c[c.path == path]
            if not len(s) or P <= 0:
                continue
            v = pd.Series((s.y.values - COST) * P, index=s.trade_date.values).sort_index()
            sim[v.index] = sim[v.index] + v.values
        print(f"{P2:>6.0%}{metrics(sim)[0]:>9.2f}{metrics(sim)[1]:>9.2f}{metrics(sim)[2]:>7.2f}")

    print("\n【3】定稿分年 (40% 仓)")
    sim = base.copy()
    v = pd.Series((c.y.values - COST) * 0.40, index=c.trade_date.values).sort_index()
    sim[v.index] = sim[v.index] + v.values
    ann, dd, cm = metrics(sim)
    yr = sim.groupby(sim.index // 10000).apply(lambda x: (1 + x / 100).prod() - 1) * 100
    for k, val in yr.items():
        s = c[c.yr == k]
        print(f"  {k}: {val:+7.1f}%   {len(s):>3} 笔  单笔 "
              f"{(s.y - COST).mean() if len(s) else 0:>5.2f}%")
    print(f"  → 年化 {ann:.2f}%  回撤 {dd:.2f}%  卡玛 {cm:.2f}  正年 {int((yr > 0).sum())}/{len(yr)}")

    print("\n【4】四轮口径对照 (同一根主策略基线 86.42/-24.24/3.57)")
    print(f"  245 报 5.49  ← 漏 1 月(31 笔)                    同口径重算 4.03(247 笔)")
    print(f"  246 报 5.16  ← 用'明天低开的票'判破位(未来信息)    同口径重算 4.03(247 笔)")
    print(f"  247 定稿 {cm:.2f}  ← 正确口径 + 深低开扩到前 {N_DEEP} 名 ({len(c)} 笔)")

    print("\n【5】集中度与稳健性")
    print(f"  票数 {c.ts_code.nunique()}  最长连买 {int(c.groupby('ts_code').size().max())} 次  "
          f"前5票 {c.ts_code.value_counts().head(5).sum() / len(c) * 100:.0f}%")
    dp = c[c.path == "深低开"]
    print(f"  深低开 {len(dp)} 笔 单笔 {(dp.y - COST).mean():.2f}% "
          f"t={(dp.y - COST).mean() / ((dp.y - COST).std(ddof=1) / np.sqrt(len(dp))):.2f}  "
          f"前段 {((dp[dp.yr <= 2020].y - COST).mean()):.2f}% "
          f"后段 {((dp[dp.yr >= 2021].y - COST).mean()):.2f}%")
    br = c[c.path == "破位"]
    print(f"  破位   {len(br)} 笔 单笔 {(br.y - COST).mean():.2f}% "
          f"t={(br.y - COST).mean() / ((br.y - COST).std(ddof=1) / np.sqrt(len(br))):.2f}")

    sim.to_frame("r").to_csv(OUT / "w247_nav.csv", encoding="utf-8-sig")


if __name__ == "__main__":
    main()
