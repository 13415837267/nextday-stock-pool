"""第247轮 E: 两个还没碰过的旋钮

① 主策略与补位同一天都出信号时, 现在无条件让主策略做。谁该做?
   (若能证明补位日比主策略日强, 让位就是白捡的收益; 若不能, 这条也封掉)
② 破位阈值收紧 (idx_dist <= -3.5 是 240 轮挖出来的), 收紧到 -4/-5/-6 单笔会不会更高?
   (资金利用率会掉, 看卡玛的净效应)
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
    print("第247轮 E: 让位判定 / 阈值收紧")
    print("=" * 78)

    tr = pd.read_csv(OUT / "w232_trades.csv")
    tr["signal"] = tr["signal"].astype(int)
    cal = sorted(int(f.split("/")[-1].replace(".csv.gz", ""))
                 for f in glob.glob(str(ROOT / "data" / "panel" / "*.csv.gz")))
    cal = np.array([d for d in cal if d >= tr.signal.min()])
    base = pd.Series(0.0, index=cal)
    for dt, g in tr.groupby("signal"):
        base[int(dt)] = g["ret"].mean()

    # ---------- ① 重叠 ----------
    print("\n【1】主策略日 / 补位日 的重叠")
    # 补位日 = 全交易日里"未被主策略占用"的日子, 这里用 246 的 256 笔反推
    pk = pd.read_csv(OUT / "w247b_N1.csv")
    pk["trade_date"] = pk["trade_date"].astype(int)
    busy = set(int(x) for x in tr.signal)
    for _, r in tr.iterrows():
        busy.add(int(r["buy"]))
        busy.add(int(r["sell"]))
    print(f"  主策略 559 个信号日, 占 {len(busy)} 个日历日(含买卖日) "
          f"/ 共 {len(cal)} 天 → 空闲 {len(cal) - len(busy)} 天")
    print(f"  补位 256 笔 落在 {len(pk)} 天, 其中 {len(set(pk.trade_date) & busy)} 天与主策略抢占"
          f" (0 才对, 因为补位只在空仓日做)")

    # 反事实: 若允许补位抢主策略日, 需要那天的补位信号。这里只能估算上限:
    print(f"  主策略日均单笔 {tr.ret.mean():.2f}%  补位日均单笔 {(pk.y - COST).mean():.2f}%")
    print(f"  → 差距 {(pk.y - COST).mean() - tr.ret.mean():+.2f}pp/笔, "
          f"但补位样本 256 对 559, 且补位日都是'市场破位'的极端日")

    # ---------- ② 阈值收紧 ----------
    print("\n【2】破位阈值收紧 (个股条件保持 -24.46)")
    d = pd.read_parquet(OUT / "w246b_all.parquet",
                        columns=["ts_code", "trade_date", "listed", "n1_gap", "y",
                                 "amt_yi"] + DHC)
    d = d[d.listed >= 250]
    d["trade_date"] = d["trade_date"].astype("int64")
    d = d[(d.trade_date >= cal[0]) & (~d.trade_date.isin(busy))]
    mk = pd.read_csv(OUT / "feat234" / "market.csv",
                     usecols=["trade_date", "idx_ma20_dist", "limit_dn"])
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

    def sel(th: float, dh: float = -24.46) -> pd.DataFrame:
        n: dict[str, int] = {}
        out = []
        for dt in sorted(gb):
            g = gb[dt]
            top = g.iloc[0]
            broke = (top.idx_ma20_dist <= th) and (top.dh60 <= dh)
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
            out.append({"trade_date": dt, "ts_code": pick.ts_code, "y": pick.y,
                        "brk": broke, "idx": top.idx_ma20_dist})
        return pd.DataFrame(out)

    print(f"{'阈值':>7}{'笔数':>6}{'破位':>6}{'单笔':>8}{'胜率':>7}"
          f"{'年化':>9}{'回撤':>9}{'卡玛':>7}")
    rows = []
    for th in (-2.5, -3.0, -3.5, -4.0, -4.5, -5.0, -6.0):
        c = sel(th)
        y = c.y - COST
        sim = base.copy()
        v = pd.Series(y.values * 0.40, index=c.trade_date.values).sort_index()
        sim[v.index] = sim[v.index] + v.values
        a, dd, cm = metrics(sim)
        print(f"{th:>7.1f}{len(c):>6}{int(c.brk.sum()):>6}{y.mean():>8.2f}"
              f"{(y > 0).mean() * 100:>7.0f}{a:>9.2f}{dd:>9.2f}{cm:>7.2f}")
        rows.append({"阈值": th, "笔数": len(c), "单笔": round(y.mean(), 3),
                     "年化": a, "回撤": dd, "卡玛": cm})
    pd.DataFrame(rows).to_csv(OUT / "w247e_th.csv", index=False, encoding="utf-8-sig")

    print("\n【3】个股条件放宽 (阈值保持 -3.5)")
    print(f"{'dh60':>8}{'笔数':>6}{'单笔':>8}{'年化':>9}{'回撤':>9}{'卡玛':>7}")
    for dh in (-15.0, -20.0, -24.46, -30.0, -35.0):
        c = sel(-3.5, dh)
        y = c.y - COST
        sim = base.copy()
        v = pd.Series(y.values * 0.40, index=c.trade_date.values).sort_index()
        sim[v.index] = sim[v.index] + v.values
        a, dd, cm = metrics(sim)
        print(f"{dh:>8.1f}{len(c):>6}{y.mean():>8.2f}{a:>9.2f}{dd:>9.2f}{cm:>7.2f}")


if __name__ == "__main__":
    main()
