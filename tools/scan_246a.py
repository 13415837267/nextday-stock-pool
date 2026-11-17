"""第246轮 A: 245 定稿的 198 笔, 按「9:25 挂单前的真实约束」重审

两个必须回答的问题:

【1】滑点。245 假设「看到 T+1 开盘价, 就按开盘价成交」。真实机制是:
     9:15-9:20 可撤单(此时只能看到虚拟开盘价, 且会变)
     9:20-9:25 不可撤单
     9:25 撮合出真实开盘价
   所以想「按开盘价买到」必须 9:20 之前挂限价单, 那时你还不知道真实开盘价;
   想「看到真实开盘价再买」就必须 9:25 之后下单, 那是连续竞价, 会吃到漂移。
   两头都不完美。本轮直接对成交价打折, 看策略还剩多少。

【2】低开幅度。低开 -0.5% 和低开 -9% 不是一回事。-9% 往往是跌停开/利空,
   反转失败概率高。245 里「低开」是一个二值门槛, 没分深浅。

数据: feat234 的 n1_gap(相对 T 收的缺口%), idx_dist, amt_yi, circ_mv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
FEAT = OUT / "feat234"


def metrics(r: pd.Series) -> tuple[float, float, float]:
    nav = (1 + r / 100).cumprod()
    ann = nav.iloc[-1] ** (244 / len(r)) - 1
    dd = (nav / nav.cummax() - 1).min()
    return round(ann * 100, 2), round(dd * 100, 2), round(ann * 100 / abs(dd * 100), 2)


def base_series() -> tuple[pd.Series, np.ndarray]:
    tr = pd.read_csv(OUT / "w232_trades.csv")
    all_dates = sorted(int(p.stem.replace(".csv", "")) for p in
                       (ROOT / "data" / "panel").glob("*.csv.gz"))
    cal = np.array([d for d in all_dates if d >= tr.signal.min()])
    base = pd.Series(0.0, index=cal)
    for dt, g in tr.groupby("signal"):
        base[int(dt)] = g["ret"].mean()
    return base, cal


def main() -> None:
    base, cal = base_series()
    pk = pd.read_csv(OUT / "w245_picks.csv")
    pk.columns = ["signal", "y"]
    pk["signal"] = pk["signal"].astype(int)

    # ---- 取 n1_gap / idx_dist / circ_mv ----
    need = pk.signal.unique()
    parts = []
    for f in sorted(FEAT.glob("f*.parquet")):
        d = pd.read_parquet(f, columns=["ts_code", "trade_date", "n1_gap",
                                        "idx_dist", "amt_yi", "circ_mv", "pct"])
        d["trade_date"] = d["trade_date"].astype("int64")
        parts.append(d[d.trade_date.isin(need)])
    f = pd.concat(parts, ignore_index=True)
    print(f"特征切片 {len(f)} 行")

    # 245 的选股代码没存, 用池子重算一遍
    POOL = OUT / "w244_pool2"
    WINS = (60, 90, 120, 150, 180, 200, 250, 300, 360)
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
    den = w.groupby("trade_date")[dhcols].transform("count")
    w["mr"] = (w[dhcols].rank(ascending=True) / den).mean(axis=1, skipna=True)
    w = w.sort_values(["trade_date", "mr", "amt_yi"], ascending=[True, True, False])
    # ★ 必须带冷却: 245 定稿是 cd=10, 冷却后若无可选则退回不排除(宁可重复不空仓)
    dayrank = {int(d): i for i, d in enumerate(cal)}
    last: dict[str, int] = {}
    rec = []
    for dt, g in w.groupby("trade_date"):
        lo = g[g.n1_ok]
        cand = lo if len(lo) else g
        ok = [(c not in last) or (dayrank[int(dt)] - last[c] >= 10)
              for c in cand["ts_code"]]
        sub = cand[np.array(ok)]
        if len(sub):
            cand = sub
        r = cand.iloc[0]
        if pd.isna(r["mr"]) or pd.isna(r["y"]):
            continue
        last[r["ts_code"]] = dayrank[int(dt)]
        rec.append({"trade_date": int(dt), "ts_code": r["ts_code"], "y": float(r["y"])})
    lo = pd.DataFrame(rec)
    print(f"低开组内 mr 最小(冷却10): {len(lo)} 天, 单笔 {lo.y.mean():.3f}")

    m = lo.merge(f, on=["ts_code", "trade_date"], how="left")
    m = m.merge(pk.rename(columns={"signal": "trade_date"}), on="trade_date",
                how="left", suffixes=("", "_p"))
    assert len(m) == 198, len(m)
    print(f"合并 {len(m)} 行, n1_gap 缺失 {m.n1_gap.isna().sum()}")

    print("\n【1】滑点敏感性 (T+2 收盘卖, 单位 pp)")
    print(f"{'滑点':>6}{'单笔':>8}{'年化':>9}{'回撤':>9}{'卡玛':>7}{'净赚天数':>9}")
    rows = []
    for slip in (0.0, 0.1, 0.2, 0.3, 0.5, 0.8):
        v = (m["y"].values - slip)
        s = pd.Series(v, index=m.trade_date.values).sort_index()
        for P in (0.5, 0.7):
            sim = base.copy()
            sim[s.index] += s.values * P
            a, dd, cm = metrics(sim)
            if P == 0.5:
                print(f"{slip:>6.1f}{s.mean():>8.3f}{a:>9.2f}{dd:>9.2f}{cm:>7.2f}"
                      f"{(s > 0).sum():>9}")
            rows.append({"滑点": slip, "仓位": P, "单笔": round(s.mean(), 3),
                         "年化": a, "回撤": dd, "卡玛": cm})
    pd.DataFrame(rows).to_csv(OUT / "w246a_slip.csv", index=False, encoding="utf-8-sig")

    print("\n【2】低开幅度分档 (滑点 0.2)")
    m["g"] = pd.cut(m.n1_gap, [-11, -5, -3, -2, -1, -0.5, 0.06],
                    labels=["<-5", "-5~-3", "-3~-2", "-2~-1", "-1~-0.5", "-0.5~0"])
    g = m.groupby("g", observed=True).agg(
        n=("y", "size"), 单笔=("y", "mean"),
        胜率=("y", lambda x: (x > 0).mean() * 100),
        中位=("y", "median"))
    g["分年正"] = m.groupby("g", observed=True).apply(
        lambda x: ((x.groupby(x.trade_date // 10000).y.mean()) > 0).mean() * 100,
        include_groups=False).round(0)
    print(g.round(2).to_string())

    print("\n【3】指数位置 x 低开票数量 (T 日收盘即可知)")
    m["nlow_b"] = pd.qcut(m.groupby("trade_date").ngroup().map(
        lambda _: 0), 1) if False else None
    print("  idx_dist 分档:")
    m["ib"] = pd.cut(m.idx_dist, [-99, -8, -6, -5, -4.5, -4, -3.5],
                     labels=["<-8", "-8~-6", "-6~-5", "-5~-4.5", "-4.5~-4", "-4~-3.5"])
    g2 = m.groupby("ib", observed=True).agg(
        n=("y", "size"), 单笔=("y", "mean"), 胜率=("y", lambda x: (x > 0).mean() * 100))
    print(g2.round(2).to_string())

    m.to_csv(OUT / "w246a_detail.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
