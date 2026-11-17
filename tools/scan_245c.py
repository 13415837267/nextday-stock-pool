"""第245轮 C: 244 的实盘口径搞保守了 —— 应该"T+1 开盘时重新选", 而不是"T日选好再祈祷低开"

244 轮的做法: T 日收盘按 9 窗口平均排名选出 top1, T+1 开盘看它是否低开, 高开就空仓。
结果 198 次信号里 82 次放弃(41%), 只剩 116 笔。

但实盘根本不是这么做的。真实流程是:
  T 日收盘 -> 算出候选池(约 1600 只)和每只的 9 窗口平均排名 -> 排序备好
  T+1 9:25 集合竞价结束 -> 看这批票的开盘价 -> 筛出低开的(约 60%)
  -> 在低开的里面挑排名最靠前(跌最透)的那只 -> 开盘价买入
这个流程完全不看未来: T+1 开盘价在 9:25 就是公开信息。

245B 已经算出: "每日在低开票里挑最弱" 198 次, 单笔 2.15%, 胜率 65.2%,
合计 426pp —— 比 244 实盘口径的 116 笔 2.45%(合计 284pp) 高 50%。
本轮把它做成净值, 与 244 定稿正面比。
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


def metrics(r: pd.Series) -> tuple[float, float, float]:
    nav = (1 + r / 100).cumprod()
    ann = nav.iloc[-1] ** (244 / len(r)) - 1
    dd = (nav / nav.cummax() - 1).min()
    return round(ann * 100, 2), round(dd * 100, 2), round(ann * 100 / abs(dd * 100), 2)


def main() -> None:
    tr = pd.read_csv(OUT / "w232_trades.csv")
    all_dates = sorted(int(p.stem.replace(".csv", "")) for p in
                       (ROOT / "data" / "panel").glob("*.csv.gz"))
    cal = np.array([d for d in all_dates if d >= tr.signal.min()])
    base = pd.Series(0.0, index=cal)
    for dt, g in tr.groupby("signal"):
        base[int(dt)] = g["ret"].mean()
    dayrank = {int(d): i for i, d in enumerate(cal)}

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
    w["y_net"] = w["y"] - COST

    def prep(g: pd.DataFrame) -> pd.DataFrame:
        den = {c: max(int(g[c].notna().sum()), 1) for c in dhcols}
        rk = pd.DataFrame({c: g[c].rank(ascending=True) / den[c] for c in dhcols})
        g = g.copy()
        g["mr"] = rk.mean(axis=1, skipna=True)
        return g

    w = pd.concat([prep(g) for _, g in w.groupby("trade_date")], ignore_index=True)
    byday = {dt: g for dt, g in w.groupby("trade_date")}

    def run(rule: str, cd: int) -> pd.Series:
        """rule: 'pick_then_skip' = 244 口径(先选再看);  'reallot' = T+1 开盘重选"""
        last: dict[str, int] = {}
        v = {}
        for dt in sorted(byday):
            g = byday[dt]
            if rule == "pick_then_skip":
                cand = g.sort_values(["mr", "amt_yi"], ascending=[True, False])
            else:
                lo = g[g.n1_ok]
                cand = (lo if len(lo) else g).sort_values(["mr", "amt_yi"],
                                                          ascending=[True, False])
            if cd > 0:
                ok = [(c not in last) or (dayrank[dt] - last[c] >= cd) for c in cand["ts_code"]]
                sub = cand[np.array(ok)]
                if len(sub):
                    cand = sub
            if not len(cand):
                continue
            r = cand.iloc[0]
            if pd.isna(r["mr"]):
                continue
            if rule == "pick_then_skip" and (not bool(r["n1_ok"]) or pd.isna(r["y"])):
                v[dt] = 0.0
                continue
            if pd.isna(r["y"]):
                v[dt] = 0.0
                continue
            v[dt] = float(r["y_net"])
            last[r["ts_code"]] = dayrank[dt]
        return pd.Series(v).sort_index()

    b = metrics(base)
    print(f"基准 年化{b[0]:.2f} 回撤{b[1]:.2f} 卡玛{b[2]:.2f}\n")
    print(f"{'方案':<28}{'冷却':>4}{'笔数':>5}{'单笔':>8}{'年化':>9}{'回撤':>9}{'卡玛':>7}")
    rows = []
    for rule, rn in (("pick_then_skip", "244 T日选/T+1高开放弃"),
                     ("reallot", "245 T+1开盘在低开票里重选")):
        for cd in (0, 5, 10, 15, 20):
            v = run(rule, cd)
            trd = v[v != 0]
            for P in (0.70,):
                sim = base.copy()
                sim[v.index] += v.values * P
                a, dd, cm = metrics(sim)
                print(f"{rn:<28}{cd:>4}{len(trd):>5}{trd.mean():>8.3f}{a:>9.2f}{dd:>9.2f}{cm:>7.2f}")
                rows.append({"方案": rn, "冷却": cd, "笔数": len(trd),
                             "单笔": round(trd.mean(), 3), "年化": a, "回撤": dd, "卡玛": cm})

    # 仓位扫(选定的重选方案, 冷却 10)
    print("\n仓位扫描 (T+1 重选, 冷却 10):")
    v = run("reallot", 10)
    for P in (0.30, 0.40, 0.50, 0.60, 0.70, 0.85, 1.00):
        sim = base.copy()
        sim[v.index] += v.values * P
        a, dd, cm = metrics(sim)
        print(f"  {P:>5.0%}  年化{a:>7.2f}  回撤{dd:>7.2f}  卡玛{cm:>6.2f}")

    # 分年
    p = pd.DataFrame({"yr": v.index // 10000, "y": v.values})
    g = p.groupby("yr").y.agg(["size", "sum", "mean"])
    g["win%"] = p.groupby("yr").y.apply(lambda x: (x > 0).mean() * 100).round(0)
    print("\n分年 (冷却10, 单笔口径):")
    print(g.round(2).to_string())
    pd.DataFrame(rows).to_csv(OUT / "w245_reallot.csv", index=False, encoding="utf-8-sig")
    v.to_frame("y").to_csv(OUT / "w245_picks.csv", encoding="utf-8-sig")


if __name__ == "__main__":
    main()
