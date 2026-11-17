"""第244轮 B: 参数联合面 + 滚动 walk-forward 参数选择

问题: 243 轮定稿的 (窗口 250, 仓位 70%, 冷却 20 日) 全部是在 11 年全样本上挑出来的
最优值。全样本最优 ≠ 未来可用。本轮做三件事:

  ① 联合面: 9 窗口 × 5 冷却 × 5 仓位 = 225 格, 看定稿值落在平台还是孤岛
  ② walk-forward: 每年初只用"该年之前"的数据选参数, 该年应用, 拼接真样本外净值
     —— 这才是实盘能拿到的路径, 因为它不偷看未来
  ③ 随机参数对照: 每年随机选参数 2000 次, 得分布。WF 若打不过随机中位数,
     说明"用历史选参数"没有信息量, 定稿值只是拟合噪声。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
POOL = OUT / "w244_pool"

WINS = (60, 90, 120, 150, 180, 200, 250, 300, 360)
COOLS = (0, 10, 20, 30, 40)
POSS = (0.30, 0.50, 0.70, 0.85, 1.00)
COST = 0.10


def metrics(r: pd.Series) -> tuple[float, float, float]:
    if not len(r):
        return 0.0, 0.0, 0.0
    nav = (1 + r / 100).cumprod()
    ann = nav.iloc[-1] ** (244 / len(r)) - 1
    dd = (nav / nav.cummax() - 1).min()
    return round(ann * 100, 2), round(dd * 100, 2), round(ann * 100 / abs(dd * 100), 2)


def main() -> None:
    sig = pd.read_parquet(POOL / "signals.parquet")
    sig["trade_date"] = sig["trade_date"].astype("int64")

    # ---- 基准: 主策略日收益(必须用完整交易日历, 否则年化会被 198 天放大) ----
    tr = pd.read_csv(OUT / "w232_trades.csv")
    all_dates = sorted(int(p.stem.replace(".csv", "")) for p in
                       (ROOT / "data" / "panel").glob("*.csv.gz"))
    cal = np.array([d for d in all_dates if d >= tr.signal.min()])
    busy = set()
    for _, r in tr.iterrows():
        busy.add(r.buy)
        busy.add(r.sell)
    sigd = set(int(x) for x in sig.trade_date.unique())
    base = pd.Series(0.0, index=cal)
    for dt, g in tr.groupby("signal"):
        base[int(dt)] = g["ret"].mean()
    print(f"日历 {cal[0]}~{cal[-1]}  {len(cal)} 天  补位可用 {len(sigd)} 天")
    b = metrics(base)
    print(f"基准(仅主策略)  年化{b[0]:>7.2f}  回撤{b[1]:>7.2f}  卡玛{b[2]:.2f}")

    # ---- 45 个 (W, C) 选股序列 ----
    seq: dict[tuple[int, int], pd.Series] = {}
    dayrank = {dt: i for i, dt in enumerate(cal)}
    for W in WINS:
        px = pd.read_parquet(POOL / f"W{W}.parquet")
        m = sig.merge(px, on=["ts_code", "trade_date"], how="inner")
        m = m.dropna(subset=["dh"])
        m["k1"] = m["dh"]
        m["k2"] = -m["amt_yi"]
        m = m.sort_values(["trade_date", "k1", "k2"])
        byday = {int(dt): g for dt, g in m.groupby("trade_date")}
        order = sorted(byday)
        for C in COOLS:
            last: dict[str, int] = {}
            v: dict[int, float] = {}
            for dt in order:
                g = byday[dt]
                cand = g
                if C > 0:
                    ok = [(c not in last) or (dayrank[dt] - last[c] >= C) for c in g["ts_code"]]
                    cand = g[np.array(ok)]
                    if not len(cand):
                        cand = g                    # 全被冷却则退回, 宁可重复不空仓
                r = cand.iloc[0]
                v[dt] = float(r["y"]) - COST
                last[r["ts_code"]] = dayrank[dt]
            seq[(W, C)] = pd.Series(v).sort_index()
        del px, m, byday
        print(f"  W={W:>3} 完成")

    # ---- ① 联合面 ----
    rows = []
    for (W, C), v in seq.items():
        for P in POSS:
            sim = base.copy()
            sim[v.index] += v.values * P
            a, dd, cm = metrics(sim)
            rows.append({"W": W, "C": C, "P": P, "n": len(v), "单笔": round(v.mean(), 3),
                         "年化": a, "回撤": dd, "卡玛": cm})
    grid = pd.DataFrame(rows)
    grid.to_csv(OUT / "w244_grid.csv", index=False, encoding="utf-8-sig")
    best = grid.sort_values("卡玛", ascending=False).head(8)
    print("\n全样本最优 8 格:")
    print(best.to_string(index=False))

    # ---- ② walk-forward ----
    grid["key"] = list(zip(grid.W, grid.C, grid.P))
    years = sorted(set(base.index // 10000))
    print(f"\n年份 {years}")
    span = base.index[base.index // 10000 >= 2017]
    a0, dd0, c0 = metrics(base.loc[span])
    print(f"对照: 2017+ 同区间仅主策略  年化{a0:>7.2f}  回撤{dd0:>7.2f}  卡玛{c0:.2f}")

    def wf_run(label: str, fixed_p: float | None = None) -> tuple[pd.Series, float]:
        parts, picks = [], []
        keyset = [k for k in grid["key"] if fixed_p is None or abs(k[2] - fixed_p) < 1e-9]
        for yr in years[1:]:                     # 第一年作初始训练集
            cand = []
            for (W, C, P) in keyset:
                v = seq[(W, C)]
                v = v[(v.index // 10000) < yr]
                if len(v) < 10:
                    continue
                sim = base.copy()
                sim[v.index] += v.values * P
                _, _, c_ = metrics(sim)
                cand.append((c_, W, C, P))
            if not cand:
                continue
            c_, W, C, P = max(cand)
            v = seq[(W, C)]
            vr = v[(v.index // 10000) == yr] * P
            s = base[(base.index // 10000) == yr].copy()
            s[vr.index] += vr.values
            parts.append(s)
            picks.append({"yr": yr, "W": W, "C": C, "P": P, "hist_kama": round(c_, 2),
                          "n": len(vr), "bar": round(float(vr.mean() / P), 3) if len(vr) else 0})
        wf = pd.concat(parts).sort_index()
        a, dd, cm = metrics(wf)
        print(f"\n★ {label}  年化{a:>7.2f}  回撤{dd:>7.2f}  卡玛{cm:.2f}  "
              f"Δ仅主策略{c0:+.2f}")
        for p in picks:
            print(f"   {p['yr']}: W={p['W']:>3} C={p['C']:>2} P={p['P']:.0%} "
                  f"(历史卡玛{p['hist_kama']:>7.2f})  {p['n']:>3} 次  单笔{p['bar']:>6.2f}%")
        pd.DataFrame(picks).to_csv(OUT / f"w244_wf_{label}.csv", index=False,
                                   encoding="utf-8-sig")
        return wf, cm

    wf_70, cm_70 = wf_run("P70_只选窗口冷却", 0.70)
    wf_unc, cm_unc = wf_run("全自由")
    wf_70.to_frame("r").to_csv(OUT / "w244_wf_nav.csv", encoding="utf-8-sig")

    print("\n同区间(2017+)固定参数对照:")
    for k in [(250, 20, 0.70), (250, 0, 1.00), (90, 0, 1.00), (150, 20, 1.00)]:
        W, C, P = k
        sim = base.loc[span].copy()
        v = seq[(W, C)]
        v = v[(v.index // 10000) >= 2017] * P
        sim[v.index] += v.values
        aa, ddd, ccc = metrics(sim)
        print(f"  固定 W={W:>3} C={C:>2} P={P:.0%}  年化{aa:>7.2f}  回撤{ddd:>7.2f}  卡玛{ccc:.2f}")

    # ---- ③ 随机参数对照 (必须锁死 P=70%, 否则"随机"也能抓到满仓的高卡玛) ----
    rng = np.random.default_rng(20261117)
    keys70 = [k for k in grid["key"] if abs(k[2] - 0.70) < 1e-9]
    rnd = []
    for trial in range(3000):
        parts_ = []
        for yr in years[1:]:
            if trial == 0:
                W, C = 250, 20                        # 第 0 次 = 定稿参数固定不变
            else:
                W, C, _P = keys70[rng.integers(len(keys70))]
                W, C = int(W), int(C)
            s = base[(base.index // 10000) == yr].copy()
            v = seq[(W, C)]
            v = v[(v.index // 10000) == yr]
            if len(v):
                s[v.index] += v.values * 0.70
            parts_.append(s)
        rnd.append(metrics(pd.concat(parts_))[2])
    rnd = np.array(rnd)
    print(f"\n随机 (W,C) 参数、P 锁 70%、3000 次卡玛: 中位 {np.median(rnd):.2f}  "
          f"5% {np.percentile(rnd, 5):.2f}  95% {np.percentile(rnd, 95):.2f}")
    print(f"  定稿 (250,20) 固定 = {rnd[0]:.2f}    WF 动态选参 = {cm_70:.2f}  "
          f"(分位 {(rnd < cm_70).mean() * 100:.0f}%)")


if __name__ == "__main__":
    main()
