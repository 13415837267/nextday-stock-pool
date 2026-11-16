"""第243轮 D: 用正确日历重算 —— 补位阈值放宽到 idx_dist<=-2.0 的真实净值

243C 纠正了 243A 的 T+0 错误: 现行"T+1 开盘买 -> T+2 收盘卖"已经是合规口径里最好的。
所以本轮唯一还能动的是**触发频率**: 243B 的地形显示 idx_dist 从 -3.51 放宽到 -2.0,
次数 192 -> 372, 单笔 1.768 -> 1.326(扣成本前), 但总收益更高。

本脚本用 2500 日完整日历重算, 并做三段检验:
  段1 2016-2019  段2 2020-2023  段3 2024-2026
以及"每年触发次数上限"的敏感性(防止放宽后集中砸在某几年)。

★ 净值口径: base 是主策略每日收益(空仓日=0), 补位日整仓插入, 不复利叠加冲突。
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "output" / "feat234"
OUT = ROOT / "output"
COST = 0.10
COLS = ["ts_code", "trade_date", "y", "n1_ok", "pct", "idx_dist", "dist_high60",
        "amt_yi", "is_paused", "is_st", "listed"]


def load():
    parts = []
    for p in sorted(FEAT.glob("f*.parquet")):
        d = pd.read_parquet(p, columns=COLS)
        d = d[(d.is_paused == 0) & (d.is_st == 0) & (d.listed >= 250)]
        d = d[d.pct.between(-11.5, 11.5) & d.y.between(-40, 60)]
        d = d[d.ts_code.str[:2].isin(["60", "00"])]
        d = d[d.n1_ok == True]
        if len(d):
            parts.append(d)
    return pd.concat(parts, ignore_index=True)


def M(r: pd.Series):
    nav = (1 + r / 100).cumprod()
    ann = nav.iloc[-1] ** (244 / len(r)) - 1
    dd = (nav / nav.cummax() - 1).min()
    return round(ann * 100, 2), round(dd * 100, 2), round(ann * 100 / abs(dd * 100), 2)


def main():
    d = load()
    cal = np.sort(d.trade_date.unique())
    tr = pd.read_csv(OUT / "w232_trades.csv")
    cal = cal[cal >= tr.signal.min()]
    tr = tr[tr.signal.isin(cal)]
    busy = set()
    for _, r in tr.iterrows():
        busy.add(r.buy)
        busy.add(r.sell)
    sig = set(tr.signal)
    free = set(x for x in cal if x not in busy and x not in sig)
    base = pd.Series(0.0, index=cal)
    for dt, g in tr.groupby("signal"):
        base[dt] = g["ret"].mean()                # 成交明细 ret 已是百分数(1.25 = 1.25%)
    b = M(base)
    print(f"日历 {len(cal)} 日 ({cal[0]}~{cal[-1]})  主策略 {len(tr)} 笔  "
          f"年化{b[0]}% 回撤{b[1]}% 卡玛{b[2]}  自由日 {len(free)}")
    nxt = {cal[i]: cal[i + 1] for i in range(len(cal) - 1)}
    nxt[cal[-1]] = cal[-1]

    def pick(lo: float, hi_high: float = -24.46):
        s = d[(d.idx_dist <= lo) & (d.dist_high60 <= hi_high)]
        days = []
        for dt in sorted(s.trade_date.unique()):
            if dt not in free:
                continue
            bd = nxt.get(dt)
            sd = nxt.get(bd, bd)
            if bd in busy or sd in busy:
                continue
            days.append(dt)
        rec = []
        for dt in days:
            g = d[(d.trade_date == dt) & d.dist_high60.notna()]
            if not len(g):
                continue
            g = g.sort_values(["dist_high60", "amt_yi"], ascending=[True, False])
            rec.append({"signal": dt, "code": g.iloc[0].ts_code, "y": g.iloc[0].y})
        return pd.DataFrame(rec)

    print("\n" + "=" * 96)
    print("① 阈值放宽的真实净值 (满仓单吊, T+2 收盘卖)")
    rows = []
    grids = [-2.0, -2.5, -3.0, -3.51]
    for lo in grids:
        Q = pick(lo)
        v = pd.Series(Q.y.values - COST, index=Q.signal.values)
        sim = base.copy()
        sim[v.index] += v.values
        a, dd, cm = M(sim)
        rows.append({"idx_dist<=": lo, "次数": len(v), "覆盖自由日%": round(len(v) / len(free) * 100),
                     "单笔": round(v.mean(), 3), "年化": a, "回撤": dd, "卡玛": cm,
                     "Δ卡玛": round(cm - b[2], 2)})
    t = pd.DataFrame(rows)
    print(t.to_string(index=False))
    t.to_csv(OUT / "w243d_net.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 96)
    print("② 三段检验 (-3.51 现行 vs -2.0 放宽)")
    seg = [(2016, 2019), (2020, 2023), (2024, 2026)]
    rows = []
    for lo in (-3.51, -2.0):
        Q = pick(lo)
        v = pd.Series(Q.y.values - COST, index=Q.signal.values)
        sim = base.copy()
        sim[v.index] += v.values
        for y0, y1 in seg:
            m = (sim.index // 10000 >= y0) & (sim.index // 10000 <= y1)
            sub = sim[m]
            a, dd, cm = M(sub)
            vb = base[m]
            ab, ddb, cmb = M(vb)
            n = ((v.index // 10000 >= y0) & (v.index // 10000 <= y1)).sum()
            rows.append({"阈值": lo, "段": f"{y0}-{y1}", "补位笔": int(n),
                         "年化": a, "回撤": dd, "卡玛": cm,
                         "基准卡玛": cmb, "Δ": round(cm - cmb, 2)})
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n" + "=" * 96)
    print("③ -2.0 的分年明细")
    Q = pick(-2.0)
    v = pd.Series(Q.y.values - COST, index=Q.signal.values)
    p = pd.DataFrame({"yr": v.index // 10000, "y": v.values})
    g = p.groupby("yr").y.agg(["size", "sum", "mean"])
    g["win%"] = p.groupby("yr").y.apply(lambda x: (x > 0).mean() * 100).round(0)
    print(g.round(2).to_string())
    run = mx = 0
    for x in v.values:
        run = run + 1 if x < 0 else 0
        mx = max(mx, run)
    vv = v.sort_values(ascending=False)
    tot = vv.sum()
    print(f"\n合计 {len(v)} 次 单笔{v.mean():+.3f}% 胜率{(v > 0).mean() * 100:.1f}% "
          f"最长连亏{mx} 前5笔占总收益{vv[:5].sum() / tot * 100:.0f}%")

    print("\n" + "=" * 96)
    print("④ 与主策略的同日冲突检查")
    Q = pick(-2.0)
    qs = set(Q.signal)
    print(f"  补位日 {len(qs)}  与主策略信号日重叠 {len(qs & sig)}  与持仓日重叠 {len(qs & busy)}")
    print(f"  补位日落在自由日 {len(qs & free)}/{len(qs)}")


if __name__ == "__main__":
    main()
