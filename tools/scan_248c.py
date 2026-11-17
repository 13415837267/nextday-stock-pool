"""第248轮 C: 资金调度模拟器

三件事一起量化:
  1. busy 口径: 粗口径排 {signal,buy,sell} 三天 vs 精确口径只排「补位买入日落在主策略持仓区间」
  2. 重叠日优先级: 48 天主策略与补位同日都想要, 谁做?
  3. 组合后净值/回撤/卡玛

S0 必须复现 247 的 290 笔 / 5.44, 否则模拟器不可信。
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
COST = 0.30
MAXT = 8


def metrics(r: pd.Series) -> tuple[float, float, float]:
    nav = (1 + r / 100).cumprod()
    ann = nav.iloc[-1] ** (244 / len(r)) - 1
    dd = (nav / nav.cummax() - 1).min()
    return round(ann * 100, 2), round(dd * 100, 2), round(ann * 100 / abs(dd * 100), 2)


def build_cal() -> list[int]:
    cal = sorted(int(f.split("/")[-1].replace(".csv.gz", ""))
                 for f in glob.glob(str(ROOT / "data" / "panel" / "*.csv.gz")))
    return cal


def simulate(cal, tr_all, cands, busy_mode="coarse", priority="main",
             p_brk=0.30, p_deep=0.50, p_main=1.0, half=False):
    """返回 (sim 序列, 主策略保留笔, 补位选中笔)"""
    idx = {d: i for i, d in enumerate(cal)}
    tr = tr_all.copy()
    tr["bi"] = tr.buy.map(idx)
    tr["si"] = tr.sell.map(idx)
    tr = tr.dropna(subset=["bi", "si"])
    tr["bi"] = tr.bi.astype(int)
    tr["si"] = tr.si.astype(int)
    sig2tr = tr.set_index("signal")

    c = cands.copy()
    c["ci"] = c.trade_date.map(idx)
    c = c.dropna(subset=["ci"])
    c["ci"] = c.ci.astype(int)

    occ = np.zeros(len(cal), dtype=bool)
    for b, s in zip(tr.bi, tr.si):
        occ[b:s + 1] = True

    def conflict(day_list, day_idx):
        if busy_mode == "coarse":
            return (day_list in set(tr.signal)) or (day_list in set(tr.buy)) or (day_list in set(tr.sell))
        return bool(occ[min(day_idx + 1, len(cal) - 1)])

    keep = []
    fill = []
    if priority == "main":
        for _, r in c.sort_values("trade_date").iterrows():
            if conflict(int(r.trade_date), int(r.ci)):
                continue
            fill.append(r)
        keep = list(range(len(tr)))
    elif priority == "fill":
        # 重叠日补位优先: 主策略那笔不要
        drop_sig = set(c.trade_date) & set(tr.signal)
        keep = [i for i, s in enumerate(tr.signal) if s not in drop_sig]
        sub = tr.iloc[keep]
        occ2 = np.zeros(len(cal), dtype=bool)
        for b, s in zip(sub.bi, sub.si):
            occ2[b:s + 1] = True
        for _, r in c.sort_values("trade_date").iterrows():
            tgt = min(int(r.ci) + 1, len(cal) - 1)
            if busy_mode == "coarse":
                if (r.trade_date in set(sub.signal)) or (r.trade_date in set(sub.buy)) \
                        or (r.trade_date in set(sub.sell)):
                    continue
            elif occ2[tgt] and r.trade_date not in drop_sig:
                continue
            fill.append(r)
    elif priority == "fill_deep":
        # 只让「深低开」重叠日: 破位日仍归主策略(破位 7 天两边几乎打平, 不必换)
        deep_days = set(c[~c.brk].trade_date)
        drop_sig = deep_days & set(tr.signal)
        keep = [i for i, sg in enumerate(tr.signal) if sg not in drop_sig]
        sub = tr.iloc[keep]
        occ2 = np.zeros(len(cal), dtype=bool)
        for b, s_ in zip(sub.bi, sub.si):
            occ2[b:s_ + 1] = True
        for _, r in c.sort_values("trade_date").iterrows():
            if r.brk and r.trade_date in set(sub.signal):
                continue
            tgt = min(int(r.ci) + 1, len(cal) - 1)
            if occ2[tgt] and r.trade_date not in drop_sig:
                continue
            fill.append(r)
    elif priority == "half":
        drop_sig = set(c.trade_date) & set(tr.signal)
        keep = list(range(len(tr)))
        sub = tr.iloc[keep]
        occ2 = np.zeros(len(cal), dtype=bool)
        for b, s in zip(sub.bi, sub.si):
            occ2[b:s + 1] = True
        for _, r in c.sort_values("trade_date").iterrows():
            tgt = min(int(r.ci) + 1, len(cal) - 1)
            if r.trade_date in drop_sig:
                fill.append(r)          # 重叠日两边都做
                continue
            if busy_mode == "coarse":
                if r.trade_date in set(sub.buy) or r.trade_date in set(sub.sell):
                    continue
            elif occ2[tgt]:
                continue
            fill.append(r)

    sub = tr.iloc[keep]
    base = pd.Series(0.0, index=cal)
    for dt, g in sub.groupby("signal"):
        base[int(dt)] = g.ret.mean()
    if half:
        for dt in set(c.trade_date) & set(tr.signal):
            base[int(dt)] = base[int(dt)] / 2.0

    n: dict[str, int] = {}
    rows = []
    for r in fill:
        chain = str(r.chain).split("|") if hasattr(r, "chain") and isinstance(r.chain, str) else []
        pick = None
        for it in chain:
            cd, yy = it.rsplit(":", 1)
            if n.get(cd, 0) < MAXT:
                pick = (cd, float(yy))
                break
        if pick is None:
            continue
        n[pick[0]] = n.get(pick[0], 0) + 1
        p = p_brk if r.brk else p_deep
        if half and r.trade_date in set(tr.signal):
            p = p / 2.0
        rows.append({"trade_date": int(r.trade_date), "ts_code": pick[0], "y": pick[1],
                     "brk": bool(r.brk), "path": r.path, "p": p})
    f = pd.DataFrame(rows)
    sim = base.copy()
    if len(f):
        v = pd.Series((f.y.values - COST) * f.p.values, index=f.trade_date.values)
        v = v.groupby(level=0).sum().sort_index()
        sim[v.index] = sim[v.index] + v.values
    return sim, sub, f


def main() -> None:
    print("=" * 78)
    print("第248轮 C: 资金调度")
    print("=" * 78)
    cal = build_cal()
    tr = pd.read_csv(OUT / "w232_trades.csv")
    tr = tr[tr.signal >= cal[0]].reset_index(drop=True)
    print(f"主策略 {len(tr)} 笔  signal 重复日 {int(tr.signal.duplicated().sum())}")
    cal = [d for d in cal if d >= tr.signal.min()]
    print(f"日历 {len(cal)} 天  {cal[0]} ~ {cal[-1]}")
    c = pd.read_csv(OUT / "w248_cands.csv")

    base_sim = pd.Series(0.0, index=cal)
    for dt, g in tr.groupby("signal"):
        base_sim[int(dt)] = g.ret.mean()

    print(f"\n主策略单独: {metrics(base_sim)[0]:.2f}% / {metrics(base_sim)[1]:.2f}% / "
          f"{metrics(base_sim)[2]:.2f}  (报 86.42 / -24.24 / 3.57)")

    plans = [
        ("S0 247定稿 (粗busy,主优先)", dict(busy_mode="coarse", priority="main")),
        ("S1 精确busy, 主优先", dict(busy_mode="exact", priority="main")),
        ("S2 精确busy, 全让位", dict(busy_mode="exact", priority="fill")),
        ("S2b 精确busy, 只深低开让位", dict(busy_mode="exact", priority="fill_deep")),
        ("S3 精确busy, 重叠各半", dict(busy_mode="exact", priority="half", half=True)),
        ("S4 粗busy, 重叠各半", dict(busy_mode="coarse", priority="half", half=True)),
        ("S5 精确busy, 主优先+深低开60%", dict(busy_mode="exact", priority="main", p_deep=0.60)),
    ]
    res = {}
    print(f"\n{'方案':<28}{'笔数':>6}{'破位':>5}{'深低':>5}{'年化':>9}{'回撤':>9}{'卡玛':>7}{'正年':>6}")
    for nm, kw in plans:
        sim, sub, f = simulate(cal, tr, c, **kw)
        a, dd, cm = metrics(sim)
        yr = sim.groupby(sim.index // 10000).apply(lambda x: (1 + x / 100).prod() - 1)
        print(f"{nm:<28}{len(f):>6}{int(f.brk.sum()):>5}{int((~f.brk).sum()):>5}"
              f"{a:>9.2f}{dd:>9.2f}{cm:>7.2f}{int((yr > 0).sum()):>4}/{len(yr)}")
        res[nm] = (sim, sub, f, (a, dd, cm))
        if nm.startswith("S0"):
            print(f"  S0 复现校验: 报 290 笔 / 128.36 / -23.59 / 5.44")

    # 重叠日明细
    ov_days = set(c.trade_date) & set(tr.signal)
    print(f"\n【重叠日 {len(ov_days)} 天】")
    m = tr.set_index("signal").loc[sorted(ov_days)]
    cc = c[c.trade_date.isin(ov_days)].set_index("trade_date")
    j = pd.DataFrame({"主策收益": m.ret - COST, "补位收益": cc.y - COST,
                      "主策票": m.code, "补位票": cc.ts_code,
                      "同票": m.code.values == cc.ts_code.values,
                      "路径": cc.path, "年份": cc.yr})
    print(f"  同票 {int(j.同票.sum())} 天 / 不同票 {int((~j.同票).sum())} 天")
    for pa in j.路径.unique():
        s = j[j.路径 == pa]
        print(f"  {pa}: {len(s)} 天  主策 {s.主策收益.mean():+.2f}%  补位 {s.补位收益.mean():+.2f}%  "
              f"补位更强 {(s.补位收益 > s.主策收益).mean()*100:.0f}%")
    dif = j.补位收益 - j.主策收益
    print(f"  差的均值 {dif.mean():+.2f}pp  t={dif.mean()/(dif.std(ddof=1)/np.sqrt(len(dif))):.2f}")
    j.rename_axis("trade_date").reset_index().to_csv(OUT / "w248_overlap_detail.csv",
                                                     index=False, encoding="utf-8-sig")

    s2 = res["S2 精确busy, 全让位"]
    s2[2].to_csv(OUT / "w248_picks_S2.csv", index=False, encoding="utf-8-sig")
    s2[0].to_frame("r").to_csv(OUT / "w248_nav_S2.csv", encoding="utf-8-sig")
    res["S0 247定稿 (粗busy,主优先)"][2].to_csv(OUT / "w248_picks_S0.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
