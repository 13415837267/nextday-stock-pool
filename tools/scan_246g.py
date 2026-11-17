"""第246轮 G: A(245口径) vs B(并列深低开) 的严格对照 —— 只看无冷却版

246f 修正资金占用日后:
  A 245口径  217 笔 单笔 1.58%  118.4/-23.59/5.02
  B 并列     236 笔 单笔 2.20%  136.8/-23.59/5.80   (70%仓 158.4/-25.86/6.13)
  C 仅深低开  28 笔 单笔10.22%  不看指数
冷却(无论 10 还是 20)在这套里全部有害, 与 243 轮的结论一致 —— 冷却换票会把好票换掉。

但 B 的优势可能全来自 2024 年那一笔 +34%。本轮做四项压力测试:
  ① 三段分年, B 是否每段都赢 A
  ② 剔除最大 5 笔后还剩多少
  ③ 集中度(票数/最长连买/前5占比) —— 这是 243 轮选冷却的唯一理由
  ④ 深低开阈值的敏感性 -4 ~ -7
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
COST = 0.10
SLIP = 0.20


def metrics(r: pd.Series) -> tuple[float, float, float]:
    nav = (1 + r / 100).cumprod()
    ann = nav.iloc[-1] ** (244 / len(r)) - 1
    dd = (nav / nav.cummax() - 1).min()
    return round(ann * 100, 2), round(dd * 100, 2), round(ann * 100 / abs(dd * 100), 2)


def main() -> None:
    tr = pd.read_csv(OUT / "w232_trades.csv")
    all_dates = sorted(int(p.stem.replace(".csv", "")) for p in
                       (ROOT / "data" / "panel").glob("*.csv.gz"))
    cal = np.array([x for x in all_dates if x >= int(tr.signal.min())])
    dayrank = {int(d): i for i, d in enumerate(cal)}
    base = pd.Series(0.0, index=cal)
    busy = set(int(x) for x in tr.signal)
    for _, r in tr.iterrows():
        busy.add(int(r["buy"]))
        busy.add(int(r["sell"]))
    for dt, g in tr.groupby("signal"):
        base[int(dt)] = g["ret"].mean()

    d = pd.read_parquet(OUT / "w246b_all.parquet",
                        columns=["ts_code", "trade_date", "dh250", "dh60", "listed",
                                 "n1_gap", "y", "amt_yi", "circ_mv"])
    d = d.dropna(subset=["dh250", "dh60", "n1_gap", "y"])
    d = d[d.listed >= 250]
    d["trade_date"] = d["trade_date"].astype("int64")
    d = d[(d.trade_date >= cal[0]) & (~d.trade_date.isin(busy))]
    mk = pd.read_csv(OUT / "feat234" / "market.csv",
                     usecols=["trade_date", "idx_ma20_dist", "breadth", "limit_dn",
                              "med_pct", "eq_ret"])
    d = d.merge(mk, on="trade_date", how="inner").dropna(subset=["idx_ma20_dist"])

    lo = d[d.n1_gap <= 0.05].sort_values(
        ["trade_date", "dh250", "amt_yi"], ascending=[True, True, False])
    lo = lo[["ts_code", "trade_date", "dh250", "dh60", "n1_gap", "y", "amt_yi",
             "circ_mv", "idx_ma20_dist"]].reset_index(drop=True)
    gb = {int(dt): g for dt, g in lo.groupby("trade_date")}

    def sel(rule: str, thr: float = -5.0, cd: int = 0) -> pd.DataFrame:
        last: dict[str, int] = {}
        out = []
        for dt in sorted(gb):
            g = gb[dt]
            top = g.iloc[0]
            broke = (top.idx_ma20_dist <= -3.5) and (top.dh60 <= -24.46)
            trig = broke if rule == "A" else (broke or (top.n1_gap <= thr))
            if not trig:
                continue
            i = dayrank[dt]
            pick = top
            if cd:
                for j in range(len(g)):
                    rr = g.iloc[j]
                    if rr.ts_code not in last or i - last[rr.ts_code] >= cd:
                        pick = rr
                        break
            last[pick.ts_code] = i
            out.append({"trade_date": dt, "ts_code": pick.ts_code, "y": pick.y,
                        "n1_gap": pick.n1_gap, "dh60": pick.dh60,
                        "idx": top.idx_ma20_dist, "brk": broke,
                        "yr": dt // 10000})
        return pd.DataFrame(out)

    def net(c: pd.DataFrame, P: float) -> tuple[float, float, float, pd.Series]:
        s = pd.Series((c["y"].values - COST - SLIP) * P, index=c.trade_date.values)
        s = s.sort_index()
        sim = base.copy()
        sim[s.index] = sim[s.index] + s.values
        return (*metrics(sim), s)

    A = sel("A")
    B = sel("B", -5.0)

    print(f"仅主策略 {metrics(base)}")
    print(f"\n{'':<8}{'笔数':>5}{'单笔':>8}{'年化':>9}{'回撤':>9}{'卡玛':>7}")
    for tag, c in (("A", A), ("B", B)):
        for P in (0.5, 0.7):
            a, dd, cm, s = net(c, P)
            print(f"{tag+' '+str(int(P*100))+'%':<8}{len(c):>5}{(s.mean()/P):>8.2f}"
                  f"{a:>9.2f}{dd:>9.2f}{cm:>7.2f}")

    print("\n【1】三段 分年对比 (50%仓)")
    for lo_y, hi_y, tag in ((2016, 2019, "2016-2019"), (2020, 2022, "2020-2022"),
                            (2023, 2026, "2023-2026")):
        row = [tag]
        for nm, c in (("A", A), ("B", B)):
            cc = c[(c.yr >= lo_y) & (c.yr <= hi_y)]
            a, dd, cm, s = net(cc, 0.5)
            row.append(f"{nm}: {len(cc):>3}笔 {s.mean()/0.5:>5.2f}% 卡玛{cm:>5.2f}")
        print("  " + "   ".join(row))

    print("\n【2】逐年（单笔%, 括号内笔数）")
    pv = []
    for yr in sorted(set(A.yr) | set(B.yr)):
        ra = A[A.yr == yr]
        rb = B[B.yr == yr]
        pv.append({"年": yr, "A笔": len(ra), "A单笔": round(ra.y.mean() - 0.3, 2) if len(ra) else None,
                   "B笔": len(rb), "B单笔": round(rb.y.mean() - 0.3, 2) if len(rb) else None})
    print(pd.DataFrame(pv).to_string(index=False))

    print("\n【3】剔除最大 5 笔后的卡玛 (50%仓)")
    for tag, c in (("A", A), ("B", B)):
        y = c.y - COST - SLIP
        drop = set(y.nlargest(5).index)
        c2 = c.loc[[i for i in c.index if i not in drop]]
        a, dd, cm, s = net(c2, 0.5)
        a0, d0, c0, s0 = net(c, 0.5)
        print(f"  {tag}: 全样本卡玛 {c0:>5.2f} -> 剔5笔后 {cm:>5.2f}"
              f"  (单笔 {s.mean()/0.5:>5.2f}%)  最大5笔占{y.nlargest(5).sum()/y.sum()*100:.0f}%")

    print("\n【4】集中度")
    for tag, c in (("A", A), ("B", B)):
        cnt = c.ts_code.value_counts()
        print(f"  {tag}: {len(c)} 笔 / {c.ts_code.nunique()} 只票  最长连买 {cnt.max()} 次  "
              f"前5票占 {cnt.head(5).sum()/len(c)*100:.0f}%  最差票 "
              f"{(c.groupby('ts_code').y.mean() - 0.3).min():.1f}%")

    print("\n【5】深低开阈值敏感性 (B, 50%仓)")
    for thr in (-4.0, -4.5, -5.0, -5.5, -6.0, -7.0, -99.0):
        c = sel("B", thr)
        a, dd, cm, s = net(c, 0.5)
        n_extra = len(c) - len(A)
        nm = "B(不加深低开= A)" if thr < -90 else f"低开<={thr}"
        print(f"  {nm:<18} n={len(c):>3} (比A多{n_extra:>2})  单笔{s.mean()/0.5:>6.2f}%  "
              f"年化{a:>7.2f}  回撤{dd:>7.2f}  卡玛{cm:>6.2f}")

    print("\n【6】B 里「破位+深低开」与「仅深低开」分别贡献")
    for tag, m in (("破位+深低开(已在A内)", B.brk), ("仅深低开(新增)", ~B.brk)):
        c = B[m]
        y = c.y - COST - SLIP
        print(f"  {tag:<22} n={len(c):>3}  单笔{y.mean():>6.2f}%  胜率"
              f"{(y>0).mean()*100:>4.0f}%  合计{y.sum():>6.0f}pp")
    A.to_csv(OUT / "w246g_A.csv", index=False, encoding="utf-8-sig")
    B.to_csv(OUT / "w246g_B.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
