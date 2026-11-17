"""第244轮 C: 参数选择的 IC 检验 + 多窗口集成（不选参数）

244B 暴露三件事:
  ① 全样本最优 (250,20) 卡玛 5.70, 但 2017+ 固定用它是 4.41 —— 比随机选参数的
     中位数 4.88 还低, 说明"精心挑的 250/20"在近期并不占优;
  ② walk-forward 每年重选 (窗口,冷却) 得 5.17, 但每年选出来的窗口在 90~250 之间乱跳;
  ③ 所以问题不是"哪个窗口最好", 而是"哪来这么多窗口可选"。

本轮两件事:
  A. IC 检验: 每年"历史卡玛排名"能不能预测"当年卡玛排名"(Spearman)。
     rho≈0 就证明选参数没有信息量, WF 的 5.17 只是分布里的运气抽样。
  B. 集成: 不选窗口, 把 9 个窗口的排序结果一起用(排名平均 / top3 投票)。
     若集成能到 5.4 以上, 就说明"消掉参数敏感性"比"找最优参数"更值钱。
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

    tr = pd.read_csv(OUT / "w232_trades.csv")
    all_dates = sorted(int(p.stem.replace(".csv", "")) for p in
                       (ROOT / "data" / "panel").glob("*.csv.gz"))
    cal = np.array([d for d in all_dates if d >= tr.signal.min()])
    base = pd.Series(0.0, index=cal)
    for dt, g in tr.groupby("signal"):
        base[int(dt)] = g["ret"].mean()
    b = metrics(base)
    span = base.index[base.index // 10000 >= 2017]
    a0, d0, c0 = metrics(base.loc[span])
    print(f"基准全样本 年化{b[0]:.2f} 回撤{b[1]:.2f} 卡玛{b[2]:.2f}   "
          f"2017+ 年化{a0:.2f} 回撤{d0:.2f} 卡玛{c0:.2f}")
    dayrank = {int(dt): i for i, dt in enumerate(cal)}

    # ---------- 宽表: 9 个窗口的 dh 并排 ----------
    wide = sig[["ts_code", "trade_date", "y", "amt_yi"]].copy()
    for W in WINS:
        px = pd.read_parquet(POOL / f"W{W}.parquet").rename(columns={"dh": f"dh{W}"})
        wide = wide.merge(px, on=["ts_code", "trade_date"], how="left")
        del px
    dhcols = [f"dh{W}" for W in WINS]
    wide = wide.dropna(subset=dhcols, how="all")
    print(f"宽表 {len(wide):,} 行  {wide.trade_date.nunique()} 天")

    # ---------- 构建 45 个单窗口序列 ----------
    seq: dict[tuple[int, int], pd.Series] = {}
    byday_w = {dt: g for dt, g in wide.groupby("trade_date")}
    for W in WINS:
        col = f"dh{W}"
        for C in COOLS:
            last: dict[str, int] = {}
            v: dict[int, float] = {}
            for dt in sorted(byday_w):
                g = byday_w[dt].dropna(subset=[col]).sort_values([col, "amt_yi"],
                                                                 ascending=[True, False])
                if not len(g):
                    continue
                cand = g
                if C > 0:
                    ok = [(c not in last) or (dayrank[int(dt)] - last[c] >= C)
                          for c in g["ts_code"]]
                    cand = g[np.array(ok)]
                    if not len(cand):
                        cand = g
                r = cand.iloc[0]
                v[int(dt)] = float(r["y"]) - COST
                last[r["ts_code"]] = dayrank[int(dt)]
            seq[(W, C)] = pd.Series(v).sort_index()

    # ---------- A. IC: 历史卡玛排名 -> 当年卡玛排名 ----------
    print("\n【A】参数选择的 IC 检验 (Spearman, 45 个参数组合)")
    ic_rows = []
    for yr in sorted(set(base.index // 10000))[1:]:
        hist, now = {}, {}
        for (W, C), v in seq.items():
            vh = v[(v.index // 10000) < yr]
            vn = v[(v.index // 10000) == yr]
            if len(vh) < 10 or len(vn) < 2:
                continue
            s = base.copy()
            s[vh.index] += vh.values * 0.70
            hist[(W, C)] = metrics(s)[2]
            s2 = base[(base.index // 10000) == yr].copy()
            s2[vn.index] += vn.values * 0.70
            now[(W, C)] = metrics(s2)[2]
        if len(hist) < 5:
            continue
        hh = pd.Series(hist)
        nn = pd.Series(now).reindex(hh.index)
        rho = hh.rank().corr(nn.rank())
        ic_rows.append({"yr": yr, "n": len(hh), "rho": round(float(rho), 3)})
        print(f"  {yr}: 参数数 {len(hh)}  秩相关 {rho:+.3f}")
    ic = pd.DataFrame(ic_rows)
    ic.to_csv(OUT / "w244_ic.csv", index=False, encoding="utf-8-sig") if False else None
    print(f"  平均 rho = {ic.rho.mean():+.3f}   正年份 {(ic.rho > 0).sum()}/{len(ic)}")

    # ---------- B. 集成 ----------
    print("\n【B】多窗口集成 (不选窗口)")

    def run_ens(mode: str, C: int) -> tuple[pd.Series, float]:
        last: dict[str, int] = {}
        v: dict[int, float] = {}
        picks: dict[int, str] = {}
        for dt in sorted(byday_w):
            g = byday_w[dt].copy()
            denom = {c: max(int(g[c].notna().sum()), 1) for c in dhcols}
            rk = pd.DataFrame({c: g[c].rank(ascending=True) / denom[c] for c in dhcols})
            g["mean_rank"] = rk.mean(axis=1, skipna=True)
            if mode == "vote":
                g["votes"] = 0
                for c in dhcols:
                    top = g.nsmallest(3, c, keep="all")["ts_code"]
                    g.loc[g.ts_code.isin(top), "votes"] += 1
                g = g.sort_values(["votes", "mean_rank", "amt_yi"],
                                  ascending=[False, True, False])
            else:
                g = g.sort_values(["mean_rank", "amt_yi"], ascending=[True, False])
            if C > 0:
                ok = [(c not in last) or (dayrank[int(dt)] - last[c] >= C)
                      for c in g["ts_code"]]
                cand = g[np.array(ok)]
                if len(cand):
                    g = cand
            r = g.iloc[0]
            if pd.isna(r["mean_rank"]):
                continue
            v[int(dt)] = float(r["y"]) - COST
            picks[int(dt)] = r["ts_code"]
            last[r["ts_code"]] = dayrank[int(dt)]
        v = pd.Series(v).sort_index()
        sim = base.copy()
        sim[v.index] += v.values * 0.70
        _, _, cm = metrics(sim)
        return v, cm, pd.Series(picks)

    print(f"\n{'方案':<22}{'次数':>6}{'单笔':>8}{'年化':>9}{'回撤':>9}{'卡玛':>7}"
          f"{'票数':>6}{'前5票%':>8}")
    rows = []
    for mode, name in (("rank", "9窗口排名平均"), ("vote", "9窗口top3投票")):
        for C in (0, 10, 20):
            v, cm, pk = run_ens(mode, C)
            sim = base.copy()
            sim[v.index] += v.values * 0.70
            a, dd, cm = metrics(sim)
            s2 = base.loc[span].copy()
            v2 = v[(v.index // 10000) >= 2017]
            s2[v2.index] += v2.values * 0.70
            _, _, c2 = metrics(s2)
            top5 = pk.value_counts().head(5).sum() / len(pk) * 100
            print(f"{name + ' C=' + str(C):<22}{len(v):>6}{v.mean():>8.3f}{a:>9.2f}"
                  f"{dd:>9.2f}{cm:>7.2f}{pk.nunique():>6}{top5:>7.0f}%"
                  f"   2017+ {c2:.2f}")
            rows.append({"方案": name, "C": C, "次数": len(v), "单笔": round(v.mean(), 3),
                         "年化": a, "回撤": dd, "卡玛": cm, "卡玛2017": c2,
                         "票数": pk.nunique(), "前5票%": round(top5, 0)})

    # 对照: 单窗口 (250,20) 与 (250,0)
    for k in ((250, 20), (250, 0), (150, 0), (90, 0)):
        v = seq[k]
        sim = base.copy()
        sim[v.index] += v.values * 0.70
        a, dd, cm = metrics(sim)
        s2 = base.loc[span].copy()
        v2 = v[(v.index // 10000) >= 2017]
        s2[v2.index] += v2.values * 0.70
        _, _, c2 = metrics(s2)
        print(f"{'单窗口 W=%d C=%d' % k:<22}{len(v):>6}{v.mean():>8.3f}{a:>9.2f}"
              f"{dd:>9.2f}{cm:>7.2f}{'':>6}   2017+ 卡玛 {c2:.2f}")
        rows.append({"方案": f"单窗口 W={k[0]} C={k[1]}", "C": k[1], "次数": len(v),
                     "单笔": round(v.mean(), 3), "年化": a, "回撤": dd, "卡玛": cm,
                     "卡玛2017": c2})

    pd.DataFrame(rows).to_csv(OUT / "w244_ens.csv", index=False, encoding="utf-8-sig")
    ic.to_csv(OUT / "w244_ic.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
