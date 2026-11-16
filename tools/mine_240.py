"""第二百四十轮: 只在"能用的日子"上挖条件 —— 把目标函数对准使用场景

前两轮的错误不在挖法, 在目标函数:
  第 237/239 轮在**全市场所有样本**上挖"什么样的票第二天涨", 却拿它去**基准空仓日**补位。
  这两件事的样本分布完全不同 —— 基准空仓的日子系统性更差(没票/风控), 在全市场上最优的
  条件, 到了空仓日往往失效(B2 就是: 全样本 1.25%, 空仓日补位 -0.16%)。

本轮把训练集直接换成使用场景: **只在基准空仓、且资金没被占用的日子上挖**。
  训练 2016-01~2020-12 的空仓日
  测试 2021-01~2026-11 的空仓日(完全样本外)
筛选同时要求: 绝对收益 > 0 (因为空仓日原本是 0) 且 扣 beta 后仍 > 0 (不能只靠大盘反弹)。
并限制覆盖率 <= 35% 的空仓日 —— 补位是"捡漏", 不是"另开一套满仓策略"。
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from mine_237 import BASE, NA, NB, load          # noqa: E402

OUT = ROOT / "output"
TRADES = OUT / "w232_trades.csv"
TRAIN_END, TEST_START = 2020, 2021
DERIV = ["mv_x_turn", "amt_x_turn", "b60_m_b20", "r5_m_r20", "h20_m_h60",
         "dd_m_pct", "pos_x_tail", "vol_x_pct", "dn_x_amp", "amp_x_turn",
         "gap_x_vr", "ln_mv"]


def eval_cond(d: pd.DataFrame, edges: dict, f1, b1, f2, b2) -> pd.DataFrame:
    m = np.ones(len(d), bool)
    for f, b in ((f1, b1), (f2, b2)):
        q = edges[f]
        v = d[f].to_numpy("float32")
        if b == 0:
            m &= v < q[0]
        elif b >= len(q):
            m &= v >= q[-1]
        else:
            m &= (v >= q[b - 1]) & (v < q[b])
    return d[m]


def stat(s: pd.DataFrame, ndays_all: int) -> dict:
    if not len(s):
        return {"n": 0, "days": 0, "cover": 0.0}
    yr = (s["trade_date"] // 10000).astype(int)
    g = s.groupby(yr)["y"].agg(["size", "mean"])
    g = g[g["size"] >= 40]
    return {"n": len(s), "days": int(s["trade_date"].nunique()),
            "cover": round(s["trade_date"].nunique() / max(ndays_all, 1), 2),
            "y": round(float(s["y"].mean()), 3),
            "y_ex": round(float(s["y_ex"].mean()), 3),
            "win": round(float((s["y"] > 0).mean() * 100), 1),
            "pos_yr": f"{int((g['mean'] > 0).sum())}/{len(g)}",
            "cons": round(float((g["mean"] > 0).mean()), 2) if len(g) else np.nan}


def main() -> None:
    d = load()
    d["y_ex"] = d["y"] - d.groupby("trade_date")["y"].transform("mean")
    cal = np.sort(d["trade_date"].unique())
    cal = cal[cal >= 20160101]

    # 基准占用 / 空仓
    tr = pd.read_csv(TRADES)
    tr = tr[tr["signal"].isin(cal)]
    busy, sig = set(), set(tr["signal"])
    for _, r in tr.iterrows():
        busy.add(r["buy"]); busy.add(r["sell"])
    free = [x for x in cal if x not in busy and x not in sig]
    print(f"窗口 {cal[0]}~{cal[-1]}  {len(cal)} 日  基准成交 {len(tr)} 笔  "
          f"可补位空仓日 {len(free)} ({len(free)/len(cal)*100:.0f}%)")

    d = d[d["trade_date"].isin(free)].copy()
    yr = (d["trade_date"] // 10000).astype(int)
    trn = d[yr <= TRAIN_END].copy()
    tst = d[yr >= TEST_START].copy()
    ntr = trn["trade_date"].nunique()
    nte = tst["trade_date"].nunique()
    print(f"训练(空仓日) {len(trn):,} 样本 / {ntr} 日  均值 {trn['y'].mean():.3f}%")
    print(f"测试(空仓日) {len(tst):,} 样本 / {nte} 日  均值 {tst['y'].mean():.3f}%\n")

    feats = [f for f in dict.fromkeys(BASE + DERIV) if f in trn.columns]
    edges, codes = {}, {}
    for f in feats:
        v = trn[f].to_numpy("float32")
        ok = np.isfinite(v)
        qs = np.unique(np.nanquantile(v[ok], np.linspace(0, 1, NB + 1)[1:-1]))
        edges[f] = qs
        b = np.digitize(v, qs).astype("int8")
        b[~ok] = NA
        codes[f] = b

    year = (trn["trade_date"].to_numpy() // 10000).astype(int)
    yrs = np.unique(year)
    yi = np.searchsorted(yrs, year).astype("int16")
    yv = trn["y"].to_numpy("float32")
    yex = trn["y_ex"].to_numpy("float32")
    nday = trn.groupby("trade_date").size().to_numpy()

    rows = []
    for a, b in itertools.combinations(feats, 2):
        key = (codes[a].astype("int32") * 6 + codes[b]) * len(yrs) + yi
        nb = 36 * len(yrs)
        cnt = np.bincount(key, minlength=nb).astype("float32").reshape(36, len(yrs))
        sy = np.bincount(key, weights=yv, minlength=nb).reshape(36, len(yrs))
        sx = np.bincount(key, weights=yex, minlength=nb).reshape(36, len(yrs))
        tot = cnt.sum(1)
        good = tot >= 600
        if not good.any():
            continue
        mean = np.divide(sy.sum(1), tot, out=np.zeros(36), where=tot > 0)
        mxe = np.divide(sx.sum(1), tot, out=np.zeros(36), where=tot > 0)
        ok_y = cnt >= 30
        pos = (sy > 0) & ok_y
        nok = ok_y.sum(1)
        cons = np.divide(pos.sum(1), nok, out=np.zeros(36), where=nok > 0)
        for g in np.where(good & (mean >= 1.0) & (mxe >= 0.2) & (cons >= 0.70))[0]:
            b1, b2 = g // 6, g % 6
            if b1 == NA or b2 == NA or nok[g] < 5:
                continue
            rows.append((a, b, int(b1), int(b2), int(tot[g]), float(mean[g]),
                         float(mxe[g]), float(cons[g])))

    r = pd.DataFrame(rows, columns=["f1", "f2", "b1", "b2", "n", "mean", "mex", "cons"])
    print(f"训练段通过 {len(r):,} 格")

    out = []
    for _, x in r.sort_values("mean", ascending=False).head(80).iterrows():
        s_tr = eval_cond(trn, edges, x.f1, x.b1, x.f2, x.b2)
        s_te = eval_cond(tst, edges, x.f1, x.b1, x.f2, x.b2)
        st, se = stat(s_tr, ntr), stat(s_te, nte)
        out.append({"f1": x.f1, "b1": x.b1, "f2": x.f2, "b2": x.b2,
                    "tr_n": st["n"], "tr_days": st["days"], "tr_cov": st["cover"],
                    "tr_y": st["y"], "tr_yex": st["y_ex"],
                    "te_n": se.get("n", 0), "te_days": se.get("days", 0),
                    "te_cov": se.get("cover", 0), "te_y": se.get("y", np.nan),
                    "te_yex": se.get("y_ex", np.nan), "te_win": se.get("win", np.nan),
                    "te_cons": se.get("cons", np.nan), "te_pos": se.get("pos_yr", "")})
    o = pd.DataFrame(out)
    o["ok"] = ((o["te_y"] > 0.8) & (o["te_yex"] > 0.15) & (o["te_cons"] >= 0.6) &
               (o["te_days"] >= 40) & (o["tr_cov"] <= 0.35))
    o = o.sort_values("te_y", ascending=False)
    print(o.round(3).head(40).to_string(index=False))
    o.to_csv(OUT / "w240_grids_oos.csv", index=False, encoding="utf-8-sig")
    print(f"\n样本外存活 {int(o['ok'].sum())} / {len(o)}")

    for _, x in o[o["ok"]].iterrows():
        q1, q2 = edges[x.f1], edges[x.f2]
        r1 = ("-inf" if x.b1 == 0 else f"{q1[x.b1-1]:.2f}", "+inf" if x.b1 >= len(q1) else f"{q1[x.b1]:.2f}")
        r2 = ("-inf" if x.b2 == 0 else f"{q2[x.b2-1]:.2f}", "+inf" if x.b2 >= len(q2) else f"{q2[x.b2]:.2f}")
        print(f"  {x.f1}∈[{r1[0]},{r1[1]}) & {x.f2}∈[{r2[0]},{r2[1]})  "
              f"训练 {x.tr_y:+.2f}%({x.tr_days}日) -> 测试 {x.te_y:+.2f}%(alpha {x.te_yex:+.2f}) "
              f"{x.te_days}日 {x.te_pos}")


if __name__ == "__main__":
    main()
