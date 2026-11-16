"""第二百三十九轮: 把挖掘做成真正的样本外 —— 用 2009~2018 挖, 拿 2019~2026 验

第 237 轮的问题: 条件是在全史(2009~2026)上挖的, 又在同一段上回测, 100% 样本内。
本轮改成标准 walk-forward:
  训练段 2009-01 ~ 2018-12  ->  定分箱阈值(只用训练段分位), 统计, 筛选
  测试段 2019-01 ~ 2026-11  ->  阈值原样照搬, 只看条件成不成立

判定标准(测试段):
  ① 单笔均值 > 0 且能覆盖足够天数
  ② 扣掉当日全市场均值后仍然 > 0 (alpha 还在)
  ③ 分年正收益率 >= 60%
三项全过才算"样本外活下来"。
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
TRAIN_END = 2018
TEST_START = 2019
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


def stats(s: pd.DataFrame) -> dict:
    if not len(s):
        return {"n": 0}
    yr = (s["trade_date"] // 10000).astype(int)
    g = s.groupby(yr)["y"].agg(["size", "mean"])
    g = g[g["size"] >= 40]
    return {"n": len(s), "days": int(s["trade_date"].nunique()),
            "y": round(float(s["y"].mean()), 3),
            "y_ex": round(float(s["y_ex"].mean()), 3),
            "win": round(float((s["y"] > 0).mean() * 100), 1),
            "pos_yr": f"{int((g['mean'] > 0).sum())}/{len(g)}",
            "cons": round(float((g["mean"] > 0).mean()), 2) if len(g) else np.nan,
            "worst_yr": round(float(g["mean"].min()), 2) if len(g) else np.nan}


def main() -> None:
    d = load()
    d["y_ex"] = d["y"] - d.groupby("trade_date")["y"].transform("mean")
    yr = (d["trade_date"] // 10000).astype(int)
    tr = d[yr <= TRAIN_END].copy()
    te = d[yr >= TEST_START].copy()
    print(f"训练段 {len(tr):,} 样本 / {tr['trade_date'].nunique()} 日")
    print(f"测试段 {len(te):,} 样本 / {te['trade_date'].nunique()} 日")
    print(f"训练均值 {tr['y'].mean():.3f}%  测试均值 {te['y'].mean():.3f}%")
    print(f"训练 alpha {tr['y_ex'].mean():.3f}%  测试 alpha {te['y_ex'].mean():.3f}%\n")

    feats = [f for f in dict.fromkeys(BASE + DERIV) if f in tr.columns]

    # ---- 分箱: 只用训练段分位 ----
    edges, codes = {}, {}
    for f in feats:
        v = tr[f].to_numpy("float32")
        ok = np.isfinite(v)
        qs = np.unique(np.nanquantile(v[ok], np.linspace(0, 1, NB + 1)[1:-1]))
        edges[f] = qs
        b = np.digitize(v, qs).astype("int8")
        b[~ok] = NA
        codes[f] = b

    year = (tr["trade_date"].to_numpy() // 10000).astype(int)
    yrs = np.unique(year)
    yi = np.searchsorted(yrs, year).astype("int16")
    yv = tr["y"].to_numpy("float32")
    nyr = len(yrs)
    print(f"训练年份 {yrs[0]}-{yrs[-1]} ({nyr} 年)  特征 {len(feats)} 个\n")

    rows = []
    for a, b in itertools.combinations(feats, 2):
        key = (codes[a].astype("int32") * 6 + codes[b]) * nyr + yi
        nb = 36 * nyr
        cnt = np.bincount(key, minlength=nb).astype("float32").reshape(36, nyr)
        sy = np.bincount(key, weights=yv, minlength=nb).reshape(36, nyr)
        tot = cnt.sum(1)
        good = tot >= 1200
        if not good.any():
            continue
        mean = np.divide(sy.sum(1), tot, out=np.zeros(36), where=tot > 0)
        ok_y = cnt >= 30
        pos = (sy > 0) & ok_y
        nok = ok_y.sum(1)
        cons = np.divide(pos.sum(1), nok, out=np.zeros(36), where=nok > 0)
        for g in np.where(good & (mean >= 1.3) & (cons >= 0.75))[0]:
            b1, b2 = g // 6, g % 6
            if b1 == NA or b2 == NA or nok[g] < 8:
                continue
            rows.append((a, b, int(b1), int(b2), int(tot[g]), float(mean[g]), float(cons[g])))

    r = pd.DataFrame(rows, columns=["f1", "f2", "b1", "b2", "n", "mean", "cons"])
    print(f"训练段通过初筛 {len(r):,} 格")
    r = r.sort_values("mean", ascending=False).head(60)
    print(f"取前 {len(r)} 格做样本外检验\n")

    out = []
    for _, row in r.iterrows():
        s_tr = eval_cond(tr, edges, row.f1, row.b1, row.f2, row.b2)
        s_te = eval_cond(te, edges, row.f1, row.b1, row.f2, row.b2)
        st, se = stats(s_tr), stats(s_te)
        out.append({"f1": row.f1, "b1": row.b1, "f2": row.f2, "b2": row.b2,
                    "train_n": st["n"], "train_y": st["y"], "train_cons": st["cons"],
                    "test_n": se.get("n", 0), "test_days": se.get("days", 0),
                    "test_y": se.get("y", np.nan), "test_yex": se.get("y_ex", np.nan),
                    "test_win": se.get("win", np.nan), "test_pos_yr": se.get("pos_yr", ""),
                    "test_cons": se.get("cons", np.nan), "test_worst": se.get("worst_yr", np.nan)})

    o = pd.DataFrame(out)
    o["存活"] = ((o["test_y"] > 0.5) & (o["test_yex"] > 0.1) &
                 (o["test_cons"] >= 0.60) & (o["test_days"] >= 60))
    o = o.sort_values("test_y", ascending=False)
    print(o.round(3).to_string(index=False))
    o.to_csv(OUT / "w239_oos.csv", index=False, encoding="utf-8-sig")

    alive = o[o["存活"]]
    print(f"\n样本外存活 {len(alive)} / {len(o)} 格")
    if len(alive):
        for _, x in alive.iterrows():
            q1, q2 = edges[x.f1], edges[x.f2]
            r1 = ("-inf" if x.b1 == 0 else f"{q1[x.b1-1]:.2f}", "+inf" if x.b1 >= len(q1) else f"{q1[x.b1]:.2f}")
            r2 = ("-inf" if x.b2 == 0 else f"{q2[x.b2-1]:.2f}", "+inf" if x.b2 >= len(q2) else f"{q2[x.b2]:.2f}")
            print(f"  {x.f1}∈[{r1[0]},{r1[1]}) & {x.f2}∈[{r2[0]},{r2[1]})  "
                  f"训练 {x.train_y:+.2f}% -> 测试 {x.test_y:+.2f}% (alpha {x.test_yex:+.2f}) "
                  f"{x.test_days}日 {x.test_pos_yr}")

    json.dump({k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in edges.items()},
              open(OUT / "w239_edges.json", "w"), ensure_ascii=False)


if __name__ == "__main__":
    main()
