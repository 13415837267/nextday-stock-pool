"""第二百三十七轮: 从历史行情里挖条件(不是验证猜想)

用户要求: 不要一轮一个假说再证伪, 要直接从 1500 万行历史数据里找出"什么样的票第二天会涨"。

做法:
  1) 特征池 = 32 个原始行情特征 + 12 个交互衍生特征(全部可由历史行情算出)
  2) 每个特征在全样本上切 5 档(全局分位), 缺失单独一档
  3) **穷举所有二维组合**(C(44,2)=946 对 x 36 格 = 3.4 万个格子)
  4) 每个格子同时算: 样本数 / 单笔均值 / 胜率 / **分年正收益率** / 2018 后均值
  5) 只保留: n>=1500, 均值>=1.2%, 胜率>=60%, 分年正>=75%, 2018 后同向
  6) 输出可读条件, 并按"与现行策略的重叠度"排序 —— 与策略不重叠的才是新条件

关键: 分年一致率是硬门槛。一个条件若 15 年里 8 年正 7 年负, 均值再高也是噪声。
输出 output/w237_grids.csv / w237_top.csv
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "output" / "feat234"
OUT = ROOT / "output"

BASE = ["pct", "amp", "close_pos", "low_tail", "up_shadow", "gap", "intraday",
        "turnover_rate", "amt_yi", "amt_ratio", "circ_mv",
        "bias5", "bias10", "bias20", "bias60",
        "dist_high60", "dist_low60", "dist_high20",
        "ret3", "ret5", "ret10", "ret20", "vol20", "maxdd20",
        "dn_streak", "up_streak", "rel_pct", "rs20",
        "breadth", "idx_dist", "listed"]

COLS = list(dict.fromkeys(["trade_date", "y", "n1_ok", "is_paused", "is_st", "pct", "listed"] + BASE))
NB = 5      # 每特征 5 档 + 1 个缺失档 = 6
NA = NB     # 缺失档编号


def load() -> pd.DataFrame:
    parts = []
    for p in sorted(FEAT.glob("f*.parquet")):
        d = pd.read_parquet(p, columns=COLS)
        d = d[(d["is_paused"] == 0) & (d["is_st"] == 0)]
        d = d[d["n1_ok"] == True]                      # noqa: E712
        d = d[d["pct"].between(-11.5, 11.5)]
        d = d[d["y"].between(-40, 60)]
        d = d[d["listed"] >= 250]
        if not len(d):
            continue
        for c in BASE:
            d[c] = d[c].astype("float32")
        # ---- 交互衍生特征(只依赖历史行情) ----
        d["mv_x_turn"] = d["circ_mv"] * d["turnover_rate"]
        d["amt_x_turn"] = d["amt_yi"] / d["turnover_rate"].clip(lower=0.1)
        d["b60_m_b20"] = d["bias60"] - d["bias20"]
        d["r5_m_r20"] = d["ret5"] - d["ret20"]
        d["h20_m_h60"] = d["dist_high20"] - d["dist_high60"]
        d["dd_m_pct"] = d["maxdd20"] - d["pct"]
        d["pos_x_tail"] = d["close_pos"] * d["low_tail"]
        d["vol_x_pct"] = d["vol20"] * d["pct"]
        d["dn_x_amp"] = d["dn_streak"] * d["amp"]
        d["amp_x_turn"] = d["amp"] * d["turnover_rate"]
        d["gap_x_vr"] = d["gap"] / (d["amt_ratio"].abs().clip(lower=0.2))
        d["ln_mv"] = np.log(d["circ_mv"].clip(lower=1))
        parts.append(d[list(dict.fromkeys(
            ["trade_date", "y", "pct", "close_pos", "turnover_rate", "circ_mv"] +
            BASE + ["mv_x_turn", "amt_x_turn", "b60_m_b20", "r5_m_r20", "h20_m_h60",
                               "dd_m_pct", "pos_x_tail", "vol_x_pct", "dn_x_amp",
                               "amp_x_turn", "gap_x_vr", "ln_mv"]))])
        print(f"  {p.stem}: {len(d):,}", flush=True)
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    d = load()
    print(f"\n可执行样本 {len(d):,} / {d['trade_date'].nunique()} 日 / 均值 {d['y'].mean():.3f}%")

    # 衍生特征去重(与 BASE 里已有的相关性过高就不用了, 这里先全留)
    feats = BASE + ["mv_x_turn", "amt_x_turn", "b60_m_b20", "r5_m_r20", "h20_m_h60",
                    "dd_m_pct", "pos_x_tail", "vol_x_pct", "dn_x_amp", "amp_x_turn",
                    "gap_x_vr", "ln_mv"]
    feats = [f for f in dict.fromkeys(feats) if f in d.columns]
    print(f"特征 {len(feats)} 个 -> 二维组合 {len(feats)*(len(feats)-1)//2} 对")

    # ---- 分箱: 全局分位, 缺失单列 ----
    codes = {}
    edges = {}
    for f in feats:
        v = d[f].to_numpy("float32")
        ok = np.isfinite(v)
        qs = np.nanquantile(v[ok], np.linspace(0, 1, NB + 1)[1:-1])
        qs = np.unique(qs)
        b = np.digitize(v, qs).astype("int8")
        b[~ok] = NA
        codes[f] = b
        edges[f] = qs

    year = (d["trade_date"].to_numpy() // 10000).astype("int16")
    yrs = np.unique(year)
    yi = np.searchsorted(yrs, year).astype("int16")
    y = d["y"].to_numpy("float32")
    win = (y > 0).astype("float32")
    nyr = len(yrs)
    print(f"年份 {yrs[0]}-{yrs[-1]} ({nyr} 年)")

    rows = []
    for a, b in itertools.combinations(feats, 2):
        key = codes[a].astype("int32") * 6 + codes[b]
        key = key * nyr + yi
        nb = 36 * nyr
        cnt = np.bincount(key, minlength=nb).astype("float32")
        sy = np.bincount(key, weights=y, minlength=nb)
        sw = np.bincount(key, weights=win, minlength=nb)
        cnt = cnt.reshape(36, nyr)
        sy = sy.reshape(36, nyr)
        sw = sw.reshape(36, nyr)
        tot = cnt.sum(1)
        good = tot >= 1500
        if not good.any():
            continue
        mean = np.divide(sy.sum(1), tot, out=np.zeros(36), where=tot > 0)
        wr = np.divide(sw.sum(1), tot, out=np.zeros(36), where=tot > 0)
        # 分年正收益率: 该年样本>=40 的年份里, 均值为正的比例
        ok_y = cnt >= 40
        pos = (sy > 0) & ok_y
        nok = ok_y.sum(1)
        cons = np.divide(pos.sum(1), nok, out=np.zeros(36), where=nok > 0)
        # 2018 后
        i18 = int(np.searchsorted(yrs, 2018))
        c18 = cnt[:, i18:].sum(1)
        s18 = sy[:, i18:].sum(1)
        m18 = np.divide(s18, c18, out=np.zeros(36), where=c18 > 0)

        for g in np.where(good & (mean >= 1.2) & (wr >= 0.60) & (cons >= 0.70))[0]:
            b1, b2 = g // 6, g % 6
            if b1 == NA or b2 == NA:      # 缺失档集中在早年, 一致率是假的
                continue
            if nok[g] < 12:
                continue
            rows.append((a, b, b1, b2, int(tot[g]), float(mean[g]), float(wr[g]),
                         int(nok[g]), float(cons[g]), float(m18[g]), int(c18[g])))
        if len(rows) and len(rows) % 20000 < 36:
            print(f"  ... {a}x{b}, 累计候选 {len(rows):,}", flush=True)

    r = pd.DataFrame(rows, columns=["f1", "f2", "b1", "b2", "n", "mean", "win",
                                    "n_year", "cons", "mean18", "n18"])
    print(f"\n通过初筛的格子 {len(r):,}")
    r = r[(r["mean18"] >= 0.8) & (r["n18"] >= 300)].sort_values("mean", ascending=False)
    print(f"2018 后同向的 {len(r):,}")

    # ---- 反查分箱区间 -> 可读条件 ----
    def rng(f: str, b: int) -> str:
        q = edges[f]
        if b == NA:
            return "缺失"
        lo = "-inf" if b == 0 else f"{q[b-1]:.2f}"
        hi = "+inf" if b >= len(q) else f"{q[b]:.2f}"
        return f"[{lo},{hi})"

    r["cond"] = [f"{a}∈{rng(a, x)} & {b_}∈{rng(b_, y_)}"
                 for a, b_, x, y_ in zip(r["f1"], r["f2"], r["b1"], r["b2"])]
    r = r[["cond", "f1", "f2", "b1", "b2", "n", "mean", "win", "n_year", "cons", "mean18", "n18"]]
    r.to_csv(OUT / "w237_grids.csv", index=False, encoding="utf-8-sig")

    # 供 verify_237b 直接消费, 避免手工抄阈值抄错
    keep = []
    for _, row in r.head(40).iterrows():
        q1, q2 = edges[row["f1"]], edges[row["f2"]]
        keep.append({
            "f1": row["f1"], "b1": int(row["b1"]),
            "lo1": None if row["b1"] == 0 else float(q1[row["b1"] - 1]),
            "hi1": None if row["b1"] >= len(q1) else float(q1[row["b1"]]),
            "f2": row["f2"], "b2": int(row["b2"]),
            "lo2": None if row["b2"] == 0 else float(q2[row["b2"] - 1]),
            "hi2": None if row["b2"] >= len(q2) else float(q2[row["b2"]]),
            "n": int(row["n"]), "mean": float(row["mean"]), "win": float(row["win"]),
            "cons": float(row["cons"]), "mean18": float(row["mean18"]), "n18": int(row["n18"]),
        })
    (OUT / "w237_top.json").write_text(
        json.dumps(keep, ensure_ascii=False, indent=1), encoding="utf-8")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 70)
    print("\n=== 按单笔收益 top 25")
    print(r.head(25)[["cond", "n", "mean", "win", "n_year", "cons", "mean18", "n18"]]
          .round(3).to_string(index=False))
    print(f"\n-> {OUT / 'w237_grids.csv'}")


if __name__ == "__main__":
    main()
