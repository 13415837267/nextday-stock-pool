"""第 204 轮: 用梯度提升树在全史面板上训练选股模型, 与现行手调评分同口径对比.

前情(本轮的训练证据):
  · ridge 线性训练出的权重, 样本外相对现行只有 +0.04pp(t=0.46) —— 线性组合已到顶
  · 现行 score 却是**单调有效**的: 全部日 top1 -0.015%, score>=1.0 的 1083 天 +0.347%,
    score>=1.8 的 125 天 +0.810%。所以"评分"这一层是有信息的, 只是线性权重榨不出来
  · 复利数学: 减少笔数必输(1.0124^717 = 7000 倍, 而 1.008^125 = 2.7 倍)。因此唯一
    有价值的方向是 **在相同笔数下把单笔做高**, 不是筛得更狠

所以本轮训练 GBDT(HistGradientBoosting), 在**同一批买入日**(regime 允许 + 现行
score>=1.0 的 698 天)上换掉"选哪只", 看单笔能不能从 +0.28% 抬起来。

输出 output/train_gbm_204.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
COST = 0.30
FEATS = ["pct", "close_pos", "amp_pct", "low_tail", "body_pct", "mom3", "mom5",
         "mom10", "mom20", "bias5", "bias20", "dist_ma5", "dist_low20",
         "dist_high20", "ma20_slope", "vr", "vr5", "amt_yi", "amt_ratio",
         "turnover_rate", "turnover_cool", "vol20", "ivol20", "prev_pct",
         "down2", "price", "mkt_ret", "excess", "gap"]


def load() -> pd.DataFrame:
    d = pd.read_parquet(OUT / "panel_train_204.parquet")
    d = d[(d.ret.abs() <= 25) & (d.gap.abs() <= 15)]
    d["d"] = d["d"].astype(int)
    d["year"] = d["d"].astype(str).str[:4].astype(int)
    m = pd.read_csv(OUT / "panel_mkt_204.csv")
    m["d"] = m["d"].astype(int)
    m = m.sort_values("d")
    nav = (1 + m["mkt_pct"] / 100).cumprod()
    m["trade_ok"] = (nav > nav.rolling(20).mean()).values
    d = d.merge(m[["d", "trade_ok", "up_share"]], on="d", how="left")
    d["up_share"] = d["up_share"].fillna(0.5)
    d["trade_ok"] = d["trade_ok"].fillna(False)
    h = d[d.hard == 1].copy()

    # 现行 score(引擎口径: 当日 dip 池内 z 化, clip ±3)
    h["logamt"] = np.log1p(h["amt_yi"].astype("float64") * 1e8)
    gg = h.groupby("d", sort=False)

    def zs(s):
        mu = s.groupby(h["d"], sort=False).transform("mean")
        sd = s.groupby(h["d"], sort=False).transform("std").replace(0, np.nan)
        return ((s - mu) / sd).fillna(0).clip(-3, 3)

    h["score"] = (0.20 * zs(h["mom20"].astype("float64"))
                  + 0.70 * zs(-h["bias20"].astype("float64"))
                  + 0.05 * zs(-h["logamt"]) + 0.02 * zs(-h["vr"].astype("float64"))
                  + 0.03 * zs(h["low_tail"].astype("float64")))
    return h


def top1(d: pd.DataFrame, col: str) -> pd.Series:
    return d.loc[d.groupby("d", sort=False)[col].idxmax()]


def main() -> None:
    h = load()
    print(f"硬条件池 {len(h):,} 行 / {h.d.nunique()} 日", flush=True)
    rows = []
    for y in sorted(h["year"].unique()):
        tr = h[(h.year >= y - 3) & (h.year < y)]
        te = h[h.year == y]
        if len(tr) < 4000 or te.empty:
            continue
        Xtr, Xte = tr[FEATS].values, te[FEATS].values
        ytr = tr["ret"].values
        mdl = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_depth=4, min_samples_leaf=40,
            l2_regularization=1.0, random_state=204)
        mdl.fit(Xtr, ytr)
        te = te.assign(gbm=mdl.predict(Xte))
        a = top1(te, "gbm")
        b = top1(te, "score")
        sub = te[(te.trade_ok) & (te.score >= 1.0)]
        a2 = top1(sub, "gbm") if len(sub) else None
        b2 = top1(sub, "score") if len(sub) else None
        rows.append({
            "year": int(y), "te_n": int(len(te)), "days": int(te.d.nunique()),
            "gbm_all": round(float((a["ret"] - COST).mean()), 3),
            "cur_all": round(float((b["ret"] - COST).mean()), 3),
            "gbm_698": round(float((a2["ret"] - COST).mean()), 3) if a2 is not None else None,
            "cur_698": round(float((b2["ret"] - COST).mean()), 3) if b2 is not None else None,
            "days_698": int(len(a2)) if a2 is not None else 0,
            # 同一只票的比例: 高说明模型基本在复刻现行评分
            "same": round(float(np.mean(a["ts_code"].astype(str).values
                                        == b["ts_code"].astype(str).values) * 100), 1),
        })
        print(f"  {y} 测试 {len(te):5d} 行 / {te.d.nunique():3d} 日 | "
              f"全 GB {rows[-1]['gbm_all']:+.3f}% vs 现行 {rows[-1]['cur_all']:+.3f}% | "
              f"698 天 GB {rows[-1]['gbm_698']} vs {rows[-1]['cur_698']} | "
              f"同票 {rows[-1]['same']}%", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "train_gbm_204_yearly.csv", index=False)
    s1 = df[["gbm_all", "cur_all"]].mean()
    s2 = df[["gbm_698", "cur_698"]].dropna().mean()
    res = {"全样本_逐日均值": {"GBDT": round(float(s1["gbm_all"]), 3),
                              "现行": round(float(s1["cur_all"]), 3),
                              "差pp": round(float(s1["gbm_all"] - s1["cur_all"]), 3),
                              "GBDT 更好年数": int((df.gbm_all > df.cur_all).sum()),
                              "总年数": int(len(df))},
           "同一买入日(698天)_逐日均值": {"GBDT": round(float(s2["gbm_698"]), 3),
                                        "现行": round(float(s2["cur_698"]), 3),
                                        "差pp": round(float(s2["gbm_698"] - s2["cur_698"]), 3),
                                        "GBDT 更好年数": int((df.gbm_698 > df.cur_698).sum())},
           "选中同一只票比例%": round(float(df["same"].mean()), 1),
           "逐年": rows}
    (OUT / "train_gbm_204.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n", json.dumps({k: v for k, v in res.items() if k != "逐年"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
