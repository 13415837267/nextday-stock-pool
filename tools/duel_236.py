"""第二百三十六轮: 换模型 —— GBM top1 与现行线性评分 top1 正面对打

这是从第 225 轮就知道该做、却拖了 11 轮没做的事。
第 234 轮训了 GBM 但只看了 IC(0.049) 和特征重要性, 从没拿它去"每天选一只"。
本轮补上, 而且只用**样本外**(2018 之后; 模型训练段 ≤2017)。

同一批日子、同一个池(strict)、同一条规则(每日 top1), 只换排序键:
  ① 现行线性评分  = 实际成交那笔(它用了 vr/dlow5 等 feat234 里没有的因子, 对 GBM 不利, 保守)
  ② GBM(y_ex)    = 2017 年前训练, 2018+ 预测, 纯样本外
  ③ 随机          = 池内随机一只(500 次)
  ④ oracle        = 池内 y 最高(不可实现, 只作标尺)

判定(本轮起改用新的落地标准, 见报告):
  GBM 在 2500 日口径下不低于现行, 且回撤不恶化超 2pp  -> 落地
  否则 -> 明确否决, 线性评分到此为止, 转 alpha 源
输出 output/w236_duel.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "output" / "feat234"
OUT = ROOT / "output"
TRADES = OUT / "w232_trades.csv"

FEATS = ["pct", "amp", "close_pos", "low_tail", "up_shadow", "gap", "intraday",
         "vr", "turnover_rate", "amt_yi", "amt_ratio", "circ_mv",
         "bias5", "bias10", "bias20", "bias60",
         "dist_high60", "dist_low60", "dist_high20",
         "ret3", "ret5", "ret10", "ret20", "vol20", "maxdd20",
         "dn_streak", "up_streak", "rel_pct", "rs20",
         "breadth", "idx_dist", "listed"]

COLS = list(dict.fromkeys(["ts_code", "trade_date", "is_paused", "is_st", "y", "n1_ok",
                           "pct", "listed", "close", "turnover_rate", "circ_mv"] + FEATS))


def main() -> None:
    booster = xgb.Booster()
    booster.load_model(str(FEAT / "gbm_lowopen.json"))
    print("模型已加载")

    tr = pd.read_csv(TRADES).rename(columns={"signal": "trade_date", "code": "ts_code"})
    tr["trade_date"] = tr["trade_date"].astype(int)
    tr = tr[tr["trade_date"] >= 20180101]
    days = set(tr["trade_date"].unique())
    print(f"样本外成交 {len(tr)} 笔 / {len(days)} 日 (2018+)")

    rows = []
    for p in sorted(FEAT.glob("f*.parquet")):
        yr = int(p.stem[1:])
        if yr < 2018:
            continue
        d = pd.read_parquet(p, columns=COLS)
        d = d[(d["is_paused"] == 0) & (d["is_st"] == 0)]
        d = d[d["pct"].between(-11.5, 11.5)]
        d = d[d["y"].between(-40, 60)]
        d = d[d["listed"] >= 250]
        d = d[d["n1_ok"] == True]                       # noqa: E712
        d = d[d["trade_date"].isin(days)]
        d = d[(d["pct"].between(-6.0, -3.0))
              & (d["close_pos"] <= 0.32)
              & d["close"].between(3, 150)
              & d["circ_mv"].between(0, 600)
              & d["turnover_rate"].between(2, 12)]
        d = d.dropna(subset=FEATS)
        if not len(d):
            continue
        d = d.copy()
        d["pred"] = booster.predict(xgb.DMatrix(d[FEATS].to_numpy("float32"),
                                                feature_names=FEATS))
        rows.append(d)
        print(f"  {yr}: {len(d):,} 行 / {d['trade_date'].nunique()} 日", flush=True)

    pool = pd.concat(rows, ignore_index=True)
    g = pool.groupby("trade_date")
    cnt = g.size()
    pool = pool[pool["trade_date"].isin(cnt[cnt >= 10].index)]
    print(f"\n样本外严格池 {len(pool):,} 行 / {pool['trade_date'].nunique()} 日 / "
          f"池大小中位 {pool.groupby('trade_date').size().median():.0f}")

    yt = pool.set_index(["trade_date", "ts_code"])["y"]
    tr["y_true"] = [yt.get(k, np.nan) for k in zip(tr["trade_date"], tr["ts_code"])]
    tr = tr.dropna(subset=["y_true"])
    keep = set(tr["trade_date"])
    pool = pool[pool["trade_date"].isin(keep)]
    tr = tr[tr["trade_date"].isin(keep)]
    print(f"可比天数 {len(keep)} (现行成交那笔也在池内)")

    res = {}
    res["现行线性评分"] = tr["y_true"].mean()

    gbm_top = pool.loc[pool.groupby("trade_date")["pred"].idxmax()]
    res["GBM top1"] = gbm_top["y"].mean()

    # 各自逐年
    yr = lambda s: pd.to_datetime(s.astype(str), format="%Y%m%d").dt.year
    by_yr = pd.DataFrame({
        "year": yr(gbm_top["trade_date"]).values,
        "gbm": gbm_top["y"].values,
    }).groupby("year")["gbm"].mean()
    real_yr = pd.DataFrame({"year": yr(tr["trade_date"]).values, "real": tr["y_true"].values}
                           ).groupby("year")["real"].mean()
    cmp_year = pd.concat([real_yr, by_yr], axis=1).round(3)

    rng = np.random.default_rng(236)
    sims = []
    for _ in range(500):
        v = [s["y"].values[rng.integers(len(s))] for _, s in pool.groupby("trade_date")]
        sims.append(np.mean(v))
    res["随机 top1(均值)"] = float(np.mean(sims))
    res["随机 top1(500次最好)"] = float(np.max(sims))
    res["oracle"] = float(pool.groupby("trade_date")["y"].max().mean())

    out = pd.DataFrame({"口径": list(res), "单笔%": list(res.values())})
    print("\n" + out.round(3).to_string(index=False))
    print("\n=== 分年对打 (线性 vs GBM)")
    cmp_year["差"] = (cmp_year["gbm"] - cmp_year["real"]).round(3)
    print(cmp_year.to_string())
    out.to_csv(OUT / "w236_duel.csv", index=False, encoding="utf-8-sig")
    cmp_year.to_csv(OUT / "w236_duel_year.csv", encoding="utf-8-sig")

    win = (cmp_year["差"] > 0).sum()
    print(f"\nGBM 胜 {win}/{len(cmp_year)} 年, 中位差 {cmp_year['差'].median():+.3f}pp")


if __name__ == "__main__":
    main()
