"""第 202 轮: 用 4.2 万个合格股票日, 替代 355 笔, 重测那几条结论.

输出: output/panel_eval_202.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"


def se_t(x: pd.Series) -> tuple[float, float]:
    x = x.dropna()
    if len(x) < 3:
        return float("nan"), float("nan")
    return float(x.mean()), float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))


def main() -> None:
    q = pd.read_parquet(OUT / "panel_dip_202.parquet")
    daily = pd.read_csv(OUT / "panel_daily_202.csv", dtype={"d": str}).set_index("d").sort_index()
    daily["nav"] = (1 + daily["ret"] / 100).cumprod()
    daily["ma20"] = daily["nav"].rolling(20).mean()
    daily["gate"] = (daily["nav"] >= daily["ma20"]).astype(int)
    daily["breadth"] = daily["up"]

    q["d"] = q["d"].astype(str)
    q = q.join(daily[["gate", "breadth"]], on="d")
    q["pool"] = q.groupby("d")["ret"].transform("size")
    q["ret"] = q["ret"].astype(float)

    res: dict = {}
    res["total"] = {"n": len(q), "days": int(q.d.nunique()),
                    "mean": round(float(q.ret.mean()), 3),
                    "median": round(float(q.ret.median()), 3),
                    "win": round(float((q.ret > 0).mean() * 100), 2),
                    "sd": round(float(q.ret.std()), 3)}
    res["by_year"] = (q.assign(y=q.d.str[:4]).groupby("y").ret
                      .agg(["size", "mean", "median", lambda s: (s > 0).mean() * 100])
                      .round(3).rename(columns={"size": "n", "<lambda_0>": "win"}).to_dict("index"))

    # 市场闸门
    g = {}
    for k, lab in [(1, "市场在MA20上"), (0, "市场在MA20下")]:
        sub = q[q.gate == k]
        m, t = se_t(sub.ret)
        g[lab] = {"n": len(sub), "mean": round(m, 3), "t": round(t, 2),
                  "win": round(float((sub.ret > 0).mean() * 100), 1),
                  "days": int(sub.d.nunique())}
    res["gate"] = g

    # 当日池深
    q["pb"] = pd.cut(q.pool, [0, 5, 10, 20, 40, 80, 10 ** 6],
                     labels=["1-5", "6-10", "11-20", "21-40", "41-80", "80+"])
    res["pool"] = (q.groupby("pb", observed=False)
                   .agg(n=("ret", "size"), days=("d", "nunique"), mean=("ret", "mean"),
                        win=("ret", lambda s: (s > 0).mean() * 100))
                   .round(3).to_dict("index"))

    # 当日合格票的平均收益 vs 市场当日等权
    day = q.groupby("d").agg(n=("ret", "size"), mean=("ret", "mean")).join(daily[["ret", "gate"]])
    day["ex"] = day["mean"] - day["ret"]
    res["excess"] = {"n_days": len(day), "mean": round(float(day.ex.mean()), 3),
                     "t": round(float(day.ex.mean() / (day.ex.std(ddof=1) / np.sqrt(len(day)))), 2),
                     "win_days": round(float((day.ex > 0).mean() * 100), 1)}

    # 特征分档
    feats = {}
    for col, bins in [("pct", [-6.01, -5.5, -5, -4.5, -4, -3.5, -3.0]),
                      ("close_pos", [-.01, .1, .2, .32]),
                      ("mom20", [4.9, 8, 12, 16, 20, 25.1]),
                      ("vr", [0, .8, 1.2, 2, 3, 4.01]),
                      ("turnover_rate", [1.9, 3, 5, 8, 12.1]),
                      ("amt_yi", [.79, 1.5, 3, 6, 12, 1e9])]:
        b = pd.cut(q[col], bins)
        feats[col] = (q.groupby(b, observed=False)
                      .agg(n=("ret", "size"), mean=("ret", "mean"),
                           win=("ret", lambda s: (s > 0).mean() * 100))
                      .round(3).astype({"n": int}).reset_index()
                      .assign(**{col: lambda d: d[col].astype(str)}).to_dict("records"))
    res["features"] = feats

    # 实盘那 717 笔 vs 同日池均值(配对)
    pair = {}
    for w in (1500, 3000):
        t = pd.read_csv(OUT / f"trades_201_{w}.csv", dtype={"signal_date": str, "code": str})
        t["d"] = t["signal_date"].str.replace("-", "")
        t["ret"] = t["ret"].astype(float)
        m = t.merge(q[["d", "ts_code", "ret"]].rename(columns={"ts_code": "code", "ret": "pool_ret"}),
                    on=["d", "code"], how="left")
        hit = m[m.pool_ret.notna()]
        dmean = q.groupby("d").ret.mean()
        hit = hit.assign(day_mean=hit.d.map(dmean))
        diff = hit.ret - hit.day_mean
        m2, t2 = se_t(diff)
        pair[w] = {"matched": len(hit), "trades": len(t),
                   "picked": round(float(hit.ret.mean()), 3),
                   "sameday_mean": round(float(hit.day_mean.mean()), 3),
                   "diff": round(m2, 3), "t": round(t2, 2),
                   "beat": round(float((diff > 0).mean() * 100), 1)}
    res["picked_vs_pool"] = pair

    json.dump(res, open(OUT / "panel_eval_202.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: res[k] for k in ("total", "gate", "excess", "picked_vs_pool")},
                     ensure_ascii=False, indent=1))
    print("pool:", json.dumps(res["pool"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
