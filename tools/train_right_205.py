"""第 205 轮: 用全史训练副策略(右侧)的"池边界 + 排序键".

问题: 副策略现行是 低波前 10% 池 + mom20/ivol 排序 + H10。
      这两个选择有没有被全史几十万个样本检验过? 本研究回答:

  1. 池: 低波分位 0.05/0.10/0.20/0.30/无限制, 各自池内等权持 H 日的净收益;
  2. 键: 在低波前 10% 池内, 11 种排序键各选 top1, 报净收益/日级 t/分年;
  3. 形态: 池内再叠加"追高"条件(收盘位置/涨幅/贴前高), 看是加分还是减分;
  4. 持有期 1/3/5/10/20, 不重叠(stride=h)版本作为对照.

判据: 只有"不重叠 + 日级 t>=2 + 多数年份为正"才算可落地;
      单格峰值一律视为噪声(196~204 轮教训).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
COST = 0.30

KEYS = [
    ("mom20_ivol", "mom20/ivol 最大(现行)", lambda d: -d["mom20"] / d["ivol20"].replace(0, np.nan)),
    ("mom20", "20日动量最大", lambda d: -d["mom20"]),
    ("mom60", "60日动量最大", lambda d: -d["mom60"]),
    ("ivol", "波动最低", lambda d: d["ivol20"]),
    ("pct", "当日涨幅最大", lambda d: -d["pct"]),
    ("excess", "当日超额最大", lambda d: -d["excess"]),
    ("close_pos", "收在日内最高", lambda d: -d["close_pos"]),
    ("dist_ph20", "最贴20日前高", lambda d: -d["dist_ph20"]),
    ("turnover_cool", "换手最降温", lambda d: d["turnover_cool"]),
    ("vr", "放量最猛", lambda d: -d["vr"]),
    ("amt", "成交额最小", lambda d: d["amt_yi"]),
]


def tstat(x: pd.Series) -> float:
    x = x.dropna()
    if len(x) < 30:
        return 0.0
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))


def main() -> None:
    d = pd.read_parquet(OUT / "panel_all_205.parquet")
    bad = (d["ret10"].abs() > 120) | (d["gap"].abs() > 15) | (d["pct"].abs() > 11.6)
    d = d[~bad].reset_index(drop=True)
    d["year"] = d["d"].astype(str).str[:4].astype(int)
    print(f"样本 {len(d):,} 日 {d['d'].nunique():,} {d['d'].min()}→{d['d'].max()}")

    # 每日低波分位(ivol20 越小越"低波")
    d["ivr"] = d.groupby("d")["ivol20"].rank(pct=True)
    res = {"pool": [], "key": [], "form": [], "hold": [], "meta": {}}

    # ---- 1. 池边界: 池内等权, 各持有期 ----
    for cut in (0.05, 0.10, 0.20, 0.30, 1.01):
        pool = d[d["ivr"] <= cut]
        row = {"cut": cut, "n": int(len(pool)),
               "per_day": round(len(pool) / d["d"].nunique(), 1)}
        for h in (1, 3, 5, 10, 20):
            if f"ret{h}" not in pool.columns:
                continue
            net = pool[f"ret{h}"] - COST
            dm = net.groupby(pool["d"]).mean()
            row[f"h{h}"] = round(float(net.mean()), 3)
            row[f"t{h}"] = round(tstat(dm), 2)
        res["pool"].append(row)
        print("池", row, flush=True)

    # ---- 2. 低波前 10% 池内的排序键 ----
    pool = d[d["ivr"] <= 0.10].copy()
    for name, label, fn in KEYS:
        pool["_k"] = fn(pool)
        x = pool[["d", "_k", "year", "ret1", "ret3", "ret5", "ret10"]].dropna(subset=["_k"])
        idx = x.groupby("d")["_k"].idxmin()
        pick = x.loc[idx]
        row = {"key": name, "label": label, "n": int(len(pick))}
        for h in (1, 3, 5, 10):
            net = pick[f"ret{h}"] - COST
            dm = net.groupby(pick["d"]).mean()
            row[f"h{h}"] = round(float(net.mean()), 3)
            row[f"t{h}"] = round(tstat(dm), 2)
        y = (pick["ret10"] - COST).groupby(pick["year"]).mean()
        row["pos_years"] = f"{int((y > 0).sum())}/{len(y)}"
        row["win"] = round(float((pick["ret10"] - COST > 0).mean() * 100), 1)
        # 相对池均值的选择 alpha(H10)
        base = (pool["ret10"] - COST).groupby(pool["d"]).mean().mean()
        row["alpha"] = round(float(row["h10"] - base), 3)
        res["key"].append(row)
        print("键", row, flush=True)

    # ---- 3. 池内再叠加形态条件(追高 vs 不追高) ----
    forms = {
        "无(池内全部)": None,
        "收盘位置>=0.8(追高)": lambda x: x["close_pos"] >= 0.8,
        "收盘位置<=0.3(收在低位)": lambda x: x["close_pos"] <= 0.3,
        "当日涨幅>=3%": lambda x: x["pct"] >= 3,
        "当日涨幅<=0": lambda x: x["pct"] <= 0,
        "创20日新高": lambda x: x["dist_ph20"] >= 0,
        "距20日前高<=-5%": lambda x: x["dist_ph20"] <= -5,
        "量比>=1.5": lambda x: x["vr"] >= 1.5,
        "量比<=1.0(缩量)": lambda x: x["vr"] <= 1.0,
    }
    for label, fn in forms.items():
        sub = pool if fn is None else pool[fn(pool)]
        net = sub["ret10"] - COST
        dm = net.groupby(sub["d"]).mean()
        y = net.groupby(sub["year"]).mean()
        row = {"form": label, "n": int(len(sub)),
               "per_day": round(len(sub) / d["d"].nunique(), 1),
               "h10": round(float(net.mean()), 3), "t": round(tstat(dm), 2),
               "win": round(float((net > 0).mean() * 100), 1),
               "pos_years": f"{int((y > 0).sum())}/{len(y)}"}
        res["form"].append(row)
        print("形态", row, flush=True)

    # ---- 4. 不重叠对照: 每 h 个交易日一笔(现行 mom20_ivol / H10) ----
    pool["_k"] = -pool["mom20"] / pool["ivol20"].replace(0, np.nan)
    idx = pool.dropna(subset=["_k"]).groupby("d")["_k"].idxmin()
    pick = pool.loc[idx].sort_values("d").reset_index(drop=True)
    for h in (1, 3, 5, 10):
        for stride in (h, 5, 10):
            sub = pick.iloc[::stride]
            net = sub[f"ret{h}"] - COST
            res["hold"].append({"h": h, "stride": stride, "n": int(len(sub)),
                                "net": round(float(net.mean()), 3),
                                "t": round(tstat(net), 2),
                                "win": round(float((net > 0).mean() * 100), 1),
                                "mult": round(float(np.prod(1 + net / 100)), 1)})
            print("不重叠", res["hold"][-1], flush=True)

    res["meta"] = {"n": int(len(d)), "days": int(d["d"].nunique())}
    json.dump(res, open(OUT / "train_right_205.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    for k in ("pool", "key", "form", "hold"):
        pd.DataFrame(res[k]).to_csv(OUT / f"train_right_205_{k}.csv",
                                    index=False, encoding="utf-8")


if __name__ == "__main__":
    main()
