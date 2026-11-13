#!/usr/bin/env python3
"""第221轮: 可转债日内 T+0 的滚动训练。

与规则版的区别: 规则版用固定阈值(gap<-0.5%, vol<3%), 这里让模型在"今天所有低开的券"
里挑哪些日内会修复。标签是 close/open-1(开盘买、收盘卖的真实 T+0 收益)。

三个必须防的坑:
1. 泄漏: gap 在开盘时已知, 可以用; 但任何用到 close 的特征都是未来函数, 一律不用。
2. 幸存者偏差: 数据含 740 只已退市券, 不退市过滤。
3. 成本: 每笔扣 0.10%(佣金双边+保守滑点), 不是零成本纸面。

滚动方式: 训练 t-3~t-1 年, 预测 t 年。每年重训, 不用未来数据。
"""
from __future__ import annotations

import json
from pathlib import Path

from sklearn.ensemble import HistGradientBoostingRegressor
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COST = 0.0010

FEATS = [
    "gap", "vol20", "rv5", "rv20", "mom3", "mom10", "mom20",
    "dist60", "dist120", "px", "log_px", "amt20", "log_amt20",
    "turn_proxy", "amp20", "gap_z", "vol_ratio",
]


def build() -> pd.DataFrame:
    d = pd.read_parquet(ROOT / "data" / "cb" / "daily.parquet")
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values(["code", "date"]).reset_index(drop=True)
    g = d.groupby("code", observed=True, sort=False)

    d["pc"] = g["close"].shift(1)
    d["gap"] = d["open"] / d["pc"] - 1
    d["r1"] = d["close"] / d["pc"] - 1
    d["intr"] = d["close"] / d["open"] - 1
    d["amt"] = d["close"] * d["volume"]
    d["amt_prev"] = g["amt"].shift(1)

    # 全部只用 T-1 及之前的信息
    # 全部 shift(1): T 日开盘下单时, 当日 close 尚不存在。少一个 shift 就是泄漏。
    d["vol20"] = g["r1"].transform(lambda x: x.rolling(20).std().shift(1))
    d["rv5"] = g["r1"].transform(lambda x: x.rolling(5).std().shift(1))
    d["rv20"] = d["vol20"]
    d["mom3"] = g["r1"].transform(lambda x: x.rolling(3).sum().shift(1))
    d["mom10"] = g["r1"].transform(lambda x: x.rolling(10).sum().shift(1))
    d["mom20"] = g["r1"].transform(lambda x: x.rolling(20).sum().shift(1))
    d["hi60"] = g["close"].transform(lambda x: x.rolling(60).max().shift(1))
    d["hi120"] = g["close"].transform(lambda x: x.rolling(120).max().shift(1))
    d["dist60"] = d["pc"] / d["hi60"] - 1
    d["dist120"] = d["pc"] / d["hi120"] - 1
    d["px"] = d["pc"]
    d["log_px"] = np.log(d["pc"].clip(lower=1))
    d["amt20"] = g["amt"].transform(lambda x: x.rolling(20).mean().shift(1))
    d["log_amt20"] = np.log1p(d["amt20"].clip(lower=1))
    # 换手代理: 成交额/价格 -> 张数, 再相对自身20日均值
    d["turn_proxy"] = d["amt_prev"] / d["amt20"].replace(0, np.nan)
    d["amp20"] = d["vol20"]          # 与 vol20 同源, 保留列名避免下游改动
    d["gap_z"] = d["gap"] / d["vol20"].replace(0, np.nan)
    d["vol_ratio"] = d["rv5"] / d["vol20"].replace(0, np.nan)

    # 池子: 只留真实可交易
    d = d[(d["gap"].between(-0.35, 0.35)) & (d["px"] >= 50) & (d["px"] <= 200)]
    d = d[d["amt_prev"] >= 5e6]
    d = d[d["intr"].between(-0.30, 0.30)]
    d["year"] = d["date"].dt.year
    d = d.dropna(subset=FEATS + ["intr"])
    return d.sort_values("date").reset_index(drop=True)


def run(d: pd.DataFrame) -> dict:
    years = sorted(d["year"].unique())
    print(f"样本 {len(d):,} 行 {d.code.nunique()} 券 {d.date.min().date()}~{d.date.max().date()}")
    print(f"年份 {years}")

    preds = []
    imp = {}
    for y in years[3:]:
        tr = d[(d["year"] >= y - 3) & (d["year"] < y)]
        te = d[d["year"] == y]
        if len(tr) < 5000 or len(te) < 300:
            continue
        m = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.04, max_leaf_nodes=31,
            min_samples_leaf=150, l2_regularization=1.0, random_state=7,
        )
        m.fit(tr[FEATS].values, tr["intr"].values)
        p = m.predict(te[FEATS].values)
        preds.append(pd.DataFrame({"date": te["date"].values, "code": te["code"].values,
                                   "pred": p, "intr": te["intr"].values, "gap": te["gap"].values,
                                   "amt_prev": te["amt_prev"].values}))
        if hasattr(m, "feature_importances_"):
            for f, v in zip(FEATS, m.feature_importances_):
                imp[f] = imp.get(f, 0) + float(v)
        ic = pd.Series(p).corr(pd.Series(te["intr"].values), method="spearman")
        print(f"  {y}: train {len(tr):>7,} test {len(te):>6,} 日IC={ic:+.4f}")

    if not preds:
        return {}
    p = pd.concat(preds, ignore_index=True)
    return {"pred": p, "imp": imp}


def bt(p: pd.DataFrame, topk: int, cost: float = COST, min_gap: float = -0.02) -> dict:
    """每天只在 gap<-0.5% 的券里按模型分挑 topk 只, 等权, 开盘买收盘卖。"""
    x = p[p["gap"] <= -0.005]
    x = x[x["gap"] >= min_gap]
    rows = []
    for dt, g in x.groupby("date", sort=True):
        if len(g) < topk:
            continue
        sel = g.nlargest(topk, "pred")
        rows.append((dt, (sel["intr"] - cost).mean()))
    if not rows:
        return {}
    t = pd.DataFrame(rows, columns=["date", "net"])
    nav = (1 + t["net"]).prod()
    c, pk, mdd = 1.0, 1.0, 0.0
    for r in t["net"]:
        c *= 1 + r
        pk = max(pk, c)
        mdd = min(mdd, c / pk - 1)
    yrs = (t["date"].max() - t["date"].min()).days / 365.25
    ann = nav ** (1 / yrs) - 1
    return {"笔数": len(t), "年化%": round(ann * 100, 1), "总倍": round(nav, 1),
            "回撤%": round(mdd * 100, 1), "卡玛": round(ann / abs(mdd), 2) if mdd else None,
            "单笔%": round(float(t["net"].mean()) * 100, 3),
            "日胜率%": round(float((t["net"] > 0).mean() * 100), 1)}


def main() -> None:
    d = build()
    out = run(d)
    if not out:
        print("无结果")
        return
    p = out["pred"]
    res: dict = {"样本": {"行": int(len(d)), "券": int(d.code.nunique())}}

    print("\n=== 模型选股 (只在 gap<-0.5% 的券里挑) ===")
    print(f"{'topk':<7}{'年化%':>9}{'总倍':>10}{'回撤%':>8}{'卡玛':>7}{'单笔%':>8}{'日胜率%':>8}")
    for k in [1, 2, 3, 5, 8]:
        r = bt(p, k)
        if r:
            res[f"top{k}"] = r
            print(f"{k:<7}{r['年化%']:>9.1f}{r['总倍']:>10.1f}{r['回撤%']:>8.1f}"
                  f"{str(r['卡玛']):>7}{r['单笔%']:>8.3f}{r['日胜率%']:>8.1f}")

    print("\n=== 对照: 同一批券随机选 ===")
    rng = np.random.default_rng(0)
    p2 = p.copy()
    p2["pred"] = rng.random(len(p2))
    for k in [1, 3, 5]:
        r = bt(p2, k)
        if r:
            res[f"随机top{k}"] = r
            print(f"{k:<7}{r['年化%']:>9.1f}{r['总倍']:>10.1f}{r['回撤%']:>8.1f}"
                  f"{str(r['卡玛']):>7}{r['单笔%']:>8.3f}{r['日胜率%']:>8.1f}")

    print("\n=== 特征重要性 ===")
    for f, v in sorted(out["imp"].items(), key=lambda x: -x[1]):
        print(f"  {f:<12}{v:>9.0f}")

    res["重要性"] = {k: round(v, 0) for k, v in sorted(out["imp"].items(), key=lambda x: -x[1])}
    (ROOT / "output").mkdir(exist_ok=True)
    (ROOT / "output" / "train_cb_221.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    p.to_parquet(ROOT / "output" / "cb_pred_221.parquet", index=False)
    print("\nsaved output/train_cb_221.json")


if __name__ == "__main__":
    main()
