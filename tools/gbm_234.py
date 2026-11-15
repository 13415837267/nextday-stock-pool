"""第二百三十四轮: 全样本梯度提升 —— 让数据自己说哪些特征有用

单变量只能看"一个特征单独的解释力"。真实世界里特征是有交互的(跌幅 x 收盘位置 x 量比),
所以再跑一次 GBM: 它自己不假设函数形式, 也不假设哪些特征重要。

口径:
  训练 2009-2017, 验证 2018-2026 (严格时间外, 不看未来)
  标签 y_ex = 当日收益 - 当日全样本均值 (剔除 beta, 只留"选股能力")
  两个 scope: all(全样本) / lowopen(低开子集, 可执行)

产出:
  output/w234_gbm_imp_{scope}.csv     特征重要性(gain)
  output/w234_gbm_pdp_{scope}.csv     单特征部分依赖(20 个分位点上的预测值)
  控制台: 训练/验证 IC、分年 IC
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "output" / "feat234"

FEATS = ["pct", "amp", "close_pos", "low_tail", "up_shadow", "gap", "intraday",
         "vr", "turnover_rate", "amt_yi", "amt_ratio", "circ_mv",
         "bias5", "bias10", "bias20", "bias60",
         "dist_high60", "dist_low60", "dist_high20",
         "ret3", "ret5", "ret10", "ret20", "vol20", "maxdd20",
         "dn_streak", "up_streak", "rel_pct", "rs20",
         "breadth", "idx_dist", "listed"]


def load(scope: str) -> tuple[pd.DataFrame, list[str]]:
    paths = sorted(FEAT.glob("f*.parquet"))
    cols = ["trade_date", "is_paused", "is_st", "y", "n1_ok", "pct", "listed"] + FEATS
    cols = list(dict.fromkeys(cols))
    parts = []
    for p in paths:
        d = pd.read_parquet(p, columns=cols)
        d = d[(d["is_paused"] == 0) & (d["is_st"] == 0)]
        d = d[d["pct"].between(-11.5, 11.5)]
        d = d[d["y"].between(-40, 60)]
        d = d[d["listed"] >= 250]
        if scope == "lowopen":
            d = d[d["n1_ok"] == True]     # noqa: E712
        if len(d):
            parts.append(d)
    d = pd.concat(parts, ignore_index=True).dropna(subset=FEATS + ["y"])
    # 横截面去均值: 剔除 beta, 只留选股信息
    mu = d.groupby("trade_date")["y"].transform("mean")
    d["y_ex"] = d["y"] - mu
    print(f"{scope}: {len(d)} 行, {d['trade_date'].nunique()} 日", flush=True)
    return d, FEATS


def main() -> None:
    scope = sys.argv[1] if len(sys.argv) > 1 else "lowopen"
    d, feats = load(scope)
    d["year"] = (d["trade_date"] // 10000).astype("int16")

    tr = d[d["year"] <= 2017]
    va = d[d["year"] >= 2018]
    # 8GB 容器: 全量 817 万行 x 32 列 + DMatrix 副本会 OOM, 分层降采样到 220 万
    cap = 2_200_000
    if len(tr) > cap:
        tr = tr.sample(n=cap, random_state=0)
    if len(va) > cap:
        va = va.sample(n=cap, random_state=0)
    print(f"训练 {len(tr)} ({tr['year'].min()}-{tr['year'].max()})  "
          f"验证 {len(va)} ({va['year'].min()}-{va['year'].max()})", flush=True)

    Xtr = tr[feats].to_numpy("float32")
    ytr = tr["y_ex"].to_numpy("float32")
    Xva = va[feats].to_numpy("float32")
    yva = va["y_ex"].to_numpy("float32")

    params = dict(objective="reg:squarederror", max_depth=6, eta=0.05,
                  subsample=0.7, colsample_bytree=0.8, min_child_weight=200,
                  reg_lambda=5.0, tree_method="hist", max_bin=64, nthread=8)
    dtr = xgb.DMatrix(Xtr, ytr, feature_names=feats)
    dva = xgb.DMatrix(Xva, yva, feature_names=feats)
    booster = xgb.train(params, dtr, num_boost_round=400,
                        evals=[(dva, "va")], early_stopping_rounds=40,
                        verbose_eval=50)

    pva = booster.predict(dva, iteration_range=(0, booster.best_iteration + 1))
    ptr = booster.predict(dtr, iteration_range=(0, booster.best_iteration + 1))

    def ic(pred: np.ndarray, yy: np.ndarray) -> float:
        return float(np.corrcoef(pred, yy)[0, 1])

    print(f"\nIC 训练 {ic(ptr, ytr):.4f}  验证 {ic(pva, yva):.4f}", flush=True)

    va = va.copy()
    va["pred"] = pva
    rows = []
    for y, g in va.groupby("year"):
        rows.append({"year": int(y), "n": len(g), "ic": round(ic(g["pred"].values, g["y_ex"].values), 4),
                     "q5-q1": round(float(g.nlargest(len(g) // 5, "pred")["y_ex"].mean()
                                          - g.nsmallest(len(g) // 5, "pred")["y_ex"].mean()), 3)})
    yr = pd.DataFrame(rows)
    print("\n分年验证 IC 与 多头五分位超额(%)", flush=True)
    print(yr.to_string(index=False), flush=True)
    yr.to_csv(ROOT / "output" / f"w234_gbm_year_{scope}.csv", index=False, encoding="utf-8-sig")

    imp = booster.get_score(importance_type="gain")
    imp_s = pd.Series(imp).sort_values(ascending=False)
    imp_s = (imp_s / imp_s.sum() * 100).round(2)
    imp_df = imp_s.rename_axis("特征").reset_index(name="gain占比%")
    print("\n==== 特征重要性 ====", flush=True)
    print(imp_df.to_string(index=False), flush=True)
    imp_df.to_csv(ROOT / "output" / f"w234_gbm_imp_{scope}.csv", index=False, encoding="utf-8-sig")

    # 单特征部分依赖: 在验证集上把该特征替换成 20 个分位点, 看预测怎么变
    pdp_rows = []
    for f in feats:
        if f not in feats:
            continue
        qs = np.nanquantile(Xva[:, feats.index(f)], np.linspace(0.02, 0.98, 20))
        for q in qs:
            X2 = Xva.copy()
            X2[:, feats.index(f)] = q
            p = booster.predict(xgb.DMatrix(X2, feature_names=feats),
                                iteration_range=(0, booster.best_iteration + 1))
            pdp_rows.append({"特征": f, "取值": float(q), "预测超额%": float(p.mean())})
        print(f"  pdp {f}", flush=True)
    pd.DataFrame(pdp_rows).to_csv(ROOT / "output" / f"w234_gbm_pdp_{scope}.csv",
                                  index=False, encoding="utf-8-sig")
    booster.save_model(str(FEAT / f"gbm_{scope}.json"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
