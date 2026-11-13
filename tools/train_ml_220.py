"""第 220 轮 · 全历史滚动训练：模型自己决定"哪天买、买哪只"

口径:
  · 严格逐年样本外: 用 t-3 ~ t-1 年训练, 预测 t 年(t = 2015..2026), 共 12 折。
  · 标签: T+1 开盘买 / T+2 收盘卖, 已扣 0.30% 双边成本(见 build_ml_220.py)。
  · 资金约束与实盘一致: 单吊全仓; 持仓期间(T+1、T+2)出现的信号全部丢弃。
  · 三种买入门槛, 都由**训练集**的分位数决定(不看测试集):
      A 每天必买     —— 笔数上限
      B 当日 top1 预测 > 训练集 top1 的 P60
      C 当日 top1 预测 > 训练集 top1 的 P85
  · 对照组: 每天在候选池里随机选 1 只(20 次平均), 这是"模型没有信息"的基线。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"

PARAMS = dict(
    objective="reg:squarederror",
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=80,
    reg_lambda=2.0,
    tree_method="hist",
    nthread=0,
    verbosity=0,
)
NROUND = 400
YEARS = list(range(2015, 2027))


def simulate(days: list[pd.DataFrame], q: float | None) -> dict:
    """按资金约束模拟。days 已按日期升序, 每个 df 含 [trade_date, ret, pred]。

    q=None 表示每天必买; 否则只在当日 top1 预测 > 阈值时买。
    阈值在 fold 层面由训练集决定, 这里通过闭包传入。
    """
    nav, peak, mdd = 1.0, 1.0, 0.0
    held_until = None          # 卖出日(含)之前不能再买
    rows = []
    skip = 0
    for dd in days:
        d = int(dd["trade_date"].iloc[0])
        if held_until is not None and d <= held_until:
            skip += 1
            continue
        top = dd.sort_values("pred", ascending=False).iloc[0]
        if q is not None and top["pred"] < q:
            skip += 1
            continue
        r = float(top["ret"]) / 100
        nav *= (1 + r)
        peak = max(peak, nav)
        mdd = min(mdd, nav / peak - 1)
        # 卖出日: 信号日之后第 2 个交易日 —— 用 days 的日期序列定位
        rows.append((d, r * 100, str(top["ts_code"])))
        idx = day_index[d]
        held_until = day_dates[min(idx + 2, len(day_dates) - 1)]
    rets = [x[1] for x in rows]
    n = len(rets)
    d0 = pd.Timestamp(str(int(days[0]["trade_date"].iloc[0])))
    d1 = pd.Timestamp(str(int(days[-1]["trade_date"].iloc[0])))
    yrs = max(1e-9, (d1 - d0).days / 365.25)
    return {
        "笔数": n, "跳过天数": skip,
        "年化%": round(((nav) ** (1 / yrs) - 1) * 100, 2),
        "总倍": round(nav, 1),
        "回撤%": round(mdd * 100, 2),
        "卡玛": round((((nav) ** (1 / yrs) - 1) * 100) / abs(mdd * 100), 2) if mdd else None,
        "单笔%": round(float(np.mean(rets)), 3) if rets else None,
        "胜率%": round(float(np.mean([r > 0 for r in rets]) * 100), 1) if rets else None,
    }


def main() -> None:
    d = pd.read_parquet(OUT / "ml_220.parquet")
    meta = json.loads((OUT / "ml_220_meta.json").read_text())
    feats = meta["feats"]
    d["trade_date"] = d["trade_date"].astype(int)
    print(f"样本 {len(d):,} 行 {d.trade_date.nunique()} 天 {d.year.min()}-{d.year.max()}", flush=True)

    global day_index, day_dates
    day_dates = sorted(d["trade_date"].unique())
    day_index = {x: i for i, x in enumerate(day_dates)}

    all_pred = []
    for y in YEARS:
        t0 = time.time()
        tr = d[(d.year >= y - 3) & (d.year <= y - 1)]
        te = d[d.year == y]
        if len(tr) < 50000 or len(te) < 5000:
            print(f"  {y}: 样本不足, 跳过 (tr={len(tr)}, te={len(te)})", flush=True)
            continue
        # 最后一年训练数据内再切 15% 做早停
        va_year = y - 1
        fit = tr[tr.year <= va_year - 1] if (tr.year == va_year).any() else tr
        val = tr[tr.year == va_year]
        if len(fit) < 50000 or len(val) < 5000:
            fit, val = tr, tr
        dtr = xgb.DMatrix(fit[feats].values, label=fit["ret"].values, nthread=0)
        dva = xgb.DMatrix(val[feats].values, label=val["ret"].values, nthread=0)
        bst = xgb.train(PARAMS, dtr, NROUND, evals=[(dva, "va")],
                        early_stopping_rounds=40, verbose_eval=False)
        dte = xgb.DMatrix(te[feats].values, nthread=0)
        pred = bst.predict(dte, iteration_range=(0, bst.best_iteration + 1))
        tr_pred = bst.predict(dtr, iteration_range=(0, bst.best_iteration + 1))

        # 训练集上"每日 top1 的预测值"分布 —— 门槛只用这个, 不碰测试集
        trtmp = pd.DataFrame({"d": fit["trade_date"].values, "p": tr_pred})
        tr_top = trtmp.groupby("d")["p"].max()
        q60, q85 = tr_top.quantile(0.60), tr_top.quantile(0.85)

        te = te[["trade_date", "ts_code", "ret"]].copy()
        te["pred"] = pred
        te["year"] = y
        te["q60"] = q60
        te["q85"] = q85
        all_pred.append(te)
        ic = te.groupby("trade_date").apply(
            lambda g: g["pred"].corr(g["ret"], method="spearman") if len(g) > 30 else np.nan)
        print(f"  {y}: best_iter={bst.best_iteration} train={len(fit):,} test={len(te):,} "
              f"日IC均值={ic.mean():+.4f} 正IC天占比={(ic > 0).mean()*100:.1f}% "
              f"({time.time()-t0:.0f}s)", flush=True)

    allp = pd.concat(all_pred, ignore_index=True)
    allp.to_parquet(OUT / "ml_220_pred.parquet", index=False)
    days = [g for _, g in allp.groupby("trade_date", sort=True)]

    th = {str(y): {"q60": float(allp[allp.year == y]["q60"].iloc[0]),
                   "q85": float(allp[allp.year == y]["q85"].iloc[0])} for y in YEARS
          if (allp.year == y).any()}

    res = {}
    for name, qf in [("A_每天必买", None), ("B_P60", 0.60), ("C_P85", 0.85)]:
        if qf is None:
            r = simulate(days, None)
            r["门槛"] = "无"
        else:
            # 逐日取当年门槛
            nav, peak, mdd = 1.0, 1.0, 0.0
            held_until, rows = None, []
            for dd in days:
                dt = int(dd["trade_date"].iloc[0])
                if held_until is not None and dt <= held_until:
                    continue
                top = dd.sort_values("pred", ascending=False).iloc[0]
                q = th[str(int(top["year"]))]
                lim = q["q60"] if qf < 0.7 else q["q85"]
                if top["pred"] < lim:
                    continue
                r_ = float(top["ret"]) / 100
                nav *= (1 + r_)
                peak = max(peak, nav)
                mdd = min(mdd, nav / peak - 1)
                rows.append(r_ * 100)
                idx = day_index[dt]
                held_until = day_dates[min(idx + 2, len(day_dates) - 1)]
            yrs = (pd.Timestamp(str(int(days[-1]["trade_date"].iloc[0])))
                   - pd.Timestamp(str(int(days[0]["trade_date"].iloc[0])))).days / 365.25
            r = {"笔数": len(rows), "年化%": round(((nav) ** (1 / yrs) - 1) * 100, 2),
                 "总倍": round(nav, 1), "回撤%": round(mdd * 100, 2),
                 "卡玛": round((((nav) ** (1 / yrs) - 1) * 100) / abs(mdd * 100), 2) if mdd else None,
                 "单笔%": round(float(np.mean(rows)), 3) if rows else None,
                 "胜率%": round(float(np.mean([x > 0 for x in rows]) * 100), 1) if rows else None,
                 "门槛": f"P{int(qf*100)}"}
        res[name] = r
        print(f"{name}: {r}", flush=True)

    # 随机基线: 每天在池内随机 1 只, 同样资金约束, 20 次
    rng = np.random.default_rng(42)
    rnd = []
    for _ in range(20):
        nav, held = 1.0, None
        cnt = 0
        for dd in days:
            dt = int(dd["trade_date"].iloc[0])
            if held is not None and dt <= held:
                continue
            r_ = float(dd["ret"].sample(1, random_state=int(rng.integers(1e9))).iloc[0]) / 100
            nav *= (1 + r_)
            cnt += 1
            held = day_dates[min(day_index[dt] + 2, len(day_dates) - 1)]
        yrs = (pd.Timestamp(str(int(days[-1]["trade_date"].iloc[0])))
               - pd.Timestamp(str(int(days[0]["trade_date"].iloc[0])))).days / 365.25
        rnd.append((((nav) ** (1 / yrs) - 1) * 100, cnt, nav))
    res["D_随机选1只"] = {"年化%": round(float(np.mean([x[0] for x in rnd])), 2),
                       "笔数": int(np.mean([x[1] for x in rnd])),
                       "总倍": round(float(np.mean([x[2] for x in rnd])), 2),
                       "门槛": "无(20次均值)"}
    print("D_随机选1只:", res["D_随机选1只"], flush=True)

    (OUT / "train_ml_220.json").write_text(
        json.dumps({"results": res, "thresholds": th,
                    "windows": f"{day_dates[0]}~{day_dates[-1]}"},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved")


if __name__ == "__main__":
    main()
