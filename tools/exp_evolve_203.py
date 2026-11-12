# -*- coding: utf-8 -*-
"""第二百零三轮 · 自进化(A): 主策略评分的权重, 由数据每季度自己重算。

问题
----
现行 dip 评分权重(mom20 0.20 / neg_bias20 0.70 / neg_amount 0.05 /
neg_vol_ratio 0.02 / low_tail 0.03)是**某一次人工定稿**写死在 config.yaml 里的,
此后从未再问过市场"这五个因子的相对重要性变了吗"。

做法
----
在 202 轮建好的 42,327 个 dip 样本面板上做 walk-forward:
  · 每季度末(3/6/9/12 月最后一个交易日)重算一次权重;
  · 只用**已结算**的样本(信号日 <= 重算日 - 3 个交易日, 即 T+2 收盘已发生),
    避免把还没走完的收益当成训练标签;
  · 权重 = max(0, 过去 750 日日频 rank IC 的均值), 归一化; 全负则退回等权;
  · 下一季度用该权重对当日候选打分, 取第 1 名, 持到 T+2 收盘, 扣 0.3% 往返成本。

对照: 同一份样本、同一个取第 1 名的规则, 只把"动态权重"换成"静态权重"。

这是"自进化"里最保守的一种: 不新增因子、不改结构, 只让**权重随数据漂移**。
它的价值必须体现在"动态 >= 静态"上, 否则就该退回静态 —— 脚本如实输出。

用法: python tools/exp_evolve_203.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"

# 面板里可用的四个因子(与 config 的 dip_weights 对应, low_tail 面板未存)
FACTORS = {"mom20": 1.0, "neg_bias20": -1.0, "neg_amount": -1.0, "neg_vol_ratio": -1.0}
# 静态权重: config.yaml 现值(mom20 .20 / neg_bias .70 / neg_amount .05 / neg_vr .02),
# 去掉 low_tail .03 后按剩余 0.97 归一化
STATIC_W = {"mom20": 0.20 / 0.97, "neg_bias20": 0.70 / 0.97,
            "neg_amount": 0.05 / 0.97, "neg_vol_ratio": 0.02 / 0.97}
TRAIN_DAYS = 750
COST = 0.3  # %


def _zscore(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """每日横截面 z 分(先 1/99 分位缩尾, 再标准化); 当日样本 < 5 只时置 0。"""
    out = pd.DataFrame(index=df.index)
    for c in cols:
        s = df[c].astype(float).clip(*df[c].quantile([0.01, 0.99]).tolist())
        g = s.groupby(df["d"])
        mu, sd = g.transform("mean"), g.transform("std").replace(0, np.nan)
        out[c] = ((s - mu) / sd).fillna(0.0)
    return out


def _daily_ic(df: pd.DataFrame, z: pd.DataFrame) -> pd.DataFrame:
    """每日 rank IC: 因子 z 分与 ret 的 Spearman 相关(当日样本 >= 8 才计算)。"""
    rows = []
    for d, idx in df.groupby("d").groups.items():
        if len(idx) < 8:
            continue
        sub_z, sub_r = z.loc[idx], df.loc[idx, "ret"].rank()
        rec = {"d": d}
        for c in z.columns:
            rec[c] = sub_z[c].corr(sub_r, method="spearman")
        rows.append(rec)
    return pd.DataFrame(rows).set_index("d").sort_index()


def _nav_stats(nav: pd.Series) -> dict:
    ret = nav.pct_change().fillna(0.0)
    yrs = len(nav) / 244.0
    total = float(nav.iloc[-1])
    ann = (total ** (1 / yrs) - 1) * 100 if yrs > 0 else 0.0
    dd = float((nav / nav.cummax() - 1).min() * 100)
    return {"总倍数": round(total, 2), "年化%": round(ann, 2), "回撤%": round(dd, 2),
            "卡玛": round(ann / abs(dd), 2) if dd < 0 else None,
            "年化波动%": round(float(ret.std() * np.sqrt(244) * 100), 2)}


def main():
    df = pd.read_parquet(OUT / "panel_dip_202.parquet").sort_values(["d", "ts_code"])
    df["d"] = df["d"].astype(str)
    df["neg_bias20"] = -df["bias20"]
    df["neg_amount"] = -df["amt_yi"]
    df["neg_vol_ratio"] = -df["vr"]
    cols = list(FACTORS)
    z = _zscore(df, cols)
    ic = _daily_ic(df, z)
    print(f"面板 {len(df)} 样本 / {df['d'].nunique()} 日, IC 覆盖 {len(ic)} 日", flush=True)

    # 每个因子的分年 IC 均值 —— 回答"权重该不该动"
    ic_y = ic.copy()
    ic_y["y"] = [i[:4] for i in ic_y.index]
    by_year = ic_y.groupby("y")[cols].mean().round(4)
    print(by_year.to_string(), flush=True)

    dates = sorted(df["d"].unique())
    pos = {d: i for i, d in enumerate(dates)}
    # 季度末: 每年 3/6/9/12 月的最后一个交易日
    rebal = [d for i, d in enumerate(dates)
             if i == len(dates) - 1 or dates[i + 1][:6] != d[:6]
             and d[4:6] in ("03", "06", "09", "12")]

    # 逐日候选打分(两套权重), 取第 1 名
    idx_by_day = {d: idx for d, idx in df.groupby("d").groups.items()}
    dyn_w = dict(STATIC_W)
    curve_dyn, curve_sta, hist_w = [], [], []
    for d in dates:
        idx = idx_by_day.get(d)
        if idx is None or len(idx) == 0:
            continue
        if d in rebal:
            p = pos[d]
            train = [x for x in dates[max(0, p - TRAIN_DAYS):p - 3]]
            sub = ic.loc[[x for x in train if x in ic.index]]
            if len(sub) >= 120:
                raw = {c: max(0.0, float(sub[c].mean())) for c in cols}
                s = sum(raw.values())
                dyn_w = ({c: raw[c] / s for c in cols} if s > 0
                         else {c: 1.0 / len(cols) for c in cols})
                hist_w.append({"d": d, **{c: round(v, 3) for c, v in dyn_w.items()}})

        def pick(w, idx):
            sc = sum(z.loc[idx, c].to_numpy() * w[c] for c in cols)
            return idx[int(np.argmax(sc))]

        r_dyn = float(df.loc[pick(dyn_w, idx), "ret"]) - COST
        r_sta = float(df.loc[pick(STATIC_W, idx), "ret"]) - COST
        curve_dyn.append((d, r_dyn))
        curve_sta.append((d, r_sta))

    res = {}
    for name, cur in (("dyn", curve_dyn), ("sta", curve_sta)):
        s = pd.Series({d: r for d, r in cur}).sort_index()
        nav = (1 + s / 100).cumprod()
        res[name] = _nav_stats(nav)
        if name == "dyn":
            nav_dyn = nav
        else:
            nav_sta = nav

    # 分年
    yearly = {}
    for tag, nav in (("dyn", nav_dyn), ("sta", nav_sta)):
        y = nav.copy()
        y.index = [str(i)[:4] for i in y.index]
        yearly[tag] = {k: round((v.iloc[-1] / v.iloc[0] - 1) * 100, 2)
                       for k, v in y.groupby(level=0)}

    out = {"panel": len(df), "ic_days": len(ic),
           "static_w": {k: round(v, 3) for k, v in STATIC_W.items()},
           "final_dyn_w": {k: round(v, 3) for k, v in dyn_w.items()},
           "dyn": res["dyn"], "sta": res["sta"], "yearly": yearly,
           "weight_history": hist_w, "ic_by_year": by_year.to_dict(),
           "corr_dyn_sta": round(float(nav_dyn.pct_change().corr(nav_sta.pct_change())), 3)}
    (OUT / "exp_evolve_203.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                             encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("dyn", "sta", "final_dyn_w", "corr_dyn_sta")},
                     ensure_ascii=False, indent=1), flush=True)
    print("分年:\n" + pd.DataFrame(yearly).to_string(), flush=True)
    print("写出 output/exp_evolve_203.json", flush=True)


if __name__ == "__main__":
    main()
