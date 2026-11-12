"""第 205 轮: 右侧(追高)条件的全史单变量扫描.

对每个候选维度分档, 报告:
  n / 每日笔数 / 单笔净收益(已扣 0.3%) / 日级 t / 胜率 / 2018 前 / 2018 后 / 正收益年数

判据: 只有"分档单调 + 两段都为正 + 多年为正"的维度才算结构性 alpha,
单点峰值一律当噪声(196~204 轮已重复踩过五次).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
COST = 0.30

SPECS = [
    ("pct", "当日涨幅%", [-9, 1, 2, 3, 5, 7, 9.5, 11.5]),
    ("dist_ph20", "距20日高点%", [-99, -15, -10, -7, -5, -3, -1, 0, 99]),
    ("dist_ph60", "距60日高点%", [-99, -25, -15, -10, -5, -2, 0, 99]),
    ("dist_ph250", "距250日高点%", [-99, -40, -25, -15, -8, -3, 0, 99]),
    ("vr", "量比(20日)", [0, 0.8, 1.2, 1.5, 2, 3, 5, 99]),
    ("vr5", "量比(5日)", [0, 0.7, 1.0, 1.5, 2, 3, 99]),
    ("amt_ratio", "额比(20日)", [0, 0.8, 1.2, 1.8, 3, 99]),
    ("turnover_rate", "换手%", [0, 3, 4, 5, 6, 8, 10, 99]),
    ("turnover_cool", "换手降温", [0, 0.8, 1.0, 1.3, 1.8, 99]),
    ("mom5", "5日动量%", [-99, -5, 0, 5, 10, 20, 99]),
    ("mom20", "20日动量%", [-99, -10, 0, 10, 20, 35, 50, 999]),
    ("mom60", "60日动量%", [-99, -20, 0, 20, 40, 80, 999]),
    ("bias20", "20日乖离%", [-99, -5, 0, 5, 10, 20, 99]),
    ("close_pos", "收盘位置", [0, 0.3, 0.5, 0.7, 0.85, 0.95, 1.01]),
    ("amp_pct", "振幅%", [0, 3, 5, 7, 10, 99]),
    ("up_streak", "连涨天数", [0, 1, 2, 3, 4, 6, 99]),
    ("lu5", "5日涨停数", [0, 1, 2, 3, 99]),
    ("ma20_slope", "MA20斜率%", [-99, -2, 0, 2, 5, 99]),
    ("ivol20", "特质波动", [-99, -1, 0, 1, 2, 99]),
    ("excess", "当日超额%", [-99, -3, 0, 2, 4, 99]),
]


def tstat(x: pd.Series) -> float:
    x = x.dropna()
    if len(x) < 30:
        return 0.0
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))


def main() -> None:
    d = pd.read_parquet(OUT / "panel_right_205.parquet")
    # 复权异常清洗
    bad = (d["ret5"].abs() > 60) | (d["gap"].abs() > 15) | (d["pct"].abs() > 11.6)
    print(f"剔除异常 {int(bad.sum()):,} / {len(d):,}")
    d = d[~bad].reset_index(drop=True)
    d["year"] = d["d"].astype(str).str[:4].astype(int)
    print(f"样本 {len(d):,}  日 {d['d'].nunique():,}  "
          f"{d['d'].min()} → {d['d'].max()}  票 {d['ts_code'].nunique():,}")
    d["net5"] = d["ret5"] - COST
    d["early"] = (d["year"] <= 2017).astype(int)

    rows, summary = [], {}
    for col, label, bins in SPECS:
        x = d[[col, "d", "net5", "year", "early"]].dropna(subset=[col]).copy()
        if not len(x):
            continue
        x["g"] = pd.cut(x[col], bins, right=False)
        tot = x.groupby("g", observed=True)["net5"].mean()
        for g, sub in x.groupby("g", observed=True):
            dm = sub.groupby("d")["net5"].mean()
            yr = sub.groupby("year")["net5"].mean()
            rows.append({
                "factor": col, "label": label, "lo": float(g.left), "hi": float(g.right),
                "n": int(len(sub)),
                "per_day": round(len(sub) / max(1, sub["d"].nunique()), 1),
                "net": round(float(sub["net5"].mean()), 3),
                "t_day": round(tstat(dm), 2),
                "win": round(float((sub["net5"] > 0).mean() * 100), 1),
                "early": round(float(sub[sub.early == 1]["net5"].mean()), 3),
                "late": round(float(sub[sub.early == 0]["net5"].mean()), 3),
                "pos_years": f"{int((yr > 0).sum())}/{len(yr)}",
                "excess_vs_all": round(float(sub["net5"].mean() - tot.mean()), 3),
            })
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "scan_right_205.csv", index=False, encoding="utf-8")
    print(res.to_string(index=False, float_format=lambda v: f"{v:g}"))

    # ---- 持有期: 全池 vs 几个候选切片 ----
    hold = {}
    for h in (1, 3, 5, 10):
        d[f"net{h}"] = d[f"ret{h}"] - COST
        hold[str(h)] = round(float(d[f"net{h}"].mean()), 3)
    print("\n全池各持有期净收益%:", hold)

    # ---- 每日 top1: 不同排序键(引擎口径) ----
    keys = {
        "pct(追最强)": ("pct", False), "vr(追最放量)": ("vr", False),
        "dist_ph20(追最贴前高)": ("dist_ph20", False),
        "mom20(追最强趋势)": ("mom20", False),
        "excess(追最抗跌后转强)": ("excess", False),
        "ivol20(追最稳)": ("ivol20", True),
    }
    rows2 = []
    for name, (col, asc) in keys.items():
        x = d[["d", col, "net5", "year"]].dropna()
        idx = x.groupby("d")[col].idxmin() if asc else x.groupby("d")[col].idxmax()
        pick = x.loc[idx]
        dm = pick.groupby("d")["net5"].mean()
        rows2.append({"key": name, "n": int(len(pick)),
                      "net": round(float(pick["net5"].mean()), 3),
                      "t_day": round(tstat(dm), 2),
                      "win": round(float((pick["net5"] > 0).mean() * 100), 1),
                      "pos_years": f"{int((pick.groupby('year')['net5'].mean() > 0).sum())}/"
                                   f"{pick['year'].nunique()}"})
    pd.DataFrame(rows2).to_csv(OUT / "scan_right_top1_205.csv", index=False, encoding="utf-8")
    print("\n每日 top1:")
    print(pd.DataFrame(rows2).to_string(index=False, float_format=lambda v: f"{v:g}"))

    json.dump({"hold": hold, "n": int(len(d)), "days": int(d["d"].nunique())},
              open(OUT / "scan_right_205.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
