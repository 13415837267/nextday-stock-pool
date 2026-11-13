"""第 220 轮诊断: 622% 是模型的本事, 还是"深跌 + 高波动"的暴露?"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"


def sim(days, pick_fn, day_index, day_dates):
    nav, peak, mdd, held, rows = 1.0, 1.0, 0.0, None, []
    for dd in days:
        d = int(dd["trade_date"].iloc[0])
        if held is not None and d <= held:
            continue
        top = pick_fn(dd)
        if top is None:
            continue
        r = float(top["ret"]) / 100
        nav *= (1 + r)
        peak = max(peak, nav)
        mdd = min(mdd, nav / peak - 1)
        rows.append((d, r * 100))
        held = day_dates[min(day_index[d] + 2, len(day_dates) - 1)]
    rets = [x[1] for x in rows]
    d0, d1 = pd.Timestamp(str(int(days[0]["trade_date"].iloc[0]))), \
        pd.Timestamp(str(int(days[-1]["trade_date"].iloc[0])))
    yrs = (d1 - d0).days / 365.25
    return {"笔数": len(rets), "年化%": round((nav ** (1 / yrs) - 1) * 100, 1),
            "总倍": round(nav, 1), "回撤%": round(mdd * 100, 1),
            "单笔%": round(float(np.mean(rets)), 3), "中位%": round(float(np.median(rets)), 3),
            "胜率%": round(float(np.mean([x > 0 for x in rets]) * 100), 1),
            "最好一笔%": round(float(np.max(rets)), 1),
            "最差一笔%": round(float(np.min(rets)), 1),
            "年度": {str(y): round(float(((1 + np.array([v / 100 for dd_, v in rows if str(dd_)[:4] == str(y)])).prod() - 1) * 100), 1)
                     for y in sorted({str(x[0])[:4] for x in rows})}}, rets


def main():
    p = pd.read_parquet(OUT / "ml_220_pred.parquet")
    # 补特征回来做单规则对照
    f = pd.read_parquet(OUT / "ml_220.parquet",
                        columns=["trade_date", "ts_code", "mom60", "amt_yi", "dist_high250", "vol20"])
    p = p.merge(f, on=["trade_date", "ts_code"], how="left")

    day_dates = sorted(p["trade_date"].unique())
    day_index = {x: i for i, x in enumerate(day_dates)}
    days = [g for _, g in p.groupby("trade_date", sort=True)]

    res = {}
    res["ML_top1"], ml_rets = sim(days, lambda dd: dd.sort_values("pred", ascending=False).iloc[0],
                                  day_index, day_dates)
    res["规则_mom60最低"], _ = sim(days, lambda dd: dd.sort_values("mom60").iloc[0],
                                 day_index, day_dates)
    res["规则_成交额最小"], _ = sim(days, lambda dd: dd.sort_values("amt_yi").iloc[0],
                                day_index, day_dates)
    res["规则_距250日高点最远"], _ = sim(days, lambda dd: dd.sort_values("dist_high250").iloc[0],
                                    day_index, day_dates)
    res["规则_波动最高"], _ = sim(days, lambda dd: dd.sort_values("vol20", ascending=False).iloc[0],
                              day_index, day_dates)

    print(f"{'策略':<20}{'笔数':>6}{'年化%':>9}{'总倍':>12}{'回撤%':>8}{'单笔%':>8}{'中位%':>8}{'胜率%':>8}")
    for k, v in res.items():
        print(f"{k:<20}{v['笔数']:>6}{v['年化%']:>9.1f}{v['总倍']:>12.1f}"
              f"{v['回撤%']:>8.1f}{v['单笔%']:>8.3f}{v['中位%']:>8.3f}{v['胜率%']:>8.1f}")

    print("\nML_top1 年度收益:")
    print(res["ML_top1"]["年度"])

    # 分布
    r = np.array(ml_rets)
    print(f"\nML 单笔分布: n={len(r)} 均值{r.mean():.3f} 中位{np.median(r):.3f} "
          f"P10{np.percentile(r,10):.2f} P90{np.percentile(r,90):.2f} 最大{r.max():.1f} 最小{r.min():.1f}")
    print(f"  去掉最好 5% 后的均值 {np.sort(r)[:-int(len(r)*0.05)].mean():.3f}")
    print(f"  正收益贡献总和 {r[r>0].sum():.0f}pp / 负收益 {r[r<0].sum():.0f}pp")

    (OUT / "diag_ml_220.json").write_text(json.dumps(
        {k: {kk: vv for kk, vv in v.items() if kk != "年度"} | {"年度": v["年度"]}
         for k, v in res.items()}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved")


if __name__ == "__main__":
    main()
