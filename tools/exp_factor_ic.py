"""第一百七十八轮: 因子 IC 体检 —— 找出到底哪个维度有排序能力。

上文 exp_score_ic 的初步结论: 现有 dip 评分与次日收益的横截面 IC ≈ 0 (t=+0.28),
第 1 名甚至低于全池均值。但那个样本是"全部达标候选", 包含大量**缺口不达标、
本来就不会买**的票。真正要问的是: 在**实际可买**的那批里(缺口达标), 有没有
任何一个维度能把好票和坏票分开?

做法
----
对每个信号日: 取全部达标候选, 记录
  · 现有 score 及其 5 个分项原始值(mom20/bias20/amount/vol_ratio/low_tail)
  · 其他候选因子(close_pos/pct/turnover_rate/ivol20/dist_low20/dist_ma5/mom3/mom5)
  · T+1 开盘缺口 gap
  · T+1开→T+2收 净收益 ret
然后分两个样本各算日度横截面 IC:
  A. 全池
  B. 只保留 gap <= 0 (实际可买)
C. 在 B 里按名次分档, 检验"评分第1名"是否优于"随机一只"

输出: output/exp_factor_ic.json + output/exp_factor_ic_rows.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context   # noqa: E402
from src.config import load_config        # noqa: E402
from src.selector import select_stocks    # noqa: E402

COST = (2.5 * 2 + 5 + 10) / 1e4

FACTORS = ["mom20", "bias20", "amount", "vol_ratio", "low_tail", "close_pos",
           "pct", "turnover_rate", "ivol20", "dist_low20", "dist_ma5", "mom3", "mom5"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=1500)
    ap.add_argument("--top", type=int, default=200)
    ap.add_argument("--out", default="output/exp_factor_ic.json")
    a = ap.parse_args()

    cfg = load_config()
    ctx = prepare_context(cfg, a.days)
    by_date, dates = ctx["by_date"], ctx["main_dates"]
    pos = {d: i for i, d in enumerate(dates)}

    rows: list[dict] = []
    for d in dates:
        i = pos[d]
        if i + 2 >= len(dates):
            continue
        bd, sd = dates[i + 1], dates[i + 2]
        picks, _st = select_stocks(d, by_date[d], cfg, None, top_n=a.top, name_filter=False)
        picks = [p for p in picks if p.get("pool", "main") == "main"]
        if not picks:
            continue
        dfb, dfs = by_date[bd], by_date[sd]
        # 当日原始行: 取因子原值(select_stocks 已带部分)
        day = by_date[d].set_index("ts_code")
        for k, p in enumerate(picks, 1):
            code = p["code"]
            mb = dfb[dfb["ts_code"] == code]
            ms = dfs[dfs["ts_code"] == code]
            if mb.empty or ms.empty or code not in day.index:
                continue
            rb, rs, r0 = mb.iloc[0], ms.iloc[0], day.loc[code]
            o, c = float(rb["open_adj"]), float(rs["close_adj"])
            pc = float(rb["prev_close_adj"])
            if o <= 0 or c <= 0 or pc <= 0:
                continue
            o_raw = float(rb["open"])
            if o_raw >= float(rb["high_limit"]) - 1e-9 or o_raw <= float(rb["low_limit"]) + 1e-9:
                continue
            ret = (c / o - 1.0 - COST) * 100.0
            gap = (o / pc - 1.0) * 100.0
            rec = {"d": d, "code": code, "rank": k, "n_pool": len(picks),
                   "score": float(p["score"]), "gap": gap, "ret": ret}
            for f in FACTORS:
                try:
                    rec[f] = float(r0[f])
                except Exception:
                    rec[f] = np.nan
            rows.append(rec)

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("无样本")

    def ic_stats(g: pd.DataFrame, col: str) -> tuple[float, float, float, int]:
        ics = []
        for _, x in g.groupby("d"):
            if len(x) < 5:
                continue
            v = x[[col, "ret"]].dropna()
            if len(v) < 5:
                continue
            ic = v[col].corr(v["ret"], method="spearman")
            if pd.notna(ic):
                ics.append(ic)
        if len(ics) < 10:
            return (np.nan,) * 3 + (len(ics),)
        arr = np.array(ics)
        return (float(arr.mean()), float(arr.mean() / (arr.std(ddof=1) / np.sqrt(len(arr)))),
                float((arr > 0).mean() * 100), len(arr))

    def report(g: pd.DataFrame, title: str):
        print(f"\n=== {title} (n={len(g)}, 日={g.d.nunique()}) ===")
        print(f"{'因子':<14}{'IC均值':>9}{'t':>8}{'IC>0%':>8}{'日数':>6}")
        res = {}
        for col in ["score"] + FACTORS:
            m, t, p, n = ic_stats(g, col)
            if np.isnan(m):
                continue
            res[col] = {"ic": round(m, 4), "t": round(t, 2), "pos": round(p, 1), "days": n}
            flag = " *" if abs(t) >= 2.0 else ""
            print(f"{col:<14}{m:>+9.4f}{t:>+8.2f}{p:>7.1f}%{n:>6}{flag}")
        return res

    out = {"days": int(df.d.nunique()), "n": int(len(df))}
    out["all"] = report(df, "A. 全池")

    buyable = df[df["gap"] <= 0]
    out["buyable"] = report(buyable, "B. 实际可买 (gap<=0)")

    print("\n=== C. 可买样本内按评分名次 ===")
    for r, g in buyable.groupby("rank"):
        if r > 5 or len(g) < 50:
            continue
        v = g.ret.to_numpy()
        t = v.mean() / (v.std(ddof=1) / np.sqrt(len(v)))
        print(f"  第{int(r)}名 n={len(v):4d} 均值={v.mean():+.3f}% 中位={np.median(v):+.3f}% "
              f"胜率={(v > 0).mean()*100:4.1f}% t={t:+.2f}")
    print(f"  可买全池 n={len(buyable)} 均值={buyable.ret.mean():+.3f}%")

    # 缺口本身的预测力(可买样本内, gap 越深越好?)
    print("\n=== D. 可买样本内 按缺口分档 ===")
    for lo, hi in [(-99, -4), (-4, -3), (-3, -2), (-2, -1), (-1, -0.5), (-0.5, 0.001)]:
        s = buyable[(buyable.gap >= lo) & (buyable.gap < hi)]
        if len(s) < 30:
            continue
        v = s.ret.to_numpy()
        t = v.mean() / (v.std(ddof=1) / np.sqrt(len(v)))
        print(f"  [{lo:>4},{hi:>6}) n={len(v):4d} 均值={v.mean():+.3f}% 胜率={(v>0).mean()*100:4.1f}% t={t:+.2f}")

    p = ROOT / a.out
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    df.to_json(ROOT / a.out.replace(".json", "_rows.json"), orient="records", force_ascii=False)
    print(f"\n写出 {p}")


if __name__ == "__main__":
    main()
