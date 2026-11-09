"""第一百七十八轮: 评分有没有排序能力? —— 主策略横截面 IC 体检。

背景
----
用户要求"每日盘后给出评分最高的一个股票作明日计划"。系统的做法是:
T 日收盘按 dip 评分排序, 取第 1 名 (exec_top_n=1), T+1 开盘买 T+2 收盘卖。
这个设计隐藏了一个从未被检验的前提 —— **评分高 = 后续收益高**。
如果横截面 Spearman IC ≈ 0, 那"评分最高"就是在合格池里随机抽一只,
前 177 轮所有基于评分的微调(权重/门槛/分档)都在优化一个无效的排序。

做法
----
对每个信号日, 取**全部**达标候选(不只 top1), 各算 T+1 开盘 → T+2 收盘 的净收益,
然后:
  1) 逐日横截面 Spearman(score, ret) → 日度 IC 序列, 看均值/t 值/正比例
  2) 按当日评分名次分档(1,2,3,...,5+)看单笔收益是否单调
  3) 按评分分位分档(前 10% / 10-25% / ...) 看是否单调
  4) 在同一批候选里对比 "永远买第 1 名" vs "随机买一只" 的期望差

输出: output/exp_score_ic.json (全量明细) + 控制台表格
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

COST = (2.5 * 2 + 5 + 10) / 1e4          # 与回测同口径: 双边佣金 + 印花税 + 滑点


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=1500)
    ap.add_argument("--top", type=int, default=200)
    ap.add_argument("--out", default="output/exp_score_ic.json")
    a = ap.parse_args()

    cfg = load_config()
    ctx = prepare_context(cfg, a.days)
    by_date, dates = ctx["by_date"], ctx["main_dates"]
    pos = {d: i for i, d in enumerate(dates)}

    names: dict[str, str] = {}
    try:
        from src.data_api import load_names  # 可选
        names = load_names() or {}
    except Exception:
        pass

    rows: list[dict] = []
    for d in dates:
        i = pos[d]
        if i + 2 >= len(dates):
            continue
        bd, sd = dates[i + 1], dates[i + 2]
        picks, _st = select_stocks(d, by_date[d], cfg, names, top_n=a.top, name_filter=False)
        if not picks:
            continue
        picks = [p for p in picks if p.get("pool", "main") == "main"]
        if not picks:
            continue
        dfb, dfs = by_date[bd], by_date[sd]
        for k, p in enumerate(picks, 1):
            code = p["code"]
            mb = dfb[dfb["ts_code"] == code]
            ms = dfs[dfs["ts_code"] == code]
            if mb.empty or ms.empty:
                continue
            rb, rs = mb.iloc[0], ms.iloc[0]
            o, c = float(rb["open_adj"]), float(rs["close_adj"])
            if o <= 0 or c <= 0:
                continue
            o_raw = float(rb["open"])
            if o_raw >= float(rb["high_limit"]) - 1e-9 or o_raw <= float(rb["low_limit"]) + 1e-9:
                continue          # 一字板买不到, 不进 IC 样本
            ret = (c / o - 1.0 - COST) * 100.0
            rows.append({"d": d, "code": code, "rank": k, "n_pool": len(picks),
                         "score": float(p["score"]), "ret": ret})

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("无样本")

    def stat(v: np.ndarray) -> str:
        if len(v) < 2:
            return f"n={len(v)}"
        t = v.mean() / (v.std(ddof=1) / np.sqrt(len(v)))
        return (f"n={len(v):5d} 均值={v.mean():+.3f}% 中位={np.median(v):+.3f}% "
                f"胜率={(v > 0).mean() * 100:4.1f}% t={t:+5.2f}")

    print(f"样本 {len(df)} 条, 覆盖 {df.d.nunique()} 个信号日, "
          f"日均候选 {len(df) / df.d.nunique():.1f} 只")

    # ---- 1. 日度横截面 IC ----
    ics, ics_k = [], []
    for _, g in df.groupby("d"):
        if len(g) < 5:
            continue
        ic = g["score"].corr(g["ret"], method="spearman")
        if pd.notna(ic):
            ics.append(ic)
        # 名次与收益的 Kendall 更稳, 但用 Spearman(-rank) 同号
        ic2 = (-g["rank"]).corr(g["ret"], method="spearman")
        if pd.notna(ic2):
            ics_k.append(ic2)
    ics = np.array(ics)
    print(f"\n=== 日度横截面 Spearman IC (score vs T+1开→T+2收 净收益) ===")
    print(f"有效日 {len(ics)}  均值 {ics.mean():+.4f}  中位 {np.median(ics):+.4f}  "
          f"标准差 {ics.std(ddof=1):.4f}")
    print(f"t = {ics.mean() / (ics.std(ddof=1) / np.sqrt(len(ics))):+.2f}   "
          f"IC>0 比例 {(ics > 0).mean() * 100:.1f}%   "
          f"ICIR = {ics.mean() / ics.std(ddof=1):+.3f}")

    # ---- 2. 按当日名次 ----
    print("\n=== 按当日评分名次 ===")
    for r, g in df.groupby("rank"):
        if r > 10 or len(g) < 30:
            continue
        print(f"  第{int(r):>2}名 {stat(g.ret.to_numpy())}")
    tail = df[df["rank"] > 10]
    if len(tail) >= 30:
        print(f"  11名以后 {stat(tail.ret.to_numpy())}")

    # ---- 3. 按评分分位 ----
    print("\n=== 按当日评分分位(池内百分位) ===")
    df["pct_in_pool"] = df.groupby("d")["score"].rank(pct=True)
    for lo, hi in [(0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 0.9), (0.9, 1.01)]:
        s = df[(df.pct_in_pool > lo) & (df.pct_in_pool <= hi)]
        print(f"  [{lo:.0%},{hi:.0%}] {stat(s.ret.to_numpy())}")

    # ---- 4. 第1名 vs 随机 vs 池均 ----
    first = df[df["rank"] == 1]
    print("\n=== 第 1 名 vs 池内随机 ===")
    print(f"  第 1 名      {stat(first.ret.to_numpy())}")
    print(f"  全池         {stat(df.ret.to_numpy())}")
    print(f"  第 1 名 - 全池 = {first.ret.mean() - df.ret.mean():+.3f}pp")

    out = {"days": df.d.nunique(), "n": len(df),
           "ic": {"n": len(ics), "mean": round(float(ics.mean()), 4),
                  "t": round(float(ics.mean() / (ics.std(ddof=1) / np.sqrt(len(ics)))), 2),
                  "pos_pct": round(float((ics > 0).mean() * 100), 1),
                  "icir": round(float(ics.mean() / ics.std(ddof=1)), 3)},
           "by_rank": {int(r): {"n": int(len(g)), "mean": round(float(g.ret.mean()), 3)}
                       for r, g in df.groupby("rank") if len(g) >= 30}}
    p = ROOT / a.out
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    df.to_json(ROOT / a.out.replace(".json", "_rows.json"), orient="records", force_ascii=False)
    print(f"\n写出 {p}")


if __name__ == "__main__":
    main()
