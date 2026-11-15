"""第230轮 · 置信度门槛：第 1 名到底有多"确信"？

229 轮的结论是"任何换人的改动都伤收益"。但为什么换人就伤？因为**第 1 名经常是随机挑的**：
评分是横截面 z 分，池里几只票常常挤在一起，top1 与 top2 只差 0.03，那第 1 名和第 2 名没有实质差别，
条件一动、名次一换，收益自然跟着随机游走。

所以本轮不加"筛选条件"，加一个**置信度条件**：
    只在 (top1 评分 − top2 评分) ≥ margin 时才买，否则当天空仓。
这是唯一不改变"谁是最好的票"、只决定"这次选择值不值得下注"的门槛。

两件事一次跑完（ctx 只建一次）：
  1) 诊断：margin 分布 + margin 分箱后的实际收益（用基准回测的成交笔）
  2) 实测：min_top1_margin 取 P25/P50/P75 与固定 0.10/0.20
"""
from __future__ import annotations

import copy
import os
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["BT_FULL_TRADES"] = "1"

from src.backtest import prepare_context, run_backtest  # noqa: E402
from src.config import load_config  # noqa: E402

OUT = ROOT / "output" / "w230_margin.csv"


def run_one(cfg, ctx) -> dict:
    res = run_backtest(cfg, None, ctx)
    m = res["metrics"]
    eq = pd.DataFrame(res["equity"])
    nav = eq["nav"].astype(float).values
    n = len(nav)
    segs = {}
    for i, tag in enumerate(["s1", "s2", "s3"]):
        seg = nav[int(n * i / 3): int(n * (i + 1) / 3)]
        yrs = max(0.3, len(seg) / 244.0)
        ann = (seg[-1] / seg[0]) ** (1 / yrs) - 1 if len(seg) > 40 else 0.0
        mdd = float((seg / pd.Series(seg).cummax().values - 1).min()) if len(seg) > 40 else 0.0
        segs[f"{tag}_年化"] = round(ann * 100, 2)
    d = {"年化": round(float(m["年化收益率%"]), 2),
         "回撤": round(float(m["最大回撤%"]), 2),
         "卡玛": round(float(m["卡玛比率"] or 0), 2),
         "笔数": int(m["交易次数"]),
         "胜率": round(float(m["胜率%"]), 1),
         "单笔": round(float(m["平均单笔收益%"]), 3),
         "空仓天数": int(m.get("空仓天数") or 0)}
    d.update(segs)
    return d, res


def main() -> None:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 2500
    cfg = load_config()
    t0 = time.time()
    ctx = prepare_context(cfg, days)
    print(f"ctx {time.time()-t0:.0f}s, {len(ctx['main_dates'])} 日\n", flush=True)

    base, res = run_one(cfg, ctx)
    print(f"基准 年化{base['年化']:.2f}% 卡玛{base['卡玛']:.2f} 单笔{base['单笔']:.3f}% "
          f"{base['笔数']}笔 三段{base['s1_年化']:.0f}/{base['s2_年化']:.0f}/{base['s3_年化']:.0f}\n", flush=True)

    rows = [{"方案": "基准", "门槛": None, **base}]
    pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8-sig")

    # ---------- 诊断：成交笔的 margin 与收益 ----------
    ts = [t for t in res["trades_all"] if t.get("ret_pct") is not None and t.get("margin") is not None]
    df = pd.DataFrame([{"margin": t["margin"], "ret": t["ret_pct"], "pool": t.get("pool")} for t in ts])
    print(f"=== margin 分布（{len(df)} 笔成交）===")
    print(df["margin"].describe(percentiles=[.1, .25, .5, .75, .9]).round(3).to_string(), flush=True)
    q = pd.qcut(df["margin"], 5, duplicates="drop")
    g = df.groupby(q, observed=True)["ret"].agg(["count", "mean", lambda s: (s > 0).mean() * 100])
    g.columns = ["笔数", "平均单笔", "胜率"]
    print("\n=== margin 五分位 → 收益 ===")
    print(g.round(3).to_string(), flush=True)
    print(f"\nmargin 与收益相关系数: {df['margin'].corr(df['ret']):.3f}", flush=True)

    # ---------- 实测 ----------
    qs = df["margin"].quantile([.25, .5, .75]).round(2).tolist()
    cands = sorted({round(v, 2) for v in qs + [0.10, 0.20] if v > 0})
    print(f"\n待测门槛: {cands}\n", flush=True)

    for v in cands:
        c = copy.deepcopy(cfg)
        c["execution"]["min_top1_margin"] = v
        t = time.time()
        r, _ = run_one(c, ctx)
        row = {"方案": f"margin>={v}", "门槛": v, **r,
               "Δ年化": round(r["年化"] - base["年化"], 2),
               "Δ卡玛": round(r["卡玛"] - base["卡玛"], 2),
               "Δ单笔": round(r["单笔"] - base["单笔"], 3),
               "Δ笔数": r["笔数"] - base["笔数"]}
        rows.append(row)
        pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8-sig")
        print(f"margin>={v:<5} 年化{r['年化']:7.2f}%({row['Δ年化']:+6.2f}) 卡玛{r['卡玛']:5.2f}"
              f"({row['Δ卡玛']:+.2f}) 单笔{r['单笔']:.3f}%({row['Δ单笔']:+.3f}) "
              f"{r['笔数']:4d}笔 空仓{r['空仓天数']}日 {time.time()-t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
