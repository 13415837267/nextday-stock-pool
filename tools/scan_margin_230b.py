"""第230轮 B · margin 诊断加深 + 领先幅度上限实测

A 批结果反直觉但很清楚：margin（top1−top2 评分差）与收益**负相关 −0.141**
  0~0.115 → 1.680% / 0.115~0.249 → 1.917% / 0.249~0.421 → 1.280% /
  0.421~0.676 → 0.878% / 0.676~1.957 → 0.512%（胜率 50%）
即"第 1 名越鹤立鸡群，越不赚钱"。机制假说：margin 大 = 该票在某维度极端 = 个股独跌，
而策略赚的是"全市场同跌后的普遍性修复"。

本批要回答三件事：
  1) 把 559 笔的 score / margin / pool_n 一起落盘，看三者谁在解释收益（避免误判）；
  2) 反过来测**上限** max_top1_margin：只在"第 1 名不突出"的日子交易；
  3) 顺带看 pool_n（池深）是不是同一个东西的另一种写法。
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

OUT = ROOT / "output" / "w230b_margin.csv"


def brief(res: dict) -> dict:
    m = res["metrics"]
    nav = pd.DataFrame(res["equity"])["nav"].astype(float).values
    n = len(nav)
    seg = {}
    for i, tag in enumerate(["s1", "s2", "s3"]):
        s = nav[int(n * i / 3): int(n * (i + 1) / 3)]
        yrs = max(0.3, len(s) / 244.0)
        ann = (s[-1] / s[0]) ** (1 / yrs) - 1 if len(s) > 40 else 0.0
        mdd = float((s / pd.Series(s).cummax().values - 1).min()) if len(s) > 40 else 0.0
        seg[f"{tag}_年化"] = round(ann * 100, 2)
        seg[f"{tag}_卡玛"] = round(ann / abs(mdd) if mdd else 0, 2)
    return {"年化": round(float(m["年化收益率%"]), 2),
            "回撤": round(float(m["最大回撤%"]), 2),
            "卡玛": round(float(m["卡玛比率"] or 0), 2),
            "笔数": int(m["交易次数"]),
            "胜率": round(float(m["胜率%"]), 1),
            "单笔": round(float(m["平均单笔收益%"]), 3),
            **seg}


def main() -> None:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 2500
    cfg = load_config()
    ctx = prepare_context(cfg, days)
    print(f"{len(ctx['main_dates'])} 日\n", flush=True)

    res = run_backtest(cfg, None, ctx)
    base = brief(res)
    print(f"基准 年化{base['年化']:.2f}% 卡玛{base['卡玛']:.2f} 单笔{base['单笔']:.3f}% {base['笔数']}笔\n",
          flush=True)

    ts = [t for t in res["trades_all"] if t.get("ret_pct") is not None]
    df = pd.DataFrame([{"date": t["signal_date"], "pool": t.get("pool"), "ret": t["ret_pct"],
                        "score": t.get("score"), "margin": t.get("margin"),
                        "pool_n": t.get("pool_n")} for t in ts])
    df.to_csv(ROOT / "output" / "w230_trades.csv", index=False, encoding="utf-8-sig")
    print(f"落盘 {len(df)} 笔\n", flush=True)

    for col in ["score", "margin", "pool_n"]:
        d = df.dropna(subset=[col])
        if not len(d):
            continue
        print(f"=== {col} 五分位（相关 {d[col].corr(d['ret']):+.3f}）===")
        q = pd.qcut(d[col], 5, duplicates="drop")
        g = d.groupby(q, observed=True)["ret"].agg(["count", "mean", lambda s: (s > 0).mean() * 100])
        g.columns = ["笔数", "平均单笔", "胜率"]
        print(g.round(3).to_string(), "\n", flush=True)

    # margin 与 pool_n 是否同一件事
    d = df.dropna(subset=["margin", "pool_n"])
    print(f"margin 与 pool_n 相关: {d['margin'].corr(d['pool_n']):+.3f}", flush=True)
    print(f"margin 与 score 相关: {d['margin'].corr(d['score']):+.3f}\n", flush=True)

    rows = [{"方案": "基准", "门槛": None, **base}]
    pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8-sig")
    for v in [0.25, 0.42, 0.60]:
        c = copy.deepcopy(cfg)
        c["execution"]["max_top1_margin"] = v
        t = time.time()
        r = brief(run_backtest(c, None, ctx))
        row = {"方案": f"margin<={v}", "门槛": v, **r,
               "Δ年化": round(r["年化"] - base["年化"], 2),
               "Δ卡玛": round(r["卡玛"] - base["卡玛"], 2),
               "Δ单笔": round(r["单笔"] - base["单笔"], 3),
               "Δ笔数": r["笔数"] - base["笔数"]}
        rows.append(row)
        pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8-sig")
        print(f"margin<={v:<5} 年化{r['年化']:7.2f}%({row['Δ年化']:+6.2f}) 卡玛{r['卡玛']:5.2f}"
              f"({row['Δ卡玛']:+.2f}) 单笔{r['单笔']:.3f}%({row['Δ单笔']:+.3f}) {r['笔数']:4d}笔 "
              f"三段{r['s1_年化']:.0f}/{r['s2_年化']:.0f}/{r['s3_年化']:.0f} {time.time()-t:.0f}s",
              flush=True)


if __name__ == "__main__":
    main()
