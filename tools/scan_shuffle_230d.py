"""第230轮 D · 决定性实验：把排序拿掉、把排序反过来，收益会怎样？

B 批的发现：score 与单笔收益的相关系数只有 +0.013，五档 1.12/1.05/1.34/1.50/1.26 无单调。
如果评分真的没信息，那么：
  D1 把权重全设为 0（等价于不看评分，按原始顺序取第 1 只）→ 收益应基本不变；
  D2 把权重取反（越差越优先）→ 收益也不该显著变差。
两个都成立，才能说"排序零贡献、alpha 全在条件筛选"。

注意：权重全 0 时 score 恒为 0，必须同时关掉 min_top1_score（否则全被门槛拦成空仓）。
"""
from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context, run_backtest  # noqa: E402
from src.config import load_config  # noqa: E402

OUT = ROOT / "output" / "w230d_shuffle.csv"


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
    return {"年化": round(float(m["年化收益率%"]), 2),
            "回撤": round(float(m["最大回撤%"]), 2),
            "卡玛": round(float(m["卡玛比率"] or 0), 2),
            "笔数": int(m["交易次数"]),
            "胜率": round(float(m["胜率%"]), 1),
            "单笔": round(float(m["平均单笔收益%"]), 3), **seg}


def main() -> None:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 2500
    cfg = load_config()
    ctx = prepare_context(cfg, days)
    print(f"{len(ctx['main_dates'])} 日\n", flush=True)

    base = brief(run_backtest(cfg, None, ctx))
    print(f"基准     年化{base['年化']:.2f}% 回撤{base['回撤']:.2f}% 卡玛{base['卡玛']:.2f} "
          f"{base['笔数']}笔 单笔{base['单笔']:.3f}%\n", flush=True)
    rows = [{"方案": "基准(按评分)", **base}]

    # D1 权重全 0
    c1 = copy.deepcopy(cfg)
    c1["selection"]["dip_weights"] = {k: 0.0 for k in c1["selection"]["dip_weights"]}
    c1["execution"]["min_top1_score"] = None
    c1["execution"]["score_tiers"] = []
    t = time.time()
    r1 = brief(run_backtest(c1, None, ctx))
    rows.append({"方案": "D1 不看评分", **r1, "Δ年化": round(r1["年化"] - base["年化"], 2),
                 "Δ卡玛": round(r1["卡玛"] - base["卡玛"], 2),
                 "Δ单笔": round(r1["单笔"] - base["单笔"], 3)})
    pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"D1 不看评分 年化{r1['年化']:7.2f}%({r1['年化']-base['年化']:+6.2f}) "
          f"卡玛{r1['卡玛']:5.2f} 单笔{r1['单笔']:.3f}% {r1['笔数']}笔 "
          f"三段{r1['s1_年化']:.0f}/{r1['s2_年化']:.0f}/{r1['s3_年化']:.0f} {time.time()-t:.0f}s", flush=True)

    # D2 权重取反
    c2 = copy.deepcopy(cfg)
    c2["selection"]["dip_weights"] = {k: -float(v) for k, v in c2["selection"]["dip_weights"].items()}
    c2["execution"]["min_top1_score"] = None
    c2["execution"]["score_tiers"] = []
    t = time.time()
    r2 = brief(run_backtest(c2, None, ctx))
    rows.append({"方案": "D2 评分取反", **r2, "Δ年化": round(r2["年化"] - base["年化"], 2),
                 "Δ卡玛": round(r2["卡玛"] - base["卡玛"], 2),
                 "Δ单笔": round(r2["单笔"] - base["单笔"], 3)})
    pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"D2 评分取反 年化{r2['年化']:7.2f}%({r2['年化']-base['年化']:+6.2f}) "
          f"卡玛{r2['卡玛']:5.2f} 单笔{r2['单笔']:.3f}% {r2['笔数']}笔 "
          f"三段{r2['s1_年化']:.0f}/{r2['s2_年化']:.0f}/{r2['s3_年化']:.0f} {time.time()-t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
