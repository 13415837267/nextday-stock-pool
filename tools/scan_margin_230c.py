"""第230轮 C · 把 margin 从"买入门槛"改成"下注倍率"

B 批结论：margin 与收益负相关（−0.141，五档 1.68/1.92/1.28/0.88/0.51），
但做成买入门槛 → 笔数从 559 掉到 360/261，年化 −26 / −44pp。**笔数是命，砍不得**。

所以只剩一种用法：照常买，只把"领先幅度大 = 大概率是个股独跌"的那些笔降仓。
笔数不变、回撤该降、卡玛有可能升。

三组档位（升序匹配，第一条满足的生效）：
  A 温和:  ≤0.25→1.0 / ≤0.42→0.85 / 其余→0.7
  B 中性:  ≤0.25→1.0 / ≤0.42→0.7  / 其余→0.5
  C 激进:  ≤0.42→1.0 / 其余→0.5
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

OUT = ROOT / "output" / "w230c_marginpos.csv"

TIERS = {
    "A 温和": [{"max": 0.25, "mult": 1.0}, {"max": 0.42, "mult": 0.85}, {"max": 99.0, "mult": 0.70}],
    "B 中性": [{"max": 0.25, "mult": 1.0}, {"max": 0.42, "mult": 0.70}, {"max": 99.0, "mult": 0.50}],
    "C 激进": [{"max": 0.42, "mult": 1.0}, {"max": 99.0, "mult": 0.50}],
}


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
    print(f"基准 年化{base['年化']:.2f}% 回撤{base['回撤']:.2f}% 卡玛{base['卡玛']:.2f} "
          f"{base['笔数']}笔 单笔{base['单笔']:.3f}%\n", flush=True)
    rows = [{"方案": "基准", **base}]

    for name, tiers in TIERS.items():
        c = copy.deepcopy(cfg)
        c["execution"]["margin_tiers"] = tiers
        t = time.time()
        r = brief(run_backtest(c, None, ctx))
        row = {"方案": name, **r,
               "Δ年化": round(r["年化"] - base["年化"], 2),
               "Δ回撤": round(r["回撤"] - base["回撤"], 2),
               "Δ卡玛": round(r["卡玛"] - base["卡玛"], 2),
               "Δ单笔": round(r["单笔"] - base["单笔"], 3),
               "Δ笔数": r["笔数"] - base["笔数"]}
        rows.append(row)
        pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8-sig")
        print(f"{name:<8} 年化{r['年化']:7.2f}%({row['Δ年化']:+6.2f}) 回撤{r['回撤']:7.2f}%"
              f"({row['Δ回撤']:+.2f}) 卡玛{r['卡玛']:5.2f}({row['Δ卡玛']:+.2f}) "
              f"{r['笔数']:4d}笔 单笔{r['单笔']:.3f}% 三段{r['s1_年化']:.0f}/{r['s2_年化']:.0f}/{r['s3_年化']:.0f} "
              f"{time.time()-t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
