"""第231轮 D · 买入端限价承接（挂更低的价格等它来）

前面三批（分散度、排序口径）都是负的。这一批换个物理量：**买入价**。

现行执行的成交结构是"9:15–9:24 挂前收价，低开则集合竞价按开盘价成交，滑点 0"
（227 轮落地，+20.87pp）。它的代价是：不管开盘跌多少都照单接。
229 轮 D 批已证明**低开深度与收益无关**（五档 1.14~1.35 平坦）——
既然"跌 5% 和跌 1% 买进去结果一样"，那就没有理由在跌 1% 时也按原价买。

于是只剩一个问法：**能不能只接更便宜的价**。
机制（第八十轮已有，默认关闭）：
  gap <= 承接线 → 开盘价照买（深低开本身就是折扣，不要白不要）
  gap >  承接线 → 在 承接线 挂全天限价单，当日最低价触及才成交（成交价确定、更便宜），否则当日空仓

代价是笔数与在场率。所以这批不是"改参数"，而是"用笔数换单笔"的汇率测试：
  单笔 +2pp、笔数 −40% 到底划不划算，只能实测。
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

OUT = ROOT / "output" / "w231d_limit.csv"


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


PLAN = [
    ("现行(开盘价照买)", {}),
    ("承接-1.5%", {"buy_gap_below": -1.5, "buy_limit_fill": 1, "buy_limit_open_max": 0.0}),
    ("承接-2.0%", {"buy_gap_below": -2.0, "buy_limit_fill": 1, "buy_limit_open_max": 0.0}),
    ("承接-2.5%", {"buy_gap_below": -2.5, "buy_limit_fill": 1, "buy_limit_open_max": 0.0}),
    ("承接-3.0%", {"buy_gap_below": -3.0, "buy_limit_fill": 1, "buy_limit_open_max": 0.0}),
    ("硬门槛-2.0%(不许等, 直接拒)", {"buy_gap_below": -2.0, "buy_limit_fill": 0}),
]


def main() -> None:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 2500
    cfg = load_config()
    ctx = prepare_context(cfg, days)
    print(f"{len(ctx['main_dates'])} 日\n", flush=True)

    rows = []
    base = None
    for name, patch in PLAN:
        c = copy.deepcopy(cfg)
        c["execution"].update(patch)
        t = time.time()
        r = brief(run_backtest(c, None, ctx))
        if base is None:
            base = r
        rows.append({"方案": name, **r,
                     "Δ年化": round(r["年化"] - base["年化"], 2),
                     "Δ回撤": round(r["回撤"] - base["回撤"], 2),
                     "Δ卡玛": round(r["卡玛"] - base["卡玛"], 2),
                     "Δ单笔": round(r["单笔"] - base["单笔"], 3),
                     "Δ笔数": r["笔数"] - base["笔数"],
                     "耗时s": round(time.time() - t, 1)})
        print(f"{name:26s} 年化{r['年化']:7.2f}% 回撤{r['回撤']:7.2f}% 卡玛{r['卡玛']:5.2f} "
              f"{r['笔数']:4d}笔 单笔{r['单笔']:6.3f}% 三段{r['s1_年化']:6.1f}/{r['s2_年化']:6.1f}/"
              f"{r['s3_年化']:6.1f} | Δ年化{rows[-1]['Δ年化']:+7.2f} Δ单笔{rows[-1]['Δ单笔']:+6.3f}",
              flush=True)
        pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"\n已写 {OUT}", flush=True)


if __name__ == "__main__":
    main()
