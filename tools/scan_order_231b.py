"""第231轮 B 部分 · 买哪一只（排序口径重扫）

A 部分（分散度）结论：拆成 N 份 → 年化 84→39/27/20/16%，回撤 −18.6→−12.4/−10.0/−7.5/−6.0%，
卡玛 4.52→3.18/2.75/2.74/2.77。**分散全面为负**。

原因算得很清楚：单笔只掉 24%（1.252%→0.951%），但仓位掉 50%。
  单吊 1.01252^56 = 2.00 倍/年；分散 1.004755^72 = 1.41 倍/年。
  分散买到的是"更平滑"，付出去的是"复利速度"，交换比不划算。

既然"买几只"的答案是 1，本轮把同一决策点的另一半问完：**这 1 只怎么选**。
230 轮已证明 score 在 1.0 以上与收益零相关（+0.013），那"按 score 排第一"
可能只是历史遗留，不是最优。缺口是 T+1 集合竞价结束即可见的真实成交信息，
且 178 轮横截面 IC 显示缺口单调有效。这里在 2500 日窗口把排序口径重扫一遍。
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

from src.backtest import prepare_context, run_backtest  # noqa: E402
from src.config import load_config  # noqa: E402

OUT = ROOT / "output" / "w231b_order.csv"


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
    ("score（现行）", {}, "score"),
    ("gap 最深优先", {"pick_order": "gap"}, "gap"),
    ("band[-3,-1]（默认）", {"pick_order": "gap_band", "pick_gap_band": [-3.0, -1.0]}, "band"),
    ("band[-4,-1.5]", {"pick_order": "gap_band", "pick_gap_band": [-4.0, -1.5]}, "band"),
    ("band[-2.5,-0.5]", {"pick_order": "gap_band", "pick_gap_band": [-2.5, -0.5]}, "band"),
    ("band[-5,-2]", {"pick_order": "gap_band", "pick_gap_band": [-5.0, -2.0]}, "band"),
]


def main() -> None:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 2500
    os.environ.pop("BT_PICK_ORDER", None)
    os.environ.pop("BT_GAP_BAND", None)
    cfg = load_config()
    ctx = prepare_context(cfg, days)
    print(f"{len(ctx['main_dates'])} 日\n", flush=True)

    rows = []
    base = None
    for name, patch, kind in PLAN:
        c = copy.deepcopy(cfg)
        c["execution"].update(patch)
        t = time.time()
        r = brief(run_backtest(c, None, ctx))
        if base is None:
            base = r
        rows.append({"方案": name, "口径": kind, **r,
                     "Δ年化": round(r["年化"] - base["年化"], 2),
                     "Δ回撤": round(r["回撤"] - base["回撤"], 2),
                     "Δ卡玛": round(r["卡玛"] - base["卡玛"], 2),
                     "Δ单笔": round(r["单笔"] - base["单笔"], 3),
                     "Δ笔数": r["笔数"] - base["笔数"],
                     "耗时s": round(time.time() - t, 1)})
        print(f"{name:22s} 年化{r['年化']:7.2f}% 回撤{r['回撤']:7.2f}% 卡玛{r['卡玛']:5.2f} "
              f"{r['笔数']:4d}笔 单笔{r['单笔']:6.3f}% | Δ年化{rows[-1]['Δ年化']:+7.2f}", flush=True)
        pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"\n已写 {OUT}", flush=True)


if __name__ == "__main__":
    main()
