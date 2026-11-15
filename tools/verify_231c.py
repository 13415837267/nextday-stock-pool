"""第231轮 C · 排序口径的全史复验（4201 日）

2500 日窗口：band[-3,-1] 年化 90.74%(+6.61) 回撤 −17.21%(+1.41) 卡玛 5.27(+0.75) 笔数 559(不变)，
三段 19.92/167.20/98.10 对基准 10.59/167.32/93.81 —— 前段+9.3、中段持平、最近段+4.3，
没有出现 229 轮 close_pos 那种"最近段掉 10pp"的形态。

按既定规矩，任何候选必须过全史（17 年）才算数。
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

OUT = ROOT / "output" / "w231c_fullhist.csv"


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
    ("score（现行）", {}),
    ("band[-3,-1]", {"pick_order": "gap_band", "pick_gap_band": [-3.0, -1.0]}),
    ("gap 最深优先", {"pick_order": "gap"}),
]


def main() -> None:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 4201
    os.environ.pop("BT_PICK_ORDER", None)
    os.environ.pop("BT_GAP_BAND", None)
    cfg = load_config()
    t0 = time.time()
    ctx = prepare_context(cfg, days)
    print(f"{len(ctx['main_dates'])} 日, ctx {time.time()-t0:.0f}s\n", flush=True)

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
        print(f"{name:16s} 年化{r['年化']:7.2f}% 回撤{r['回撤']:7.2f}% 卡玛{r['卡玛']:5.2f} "
              f"{r['笔数']:4d}笔 单笔{r['单笔']:6.3f}% 三段{r['s1_年化']:6.1f}/{r['s2_年化']:6.1f}/"
              f"{r['s3_年化']:6.1f} | Δ年化{rows[-1]['Δ年化']:+7.2f} Δ卡玛{rows[-1]['Δ卡玛']:+5.2f}",
              flush=True)
        pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"\n已写 {OUT}", flush=True)


if __name__ == "__main__":
    main()
