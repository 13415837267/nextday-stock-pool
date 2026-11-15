"""第230轮 E · 评分门槛重扫（唯一没被扫过的"条件"）

D 批拆出一个关键事实：
  · 不看评分（权重全 0 + 关门槛）→ 年化 36.8%（−47pp）、单笔 0.426%、766 笔
  · 评分取反（同样关门槛）→ 51.5%（−33pp）、单笔 0.721%、808 笔
  · 基准（权重正常 + 门槛 1.0）→ 84.1%、单笔 1.252%、559 笔
说明评分系统整体是有价值的，但"门槛 1.0"和"1.0 以上的排序"各贡献多少还没分开。
而 1.0 这个值来自第九十五轮，此后 130 多轮没再动过 —— 它是唯一没被重扫的条件。

本批扫 min_top1_score：None(关) / 1.1 / 1.2 / 1.3 / 1.5 / 1.8，一次 ctx 跑完。
判据同 229：年化↑、卡玛↑、单笔↑、三段不塌。
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

OUT = ROOT / "output" / "w230e_scoregate.csv"


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
    print(f"基准(1.0) 年化{base['年化']:.2f}% 回撤{base['回撤']:.2f}% 卡玛{base['卡玛']:.2f} "
          f"{base['笔数']}笔 单笔{base['单笔']:.3f}%\n", flush=True)
    rows = [{"方案": "1.0(现行)", "门槛": 1.0, **base}]
    pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8-sig")

    for v in [None, 1.1, 1.2, 1.3, 1.5, 1.8]:
        c = copy.deepcopy(cfg)
        c["execution"]["min_top1_score"] = v
        t = time.time()
        r = brief(run_backtest(c, None, ctx))
        row = {"方案": ("关门槛" if v is None else f"{v}"), "门槛": v, **r,
               "Δ年化": round(r["年化"] - base["年化"], 2),
               "Δ卡玛": round(r["卡玛"] - base["卡玛"], 2),
               "Δ单笔": round(r["单笔"] - base["单笔"], 3),
               "Δ笔数": r["笔数"] - base["笔数"]}
        rows.append(row)
        pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8-sig")
        print(f"{str(v):<6} 年化{r['年化']:7.2f}%({row['Δ年化']:+6.2f}) 卡玛{r['卡玛']:5.2f}"
              f"({row['Δ卡玛']:+.2f}) 单笔{r['单笔']:.3f}%({row['Δ单笔']:+.3f}) {r['笔数']:4d}笔 "
              f"胜率{r['胜率']:.1f}% 三段{r['s1_年化']:.0f}/{r['s2_年化']:.0f}/{r['s3_年化']:.0f} "
              f"{time.time()-t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
