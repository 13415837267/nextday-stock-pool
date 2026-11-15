"""第二百三十四轮: 放开选股池 —— 全样本说"越小越好", 而策略只买主板

全市场 817 万低开样本上, circ_mv 与次日收益的单调性 -0.946(26 档),
是最强的单变量之一: 最小档 0.717% vs 最大档 0.078%。
但 config 里 universe.boards = ["主板"] —— 策略从早期版本起就只买主板,
把创业板/科创板(小市值最集中的地方)整块排除了。

本轮直接测: 放开板块 & 限制市值上限 & 要求最小振幅。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context, run_backtest      # noqa: E402
from src.config import load_config                          # noqa: E402

OUT = ROOT / "output" / "w234_board.csv"


def brief(res: dict) -> dict:
    m = res["metrics"]
    nav = pd.DataFrame(res["equity"])["nav"].astype(float).values
    n = len(nav)
    seg = {}
    for i, tag in enumerate(["s1", "s2", "s3"]):
        x = nav[int(n * i / 3): int(n * (i + 1) / 3)]
        yrs = max(0.3, len(x) / 244.0)
        seg[tag] = round(((x[-1] / x[0]) ** (1 / yrs) - 1) * 100, 1) if len(x) > 40 else None
    return {"年化": round(float(m["年化收益率%"]), 2),
            "回撤": round(float(m["最大回撤%"]), 2),
            "卡玛": round(float(m["卡玛比率"] or 0), 2),
            "笔数": int(m["交易次数"]),
            "胜率": round(float(m["胜率%"]), 1),
            "单笔": round(float(m["平均单笔收益%"]), 3), **seg}


def mk(base: dict, **kw) -> dict:
    c = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
    c["universe"] = dict(base["universe"])
    c["selection"] = dict(base["selection"])
    c["execution"] = dict(base["execution"])
    for k, v in kw.items():
        if k in ("boards", "circ_mv_max", "circ_mv_min", "amp_min", "price_min",
                 "price_max", "amount_min_yi", "turnover_min", "turnover_max"):
            c["universe"][k] = v
        elif k.startswith("dip_"):
            c["selection"][k] = v
        else:
            c["execution"][k] = v
    return c


def main() -> None:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 2500
    base = load_config()
    t0 = time.time()
    ctx = prepare_context(base, days)
    print(f"ctx {time.time()-t0:.0f}s, {len(ctx['main_dates'])} 日\n", flush=True)

    plan = [
        ("B0 主板(现行)", {}),
        ("B1 主板+创业板", dict(boards=["主板", "创业板"])),
        ("B2 主板+创业板+科创", dict(boards=["主板", "创业板", "科创板"])),
        ("C1 市值上限 80亿", dict(circ_mv_max=80)),
        ("C2 市值上限 50亿", dict(circ_mv_max=50)),
        ("C3 市值上限 30亿", dict(circ_mv_max=30)),
        ("D1 振幅下限 4%", dict(amp_min=4.0)),
        ("D2 振幅下限 6%", dict(amp_min=6.0)),
        ("D3 振幅下限 8%", dict(amp_min=8.0)),
    ]
    rows = []
    for name, kw in plan:
        c = mk(base, **kw)
        t = time.time()
        r = brief(run_backtest(c, None, ctx))
        r["方案"] = name
        if rows:
            r["Δ年化"] = round(r["年化"] - rows[0]["年化"], 2)
            r["Δ笔数"] = r["笔数"] - rows[0]["笔数"]
        else:
            r["Δ年化"], r["Δ笔数"] = 0.0, 0
        rows.append(r)
        print(f"{name:22s} 年化 {r['年化']:7.2f}% ({r['Δ年化']:+6.2f})  "
              f"回撤 {r['回撤']:7.2f}%  卡玛 {r['卡玛']:5.2f}  "
              f"笔数 {r['笔数']:4d} ({r['Δ笔数']:+4d})  单笔 {r['单笔']:+.3f}%  "
              f"[{time.time()-t:.0f}s]", flush=True)
        pd.DataFrame(rows)[["方案", "年化", "回撤", "卡玛", "笔数", "胜率", "单笔",
                            "s1", "s2", "s3", "Δ年化", "Δ笔数"]].to_csv(
            OUT, index=False, encoding="utf-8-sig")

    d = pd.DataFrame(rows)
    print("\n" + d[["方案", "年化", "回撤", "卡玛", "笔数", "胜率", "单笔",
                    "s1", "s2", "s3", "Δ年化", "Δ笔数"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
