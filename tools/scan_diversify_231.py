"""第231轮 · 持仓只数（分散度）——唯一从没测过的维度

230轮之前的全部实验都在"买哪一只"上做（条件、排序、门槛、仓位曲线），
但整条策略从第一天起就有一个从未被检验的隐含前提：**每天最多买 1 只、满仓单吊**。

这个前提对 3 万本金的人来说不是学术问题：
  单吊 = 任何一笔亏损都是全额承担，回撤 −18% 就是 −5400 元。
  历史上 559 笔里亏 5% 以上的有多少笔，这些笔在单吊下全部砸在净值上。

本轮只问一件事：把每天的仓位拆成 N 份，年化会掉多少、回撤能降多少、卡玛往哪走。
N 只是"每天最多买几只"，不是"必然买几只"——不足 N 只时按实际成交只数投入，
剩余留现金（不做强行补仓，避免引入路径依赖）。

A 组（无限资金口径, 100 万, 无手数约束）: N=1/2/3/4/5 等权
B 组（现实口径, 3 万本金 + 100 股整数倍）: N=1/2/3
  3 万是用户真实本金；股价 30 元的票一手 3000 元，分 2 只各 1.5 万够买 5 手；
  股价 100 元的票一手 1 万，分 2 只各 1.5 万只能买 1 手(资金利用率 67%)。
  手数约束在小资金 + 分散的组合里会咬人，必须实测而不是估算。
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

OUT = ROOT / "output" / "w231_diversify.csv"   # 实际文件名在 main 里按组改


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


def mk(cfg: dict, n: int, cap: float | None = None, lot: int = 0,
       capital: float | None = None) -> dict:
    c = copy.deepcopy(cfg)
    ex = c["execution"]
    ex["max_buy_per_day"] = int(n)
    ex["alloc_frac_pct"] = round(100.0 / n, 4)
    # pos_frac_cap 必须同步收紧: 否则 mkt_boost(×1.5) 会把单只仓位顶回 100%,
    # 分散只是名义上的。cap = 单只预算上限。
    ex["pos_frac_cap"] = round(float(cap if cap is not None else 1.0 / n), 6)
    if lot:
        ex["lot_size"] = int(lot)
    if capital:
        c["backtest"]["initial_capital"] = float(capital)
    return c


def main() -> None:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 2500
    cfg = load_config()
    ctx = prepare_context(cfg, days)
    print(f"{len(ctx['main_dates'])} 日\n", flush=True)

    base = brief(run_backtest(cfg, None, ctx))
    print(f"基准 N=1 年化{base['年化']:.2f}% 回撤{base['回撤']:.2f}% 卡玛{base['卡玛']:.2f} "
          f"{base['笔数']}笔 单笔{base['单笔']:.3f}%\n", flush=True)
    rows = [{"方案": "A0 现行(单吊)", "组": "A", **base}]

    only = sys.argv[2].upper() if len(sys.argv) > 2 else ""
    global OUT
    if only:
        OUT = ROOT / "output" / f"w231{only.lower()}_diversify.csv"
    plan = [
        ("A1 2只等权", "A", dict(n=2)),
        ("A2 3只等权", "A", dict(n=3)),
        ("A3 4只等权", "A", dict(n=4)),
        ("A4 5只等权", "A", dict(n=5)),
        ("B0 3万·单吊", "B", dict(n=1, lot=100, capital=30000)),
        ("B1 3万·2只", "B", dict(n=2, lot=100, capital=30000)),
        ("B2 3万·3只", "B", dict(n=3, lot=100, capital=30000)),
        ("B3 3万·2只·10万对照", "B", dict(n=2, lot=100)),
    ]
    ref_map: dict[str, dict] = {"A": rows[0], "B": None}
    for name, grp, kw in plan:
        if only and grp != only:
            continue
        c = mk(cfg, **kw)
        t = time.time()
        r = brief(run_backtest(c, None, ctx))
        ref = ref_map[grp] or rows[0]
        row = {"方案": name, "组": grp, **r,
               "Δ年化": round(r["年化"] - ref["年化"], 2),
               "Δ回撤": round(r["回撤"] - ref["回撤"], 2),
               "Δ卡玛": round(r["卡玛"] - ref["卡玛"], 2),
               "Δ单笔": round(r["单笔"] - ref["单笔"], 3),
               "Δ笔数": r["笔数"] - ref["笔数"],
               "耗时s": round(time.time() - t, 1)}
        rows.append(row)
        if grp == "B" and ref_map["B"] is None:
            ref_map["B"] = row      # B0（3万·单吊+手数）作为 B 组基准
        print(f"{name:22s} 年化{r['年化']:7.2f}% 回撤{r['回撤']:7.2f}% 卡玛{r['卡玛']:5.2f} "
              f"{r['笔数']:4d}笔 单笔{r['单笔']:6.3f}% | Δ年化{row['Δ年化']:+7.2f} "
              f"Δ回撤{row['Δ回撤']:+6.2f} Δ卡玛{row['Δ卡玛']:+5.2f}", flush=True)
        pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8-sig")

    pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"\n已写 {OUT}", flush=True)


if __name__ == "__main__":
    main()
