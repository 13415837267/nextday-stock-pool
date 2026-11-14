"""第227轮B · price_max 单调性检验（2500 日 / 10 年）

动机：2500 日复验里 price_max=100 是唯一同时改善年化(+2.75pp)与卡玛(+0.13)的候选，
但三段窗口方向不一致（1500日 +6.5pp / 750日 −1.9pp / 2500日 +2.75pp）。
若真实效应，则应随价格上限单调；若是窗口噪声，则应锯齿乱跳。

判据：年化与卡玛同时随 price_max 单调（或至少存在单一最优且两侧递减），
且最优档全史与前/后半段都不低于 150。
"""
from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest import prepare_context, run_backtest  # noqa: E402
from src.config import load_config  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "output" / "w227b_price.csv"


def seg(res: dict, half: int) -> dict:
    eq = pd.DataFrame(res["equity"]).rename(columns={"d": "date"})
    eq["date"] = eq["date"].astype(str).str.replace("-", "", regex=False)
    mid = len(eq) // 2
    s = eq.iloc[:mid] if half == 0 else eq.iloc[mid:]
    if len(s) < 30:
        return {"ann": 0.0, "cal": 0.0}
    nav = s["nav"].astype(float).values
    yrs = max(0.3, len(s) / 244.0)
    ann = (nav[-1] / nav[0]) ** (1 / yrs) - 1
    mdd = float((nav / pd.Series(nav).cummax().values - 1).min())
    return {"ann": round(ann * 100, 2), "cal": round(ann / abs(mdd) if mdd else 0, 2)}


def main() -> None:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 2500
    levels = [60, 80, 100, 120, 150]
    base = load_config()
    t0 = time.time()
    ctx = prepare_context(base, days)
    print(f"ctx {time.time()-t0:.0f}s\n", flush=True)

    rows = []
    for v in levels:
        c = copy.deepcopy(base)
        c["universe"]["price_max"] = float(v)
        t = time.time()
        res = run_backtest(c, days, ctx)
        m = res["metrics"]
        a, b = seg(res, 0), seg(res, 1)
        rows.append({
            "price_max": v,
            "年化": round(float(m["年化收益率%"]), 2),
            "回撤": round(float(m["最大回撤%"]), 2),
            "卡玛": round(float(m["卡玛比率"] or 0), 2),
            "笔数": int(m["交易次数"]),
            "胜率": round(float(m["胜率%"]), 1),
            "单笔": round(float(m["平均单笔收益%"]), 3),
            "前半年化": a["ann"], "前半卡玛": a["cal"],
            "后半年化": b["ann"], "后半卡玛": b["cal"],
        })
        pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8-sig")
        r = rows[-1]
        print(f"price_max={v:<4} 年化{r['年化']:7.2f}% 回撤{r['回撤']:7.2f}% 卡玛{r['卡玛']:5.2f}"
              f" | 前{r['前半年化']:7.2f}% 后{r['后半年化']:7.2f}% | {r['笔数']}笔 {time.time()-t:.0f}s",
              flush=True)

    d = pd.DataFrame(rows)
    print("\n=== 单调性 ===")
    print(d[["price_max", "年化", "卡玛"]].to_string(index=False))
    ann = d["年化"].values
    cal = d["卡玛"].values
    mono = all(ann[i] >= ann[i + 1] for i in range(len(ann) - 1))
    mono_lo = all(ann[i] <= ann[i + 1] for i in range(len(ann) - 1))
    print(f"年化单调递减={mono} 单调递增={mono_lo} 极差={ann.max()-ann.min():.2f}pp")
    best = d.loc[d["年化"].idxmax()]
    print(f"最优档 price_max={best['price_max']:g} 年化{best['年化']}% 卡玛{best['卡玛']}")


if __name__ == "__main__":
    main()
