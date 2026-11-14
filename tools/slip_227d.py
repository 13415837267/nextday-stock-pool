"""第227轮D · 买入端滑点归零的真实效果（分端滑点）

机制依据（不是回测美化）：
  本策略只在 T+1 **低开**时买入。9:15~9:24 挂"前收盘价"限价买单，
  集合竞价撮合规则决定成交价 = 开盘价（因为低开，开盘价 < 挂单价）。
  也就是说买入端滑点在机制上为 0，此前统一扣千 1 是系统性低估。
  卖出端是 T+2 收盘市价单，滑点真实存在，必须保留。

四档：
  A 旧口径        买千1 卖千1（应等于历史数值，用于校验口径一致性）
  B 买0  卖千1     ← 只需做到"9:15~9:24 挂前收价限价单"
  C 买0  卖千1 佣金万1
  D 买0  卖千0.5 （卖出也用限价/分批的乐观情形）
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

OUT = Path(__file__).resolve().parents[1] / "output" / "w227d_slip.csv"


def seg(res: dict, half: int) -> tuple[float, float]:
    eq = pd.DataFrame(res["equity"]).rename(columns={"d": "date"})
    eq["date"] = eq["date"].astype(str).str.replace("-", "", regex=False)
    mid = len(eq) // 2
    s = eq.iloc[:mid] if half == 0 else eq.iloc[mid:]
    if len(s) < 30:
        return 0.0, 0.0
    nav = s["nav"].astype(float).values
    yrs = max(0.3, len(s) / 244.0)
    ann = (nav[-1] / nav[0]) ** (1 / yrs) - 1
    mdd = float((nav / pd.Series(nav).cummax().values - 1).min())
    return round(ann * 100, 2), round(ann / abs(mdd) if mdd else 0, 2)


def main() -> None:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 2500
    base = load_config()
    t0 = time.time()
    ctx = prepare_context(base, days)
    print(f"ctx {time.time()-t0:.0f}s, {days} 日\n", flush=True)

    cases = [
        ("A 旧口径 买千1卖千1", {}),
        ("B 买0 卖千1", {"slippage_buy": 0.0, "slippage_sell": 0.001}),
        ("C 买0 卖千1 佣金万1", {"slippage_buy": 0.0, "slippage_sell": 0.001,
                                 "commission_rate": 0.0001}),
        ("D 买0 卖千0.5", {"slippage_buy": 0.0, "slippage_sell": 0.0005}),
        ("E 买0 卖0", {"slippage_buy": 0.0, "slippage_sell": 0.0}),
        ("F 买0 卖0 佣金万1", {"slippage_buy": 0.0, "slippage_sell": 0.0,
                               "commission_rate": 0.0001}),
    ]
    rows = []
    for name, p in cases:
        c = copy.deepcopy(base)
        for k, v in p.items():
            c["execution"][k] = v
        t = time.time()
        res = run_backtest(c, days, ctx)
        m = res["metrics"]
        a = seg(res, 0)
        b = seg(res, 1)
        rows.append({
            "方案": name,
            "年化": round(float(m["年化收益率%"]), 2),
            "回撤": round(float(m["最大回撤%"]), 2),
            "卡玛": round(float(m["卡玛比率"] or 0), 2),
            "笔数": int(m["交易次数"]),
            "胜率": round(float(m["胜率%"]), 1),
            "单笔": round(float(m["平均单笔收益%"]), 3),
            "前半年化": a[0], "后半年化": b[0],
        })
        pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8-sig")
        r = rows[-1]
        print(f"{name:22s} 年化{r['年化']:7.2f}% 回撤{r['回撤']:7.2f}% 卡玛{r['卡玛']:5.2f}"
              f" 单笔{r['单笔']:6.3f}% | 前{r['前半年化']:7.2f}% 后{r['后半年化']:7.2f}%"
              f" | {r['笔数']}笔 {time.time()-t:.0f}s", flush=True)

    d = pd.DataFrame(rows)
    b0 = d.iloc[0]
    print("\n=== 相对旧口径 ===")
    for _, r in d.iloc[1:].iterrows():
        print(f"{r['方案']:22s} Δ年化{r['年化']-b0['年化']:+6.2f}pp "
              f"Δ卡玛{r['卡玛']-b0['卡玛']:+5.2f} Δ单笔{r['单笔']-b0['单笔']:+.3f}%")


if __name__ == "__main__":
    main()
