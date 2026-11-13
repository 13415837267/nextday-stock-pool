"""第225轮G · 分年归因：B(neg_bias20=0.85) 的改进是结构性还是某一年的运气

全史窗口 B 反而更差(-2.3pp/卡玛 -1.0), 但 1500/750 窗口 B 明显更好(+2.4pp/回撤改善)。
这种"近端好、远端差"有两种可能:
  ① 市场结构变了(注册制/20cm/量化占比上升), 近端才是未来 —— 可以落地
  ② 只是 2021-2026 某一年恰好撞上 —— 过拟合, 不能落地
判据: 逐笔按年算收益, 看 2021-2026 是否**逐年**都赢; 若只有 1-2 年赢, 判定为②。

同时输出: 每年交易笔数, 用于判断"某年样本是否太少"。
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context, run_backtest  # noqa: E402
from src.config import load_config  # noqa: E402


def yearly(res: dict) -> pd.DataFrame:
    eq = pd.DataFrame(res["equity"])
    eq["d"] = pd.to_datetime(eq["d"])
    eq["y"] = eq["d"].dt.year
    out = []
    for y, g in eq.groupby("y"):
        nav = g["nav"].values
        out.append({"年": int(y), "收益%": round((nav[-1] / nav[0] - 1) * 100, 2)})
    return pd.DataFrame(out)


def main() -> None:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 4100
    base = load_config()
    ctx = prepare_context(base, days)
    print(f"窗口 {len(ctx['main_dates'])} 日", flush=True)

    res = {}
    for nm, w in [("A", 0.70), ("B", 0.85)]:
        c = copy.deepcopy(base)
        c["selection"]["dip_weights"]["neg_bias20"] = w
        res[nm] = run_backtest(c, days, ctx)
        m = res[nm]["metrics"]
        print(f"{nm}: 年化 {m['年化收益率%']}% 回撤 {m['最大回撤%']}% 卡玛 {m['卡玛比率']} "
              f"笔 {m['交易次数']}", flush=True)

    ya, yb = yearly(res["A"]), yearly(res["B"])
    df = ya.merge(yb, on="年", suffixes=("_A", "_B"))
    df["差"] = (df["收益%_B"] - df["收益%_A"]).round(2)
    df["B胜"] = df["差"] > 0
    print("\n=== 分年收益 ===")
    print(df.to_string(index=False))
    recent = df[df["年"] >= 2021]
    print(f"\n2021起: {len(recent)} 年, B 胜 {int(recent['B胜'].sum())} 年, "
          f"累计 A {recent['收益%_A'].sum():.1f}% vs B {recent['收益%_B'].sum():.1f}%")
    print(f"全期: {len(df)} 年, B 胜 {int(df['B胜'].sum())} 年")
    df.to_csv(ROOT / "output" / "w225g_yearly.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
