"""第249轮 A: 合入口径 vs 拼接口径 —— 差额从哪来

248 的 5.76 是「每笔都满仓」算的（base 直接用 w232_trades 的 ret 当当日收益）。
真实引擎里主策略有降仓（gap/pool/knife/dd 多档 tier），平均仓位远低于 100%。
这里用合入后的净值反推真实仓位，量化这个差。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"


def m(r: pd.Series) -> tuple[float, float, float]:
    nav = (1 + r / 100).cumprod()
    ann = nav.iloc[-1] ** (252 / len(r)) - 1
    dd = (nav / nav.cummax() - 1).min()
    return round(ann * 100, 2), round(dd * 100, 2), round(ann * 100 / abs(dd * 100), 2)


def to_ret(eq: list[dict]) -> pd.Series:
    s = pd.Series({int(x["d"].replace("-", "")): float(x["nav"]) for x in eq}).sort_index()
    return (s.pct_change().dropna() * 100).rename("r")


def main() -> None:
    j = json.loads((OUT / "w248g_combo.json").read_text(encoding="utf-8"))
    print("=" * 78)
    print("第249轮 A: 合入口径 vs 拼接口径")
    print("=" * 78)

    rs = {k: to_ret(v["equity"]) for k, v in j.items()}
    print(f"\n{'方案':<16}{'交易日':>7}{'有收益日':>8}{'零收益日':>8}{'非零日均':>9}{'全年化':>9}")
    for k, r in rs.items():
        nz = r[r.abs() > 0.01]
        print(f"{k:<16}{len(r):>7}{len(nz):>8}{len(r)-len(nz):>8}{nz.mean():>8.3f}%"
              f"{m(r)[0]:>8.2f}%")

    print("\n【1】主策略满仓口径 vs 真实账户口径")
    print("  248 拼接口径: 559 笔 × 单笔 1.28% 当每日收益(等价于每笔 100% 仓) → 86.42% / -24.24% / 3.57")
    a = rs["A 主策略单独"]
    nz = a[a.abs() > 0.01]
    print(f"  真实账户:      {len(nz)} 个有收益日, 日均 {nz.mean():.3f}% → "
          f"{m(a)[0]:.2f}% / {m(a)[1]:.2f}% / {m(a)[2]:.2f}")
    print(f"  → 真实日均 {nz.mean():.3f}% vs 满仓口径的 ~1.28%×交易频率；"
          f"248 按 100% 仓算回撤，所以把主策略的风险估大了")

    print("\n【2】三条腿的逐年（合入口径）")
    yr = pd.DataFrame({k: v.groupby(v.index // 10000).apply(
        lambda x: (1 + x / 100).prod() - 1) * 100 for k, v in rs.items()})
    yr["补位贡献_C-A"] = yr["C 248 定稿(让位)"] - yr["A 主策略单独"]
    yr["补位贡献_B-A"] = yr["B 补位只填空档"] - yr["A 主策略单独"]
    print(yr.round(1).to_string())

    print("\n【3】最大回撤发生在哪")
    for k, r in rs.items():
        nav = (1 + r / 100).cumprod()
        dd = nav / nav.cummax() - 1
        i = dd.idxmin()
        peak = nav.loc[:i].idxmax()
        print(f"  {k:<16}{dd.min()*100:>7.2f}%   {peak} → {i}")

    print("\n【4】B vs C: 让位那 34 笔换来了什么")
    b, c = rs["B 补位只填空档"], rs["C 248 定稿(让位)"]
    print(f"  B {m(b)}   C {m(c)}")
    diff = (c - b)
    print(f"  日收益差合计 {diff.sum():+.1f}pp  >0 的天数 {int((diff>0).sum())} / {int((diff<0).sum())}")
    print(f"  最差的 5 天差: {diff.nsmallest(5).round(2).to_dict()}")

    print("\n【5】空仓率（收盘无持仓的交易日占比）")
    for k, r in rs.items():
        nz = (r.abs() > 0.01).sum()
        print(f"  {k:<16}空仓 {(len(r)-nz)/len(r)*100:.0f}%")


if __name__ == "__main__":
    main()
