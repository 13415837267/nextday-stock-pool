"""第251轮 A: 补位笔级体检 —— 找出"哪些补位是噪音"

用户诉求: 这是单吊系统, 补位 2500 日做了 306 笔(年均 28 笔), 把系统做成了高频。
    主策略 563 笔 / 2500 日 = 年均 56 笔, 单吊的合理节奏。
    补位要能少做, 就得先知道砍哪一类不掉肉。

本脚本只看笔级收益(毛 y - 0.3% 成本), 不做资金占用/净值, 目的:
  1. 破位日的"波"结构 —— 一次暴跌连续几天都在破位, 每天都在抄底, 其实是同一笔观点
  2. 收益集中在哪些笔 —— 少数极端笔是否撑起了全部收益
  3. 各维度分档的单调性 —— 指数深度 / dh60 深度 / 缺口深度 / 排名
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "output"
COST = 0.3


def seg(df: pd.DataFrame, col: str, bins: list[float], lab: list[str]) -> None:
    d = df.copy()
    d["g"] = pd.cut(d[col], bins=bins, labels=lab, right=False)
    t = d.groupby("g", observed=True).agg(
        n=("r", "size"), 单笔=("r", "mean"), 胜率=("r", lambda s: (s > 0).mean() * 100),
        合计=("r", "sum"), 最差=("r", "min"), 最好=("r", "max"))
    t["单笔"] = t["单笔"].round(2)
    t["胜率"] = t["胜率"].round(0)
    t["合计"] = t["合计"].round(1)
    t["最差"] = t["最差"].round(1)
    t["最好"] = t["最好"].round(1)
    print(f"\n-- 按 {col} --")
    print(t.to_string())


def main() -> None:
    c = pd.read_csv(OUT / "w248_cands.csv")
    c["r"] = c["y"] - COST
    c["yr"] = c.trade_date // 10000
    c = c.sort_values("trade_date").reset_index(drop=True)

    print("=" * 84)
    print(f"全量候选 {len(c)} 天   破位 {int(c.brk.sum())} / 深低开 {int((~c.brk).sum())}")
    print(f"单笔 {c.r.mean():.2f}%  合计 {c.r.sum():.0f}pp  胜率 {(c.r>0).mean()*100:.0f}%")
    print("=" * 84)

    b = c[c.brk].copy()
    d = c[~c.brk].copy()

    # ---------------- 1. 破位日的"波"结构 ----------------
    # 破位日本身就是暴跌日, 一次下跌往往连续 2-5 天都满足条件。
    # 用交易日序号(ci)差 <= 5 归为同一波。
    b = b.sort_values("ci").reset_index(drop=True)
    wave, last = [], None
    for x in b.ci:
        if last is None or x - last > 5:
            wave.append(len(wave) + 1)
        else:
            wave.append(wave[-1])
        last = x
    b["wave"] = wave
    b["seq"] = b.groupby("wave").cumcount() + 1
    nw = b.wave.nunique()
    print(f"\n[破位] {len(b)} 笔 落在 {nw} 波里（间隔 5 个交易日内的算同一波）")
    print(b.groupby("seq").agg(笔数=("r", "size"), 单笔=("r", "mean"),
                               胜率=("r", lambda s: (s > 0).mean() * 100)).round(2).to_string())
    sz = b.groupby("wave").size()
    print(f"  每波笔数: 1 笔 {int((sz==1).sum())} 波 / 2 笔 {int((sz==2).sum())} 波 / "
          f"3 笔 {int((sz==3).sum())} 波 / 4+ 笔 {int((sz>=4).sum())} 波")
    print(f"  只做每波第 1 笔: {nw} 笔 单笔 {b[b.seq==1].r.mean():.2f}% 合计 {b[b.seq==1].r.sum():.0f}pp")
    print(f"  只做每波前 2 笔: {int((b.seq<=2).sum())} 笔 单笔 {b[b.seq<=2].r.mean():.2f}% "
          f"合计 {b[b.seq<=2].r.sum():.0f}pp")
    print(f"  第 2 笔及以后: {int((b.seq>1).sum())} 笔 单笔 {b[b.seq>1].r.mean():.2f}% "
          f"合计 {b[b.seq>1].r.sum():.0f}pp")

    # ---------------- 2. 收益集中度 ----------------
    print("\n[收益集中度]")
    for nm, s in (("破位", b.r), ("深低开", d.r)):
        s = s.sort_values()
        tot = s.sum()
        print(f"  {nm}: 合计 {tot:.0f}pp | 最差 5 笔 {s.head(5).sum():.0f}pp | "
              f"最好 5 笔 {s.tail(5).sum():.0f}pp | 中位 {s.median():.2f}%")
        if tot:
            pos = s[s > 0].sum()
            print(f"     正收益笔贡献 {pos:.0f}pp, 前 10% 笔贡献 "
                  f"{s.tail(max(1, int(len(s)*0.1))).sum():.0f}pp")

    # ---------------- 3. 分档 ----------------
    seg(b, "idx", [-99, -5, -4.2, -3.51], ["<=-5%", "-5~-4.2", "-4.2~-3.51"])
    seg(b, "dh60", [-99, -55, -45, -38, -35], ["<=-55", "-55~-45", "-45~-38", "-38~-35"])
    seg(d, "n1_gap", [-9.5, -8, -6.5, -5.01], ["-9.5~-8", "-8~-6.5", "-6.5~-5"])
    seg(d, "rk", [0, 1, 3, 6, 11], ["rank1", "rank2-3", "rank4-6", "rank7-10"])

    # ---------------- 4. 逐年 ----------------
    print("\n[逐年]")
    t = c.groupby(["yr", "path"]).agg(n=("r", "size"), 单笔=("r", "mean"), 合计=("r", "sum"))
    t["单笔"] = t["单笔"].round(2)
    t["合计"] = t["合计"].round(1)
    print(t.unstack("path").fillna(0).to_string())

    b.to_csv(OUT / "w251_brk.csv", index=False, encoding="utf-8-sig")
    print(f"\n已写 {OUT/'w251_brk.csv'}")


if __name__ == "__main__":
    main()
