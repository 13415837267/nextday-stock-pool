"""第251轮 B: 收敛规则网格 —— 砍到多少笔最划算

用户: 单吊系统, 补位一年 30 笔太多。
A 已看出三处明显漏点:
  · 指数 -4.2~-3.51 这档 77 笔, 单笔只有 0.64%(全量 1.46%)
  · dh60 -38~-35 这档 12 笔, 单笔 -0.63%
  · 深低开 -9.5~-8 这档 22 笔, 单笔 0.99%(跌停附近开盘往往继续跌)
本脚本把"门槛"和"波内去重"交叉扫一遍, 输出笔数/单笔/合计的性价比曲面。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
COST = 0.3


def waves(df: pd.DataFrame, gap: int) -> pd.Series:
    d = df.sort_values("ci")
    w, last = [], None
    for x in d.ci:
        if last is None or x - last > gap:
            w.append(len(w) + 1)
        else:
            w.append(w[-1])
        last = x
    return pd.Series(w, index=d.index)


def evaluate(c: pd.DataFrame, idx_max: float, dh_max: float, dlo: float,
             wave_cap: int | None, rk_max: int | None) -> dict:
    b = c[c.brk].copy()
    b = b[(b.idx <= idx_max) & (b.dh60 <= dh_max)]
    if wave_cap:
        b["w"] = waves(b, 5)
        b["sq"] = b.groupby("w").cumcount() + 1
        b = b[b.sq <= wave_cap]
    d = c[~c.brk].copy()
    d = d[(d.n1_gap >= dlo) & (d.n1_gap <= -5.0)]
    if rk_max:
        d = d[d.rk <= rk_max]
    t = pd.concat([b, d])
    return {"破位": len(b), "深低开": len(d), "合计": len(t),
            "单笔": t.r.mean() if len(t) else float("nan"),
            "胜率": (t.r > 0).mean() * 100 if len(t) else float("nan"),
            "合计pp": t.r.sum(), "最差": t.r.min() if len(t) else float("nan")}


def main() -> None:
    c = pd.read_csv(OUT / "w248_cands.csv")
    c["r"] = c["y"] - COST

    base = evaluate(c, -99, 99, -9.5, None, None)
    print("=" * 96)
    print("基准(251 之前):", base)
    print("=" * 96)

    rows = []
    for idx_max in (-3.51, -4.2, -5.0):
        for dh_max in (-35.0, -42.0, -45.0):
            for dlo in (-9.5, -8.0):
                for wc in (None, 2, 1):
                    for rk in (None, 3):
                        r = evaluate(c, idx_max, dh_max, dlo, wc, rk)
                        r.update(idx=idx_max, dh=dh_max, dlo=dlo,
                                 wave="不限" if wc is None else f"每波<={wc}",
                                 rk="不限" if rk is None else f"前{rk}")
                        rows.append(r)
    t = pd.DataFrame(rows)[["idx", "dh", "dlo", "wave", "rk", "破位", "深低开", "合计",
                            "单笔", "胜率", "合计pp", "最差"]]
    t["单笔"] = t["单笔"].round(2)
    t["胜率"] = t["胜率"].round(0)
    t["合计pp"] = t["合计pp"].round(0)
    t["最差"] = t["最差"].round(1)

    print("\n【按合计 pp 排序, 只要笔数 <= 160 的】")
    s = t[t["合计"] <= 160].sort_values("合计pp", ascending=False)
    print(s.head(18).to_string(index=False))

    print("\n【按单笔排序, 只要笔数 >= 40 的】")
    s2 = t[(t["合计"] >= 40) & (t["合计"] <= 200)].sort_values("单笔", ascending=False)
    print(s2.head(16).to_string(index=False))

    print("\n【几组有代表性的方案】")
    picks = [
        ("原样", -99, 99, -9.5, None, None),
        ("只收门槛(温和)", -4.2, -42.0, -8.0, None, None),
        ("门槛+每波2笔", -4.2, -42.0, -8.0, 2, None),
        ("门槛+每波1笔", -4.2, -42.0, -8.0, 1, None),
        ("更狠: -5/-45/+每波1", -5.0, -45.0, -8.0, 1, None),
        ("门槛+每波1+前3", -4.2, -42.0, -8.0, 1, 3),
    ]
    rr = []
    for nm, a, bb, cc, wc, rk in picks:
        v = evaluate(c, a, bb, cc, wc, rk)
        v["方案"] = nm
        rr.append(v)
    p = pd.DataFrame(rr)[["方案", "破位", "深低开", "合计", "单笔", "胜率", "合计pp", "最差"]]
    p["单笔"] = p["单笔"].round(2)
    p["胜率"] = p["胜率"].round(0)
    p["合计pp"] = p["合计pp"].round(0)
    p["最差"] = p["最差"].round(1)
    print(p.to_string(index=False))

    t.to_csv(OUT / "w251_grid.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
