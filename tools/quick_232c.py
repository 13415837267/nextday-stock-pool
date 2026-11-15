"""第232轮 C · T+1 收盘状态 → 决定 T+2 是否提前卖（截左尾）

现行卖出是"T+2 收盘一次性卖"，不看 T+1 发生什么。
但 T+1 收盘时我们已经知道这笔是浮盈还是浮亏 —— 这是完全可执行的信息
（T+1 收盘后挂 T+2 的开盘卖单）。

问题：截掉"T+1 就亏"的那批（T+2 开盘卖），能不能提升单笔？
  · 如果亏损会延续 → 早卖避损，正贡献
  · 如果亏损会反弹 → 早卖砍掉了"深 V"，负贡献
第 224 轮测过"T+1 开盘就低开则跳过买入"，测过"持有期顺延"，但没测过
"用 T+1 收盘的浮盈状态决定 T+2 怎么卖"。这是个条件化持有期。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PANEL = ROOT / "data" / "panel"
OUT = ROOT / "output" / "w232c_exit.csv"


def px(d: str, code: str) -> tuple[float, float] | None:
    f = PANEL / f"{d}.csv.gz"
    if not f.exists():
        return None
    df = pd.read_csv(f, usecols=["ts_code", "open", "close"])
    r = df[df["ts_code"] == code]
    if not len(r):
        return None
    return float(r["open"].iloc[0]), float(r["close"].iloc[0])


def main() -> None:
    t = pd.read_csv(ROOT / "output" / "w232_trades.csv", dtype={"signal": str})
    dates = sorted(p.stem.split(".")[0] for p in PANEL.glob("*.csv.gz"))
    idx = {d: i for i, d in enumerate(dates)}

    rows = []
    for _, r in t.iterrows():
        i0 = idx.get(r["signal"])
        if i0 is None or i0 + 2 >= len(dates):
            continue
        p1, p2 = px(dates[i0 + 1], r["code"]), px(dates[i0 + 2], r["code"])
        if not (p1 and p2):
            continue
        O1, C1 = p1
        O2, C2 = p2
        if min(O1, C1, O2, C2) <= 0:
            continue
        rows.append({"signal": r["signal"], "year": r["signal"][:4],
                     "O1": O1, "C1": C1, "O2": O2, "C2": C2})
    d = pd.DataFrame(rows)
    d["ret1"] = (d["C1"] / d["O1"] - 1) * 100      # T+1 收盘时浮盈
    d["hold"] = (d["C2"] / d["O1"] - 1) * 100      # 现行：持到 T+2 收盘
    d["early"] = (d["O2"] / d["O1"] - 1) * 100     # T+2 开盘卖
    print(f"样本 {len(d)} 笔；T+1 收盘浮盈均值 {d['ret1'].mean():+.3f}% "
          f"（>0 占 {(d['ret1'] > 0).mean()*100:.1f}%）\n", flush=True)

    RULES = {
        "现行(全部持到T+2收盘)": lambda x: d["hold"],
        "T+1收盘亏(<0)则T+2开盘卖": lambda x: d["early"].where(d["ret1"] < 0, d["hold"]),
        "T+1收盘亏<-2%则早卖": lambda x: d["early"].where(d["ret1"] < -2, d["hold"]),
        "T+1收盘亏<-5%则早卖": lambda x: d["early"].where(d["ret1"] < -5, d["hold"]),
        "T+1收盘盈>3%则早卖(止盈)": lambda x: d["early"].where(d["ret1"] > 3, d["hold"]),
        "T+1收盘盈>8%则早卖(止盈)": lambda x: d["early"].where(d["ret1"] > 8, d["hold"]),
        "全部T+2开盘卖": lambda x: d["early"],
    }
    base = d["hold"]
    out = []
    for name, fn in RULES.items():
        s = fn(None)
        out.append({
            "规则": name, "单笔": round(s.mean(), 3),
            "Δ单笔": round(s.mean() - base.mean(), 3),
            "胜率": round((s > 0).mean() * 100, 1),
            "中位": round(s.median(), 3),
        })
    o = pd.DataFrame(out)
    print(o.to_string(index=False))

    # 分年看"亏则早卖"是否稳定
    print("\n=== 分年（亏则早卖 vs 现行）===")
    d["rule"] = d["early"].where(d["ret1"] < 0, d["hold"])
    g = d.groupby("year").agg(现行=("hold", "mean"), 亏则早卖=("rule", "mean"),
                              笔数=("hold", "size")).round(2)
    g["Δ"] = (g["亏则早卖"] - g["现行"]).round(2)
    print(g.to_string())

    # 截断的样本到底是什么：T+1 亏的笔，T+2 开盘 vs 收盘
    sub = d[d["ret1"] < 0]
    print(f"\nT+1 收盘亏损的 {len(sub)} 笔：T+2 开盘卖 {sub['early'].mean():+.3f}% vs "
          f"T+2 收盘卖 {sub['hold'].mean():+.3f}% → 早卖{'赚' if sub['early'].mean() > sub['hold'].mean() else '亏'} "
          f"{abs(sub['early'].mean() - sub['hold'].mean()):.3f}pp")
    o.to_csv(OUT, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
