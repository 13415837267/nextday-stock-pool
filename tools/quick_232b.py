"""第232轮 B · 不改引擎先算账：T日尾盘买(持1天) vs T+1开盘买(持2天)

背景：现行结构的收益 = C2 / O1 - 1（T+1 开盘买、T+2 收盘卖）。
227 轮把滑点归零，本质是**白拿了 T+1 的低开折扣**（O1 通常比 C0 低 1% 左右）。

但 229 轮测出 alpha 寿命只有 1 天（持有期顺延 −66~−103pp），
这意味着 T+2 那一天很可能是负贡献。两个效应方向相反：
  低开折扣   +1%    （现行独有）
  多持一天   −?     （现行独有）

如果"多持一天"的损失大于 1%，那"T 日尾盘买、T+1 收盘卖"（持 1 天）更好。
这只是个减法，不需要改引擎 —— 用面板原始价直接算 559 笔的两种口径。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PANEL = ROOT / "data" / "panel"
OUT = ROOT / "output" / "w232b_hold.csv"


def day_close(d: str, code: str) -> dict | None:
    f = PANEL / f"{d}.csv.gz"
    if not f.exists():
        return None
    df = pd.read_csv(f, usecols=["ts_code", "open", "close"])
    r = df[df["ts_code"] == code]
    if not len(r):
        return None
    return {"open": float(r["open"].iloc[0]), "close": float(r["close"].iloc[0])}


def main() -> None:
    t = pd.read_csv(ROOT / "output" / "w232_trades.csv", dtype={"signal": str, "buy": str, "sell": str})
    # 交易日序列（面板文件名即交易日）
    dates = sorted(p.stem.split(".")[0] for p in PANEL.glob("*.csv.gz"))
    idx = {d: i for i, d in enumerate(dates)}

    rows = []
    for _, r in t.iterrows():
        i0 = idx.get(r["signal"])
        if i0 is None or i0 + 2 >= len(dates):
            continue
        d0, d1, d2 = dates[i0], dates[i0 + 1], dates[i0 + 2]
        r0, r1, r2 = day_close(d0, r["code"]), day_close(d1, r["code"]), day_close(d2, r["code"])
        if not (r0 and r1 and r2):
            continue
        C0, O1, C1, C2 = r0["close"], r1["open"], r1["close"], r2["close"]
        if min(C0, O1, C1, C2) <= 0:
            continue
        rows.append({
            "signal": r["signal"], "code": r["code"], "pool": r["pool"],
            "cur": (C2 / O1 - 1) * 100,        # 现行：T+1 开盘买，T+2 收盘卖
            "alt": (C1 / C0 - 1) * 100,        # 替代：T 日收盘买，T+1 收盘卖
            "gap": (O1 / C0 - 1) * 100,        # T+1 低开幅度
            "c2c1": (C2 / C1 - 1) * 100,       # T+2 相对 T+1 收盘
            "d2o1": (C1 / O1 - 1) * 100,       # T+1 日内
        })
    d = pd.DataFrame(rows)
    d.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"样本 {len(d)} 笔\n", flush=True)
    for c, lab in [("cur", "现行 C2/O1-1"), ("alt", "替代 C1/C0-1"),
                   ("gap", "T+1低开"), ("d2o1", "T+1日内"), ("c2c1", "T+2相对T+1")]:
        s = d[c]
        print(f"{lab:14s} 均值 {s.mean():+6.3f}% 中位 {s.median():+6.3f}% "
              f"胜率 {(s > 0).mean()*100:5.1f}%")
    win = (d["alt"] > d["cur"]).mean() * 100
    print(f"\n替代胜出（逐笔）{win:.1f}%")
    print(f"现行 {d['cur'].mean():.3f}%  vs  替代 {d['alt'].mean():.3f}%  "
          f"差 {d['alt'].mean() - d['cur'].mean():+.3f}pp")
    print(f"低开折扣 {(-d['gap']).mean():.3f}pp，多持一天 {d['c2c1'].mean():+.3f}pp")


if __name__ == "__main__":
    main()
