"""第232轮 D · alpha 衰减曲线：买入后第 k 个交易日各贡献多少

229 轮用"持有期顺延"测出延长持有 −66~−103pp，结论写成"alpha 寿命 1 天"。
但那是**累计**口径。本脚本把累计拆成逐日：买入后第 1/2/3/4/5 天各自贡献多少，
拐点在第几天 —— 这决定了"能不能多持一天"这种结构性选择还有没有空间。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PANEL = ROOT / "data" / "panel"
OUT = ROOT / "output" / "w232d_decay.csv"
KMAX = 6


def closes(code: str, ds: list[str]) -> list[float | None]:
    out = []
    for d in ds:
        f = PANEL / f"{d}.csv.gz"
        if not f.exists():
            out.append(None)
            continue
        df = pd.read_csv(f, usecols=["ts_code", "open", "close"])
        r = df[df["ts_code"] == code]
        out.append(float(r["close"].iloc[0]) if len(r) else None)
    return out


def main() -> None:
    t = pd.read_csv(ROOT / "output" / "w232_trades.csv", dtype={"signal": str})
    dates = sorted(p.stem.split(".")[0] for p in PANEL.glob("*.csv.gz"))
    idx = {d: i for i, d in enumerate(dates)}

    rows = []
    for _, r in t.iterrows():
        i0 = idx.get(r["signal"])
        if i0 is None or i0 + KMAX >= len(dates):
            continue
        ds = dates[i0: i0 + KMAX + 1]
        cs = closes(r["code"], ds)
        if any(c is None or c <= 0 for c in cs):
            continue
        # 第 k 天 = 从 T+k-1 收盘到 T+k 收盘（T+1 的第一段用开盘价起算，单独处理）
        row = {"signal": r["signal"], "ret": float(r["ret"])}
        for k in range(1, KMAX + 1):
            row[f"d{k}"] = (cs[k] / cs[k - 1] - 1) * 100
        rows.append(row)
    d = pd.DataFrame(rows)
    d.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"样本 {len(d)} 笔\n")

    print("=== 买入后逐日贡献（收盘到收盘）===")
    for k in range(1, KMAX + 1):
        s = d[f"d{k}"]
        cum = (1 + s / 100).prod() - 1
        print(f"第{k}天 {s.mean():+6.3f}%  胜率 {(s > 0).mean()*100:5.1f}%  "
              f"累计 {(cum*100):+7.1f}%")
    print("\n（注：第1天为 T 日收盘→T+1 收盘；实盘从 T+1 开盘进，%s）"
          % "故实盘第1天贡献约为 T+1 日内 +0.85%")


if __name__ == "__main__":
    main()
