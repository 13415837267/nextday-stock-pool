"""第二百三十七轮 B: 把挖出来的 6 个条件验一遍(分年 / 重叠 / 能不能只买一只)

第 237 轮从 817 万可执行样本、903 对二维组合里挖出 6 个通过硬门槛的格子。
本轮不给结论, 只把每个条件查清楚三件事:
  ① 分年收益(哪几年靠什么赚的, 有没有一两年撑着)
  ② 与现行策略 559 笔的重叠(重叠高 = 老东西换说法; 重叠 0 = 新方向)
  ③ 每天只买一只时的表现(条件内按 rel_pct 极端度排序)

输出 output/w237b_cond{1..6}.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "output" / "feat234"
OUT = ROOT / "output"
TRADES = OUT / "w232_trades.csv"

COLS = ["ts_code", "trade_date", "y", "n1_ok", "pct", "rel_pct", "bias10", "bias20",
        "ret10", "maxdd20", "vol20", "is_paused", "is_st", "listed", "circ_mv",
        "close", "turnover_rate", "amp"]

CONDS = {
    "C1 10日急杀但20日仍强": lambda d: (d.bias10 < -3.06) & (d.bias20 > 4.77),
    "C2 10日急杀但10日仍涨": lambda d: (d.bias10 < -3.06) & (d.ret10 > 6.38),
    "C3 普涨日滞涨":         lambda d: (d.pct > 1.79) & (d.rel_pct < -1.61),
    "C4 抗跌但单日急杀":     lambda d: (d.rel_pct > 1.29) & ((d.maxdd20 - d.pct) > 1.71),
    "C5 弱于大盘+高波动杀":  lambda d: (d.rel_pct < -1.61) & ((d.vol20 * d.pct) > 4.41),
    "C6 20日强但10日回调":   lambda d: (d.bias20 > 4.77) & d.ret10.between(-5.87, -1.72),
}


def load() -> pd.DataFrame:
    parts = []
    for p in sorted(FEAT.glob("f*.parquet")):
        d = pd.read_parquet(p, columns=COLS)
        d = d[(d["is_paused"] == 0) & (d["is_st"] == 0)]
        d = d[d["n1_ok"] == True]                       # noqa: E712
        d = d[d["pct"].between(-11.5, 11.5)]
        d = d[d["y"].between(-40, 60)]
        d = d[d["listed"] >= 250]
        if len(d):
            parts.append(d)
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    d = load()
    d["year"] = (d["trade_date"] // 10000).astype(int)
    d["key"] = d["ts_code"] + "_" + d["trade_date"].astype(str)
    tr = pd.read_csv(TRADES).rename(columns={"signal": "trade_date", "code": "ts_code"})
    tr["key"] = tr["ts_code"] + "_" + tr["trade_date"].astype(str)
    tk = set(tr["key"])
    print(f"可执行样本 {len(d):,} / 现行成交 {len(tr)} 笔\n")

    for name, fn in CONDS.items():
        m = fn(d)
        s = d[m]
        if not len(s):
            print(f"{name}: 空")
            continue
        hit = len(set(s["key"]) & tk)
        ny = s.groupby("year")["y"].agg(["size", "mean"])
        ny = ny[ny["size"] >= 40]
        # 每天只买一只: rel_pct 最极端(滞涨则最负, 其余最正) —— 这里统一取 |rel_pct| 最大
        one = s.loc[s.groupby("trade_date")["rel_pct"].apply(
            lambda x: (x - x.mean()).abs().idxmax()).values]
        print(f"{name}")
        print(f"  样本 {len(s):,} / {s['trade_date'].nunique()} 日 ({len(s)/s['trade_date'].nunique():.1f} 只/日)  "
              f"单笔 {s['y'].mean():+.3f}%  胜率 {(s['y']>0).mean()*100:.1f}%  "
              f"| 与现行成交重叠 {hit} 笔")
        print(f"  分年 {ny['mean'].round(2).to_dict()}")
        print(f"  正年 {int((ny['mean']>0).sum())}/{len(ny)}  最差 {ny['mean'].min():+.2f}% ({ny['mean'].idxmin()})  "
              f"最好 {ny['mean'].max():+.2f}% ({ny['mean'].idxmax()})")
        print(f"  只买一只(rel_pct 最极端) {one['y'].mean():+.3f}%  胜率 {(one['y']>0).mean()*100:.1f}%  "
              f"{len(one)} 日")
        ny.to_csv(OUT / f"w237b_{name.split()[0]}.csv", encoding="utf-8-sig")
        print()


if __name__ == "__main__":
    main()
