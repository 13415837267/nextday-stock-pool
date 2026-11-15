"""第二百三十五轮 B: 换一套评分, 还能不能一样好?

第 235 轮已证明: 评分选股不是噪声(置换 5000 次全负, 配对 t=4.35)。
但"不是噪声"离"是最好的"还很远。这一脚本问: 在**同一个池、同一条规则(每日 top1)**
下, 换一个更笨的评分, 会不会一样好, 甚至更好?

若换评分后单笔收益掉到 0.6%(= 池均值), 说明评分权重是精确调出来的, 别的都不行 -> "最好"有据;
若换评分仍有 1.2~1.4%, 说明"每日 top1 + 这个池"本身就是印钞机, 评分只是随便挑一个 -> "最好"不成立。

样本: 取真实选中股落在 strict 池内的 510 天, 各评分都在**同一批天**上取 top1, 保证可比。
对手(每个因子正反两个方向都试, 取较强方向 —— 这是对"当前评分"最不利的设定):
  dist_high20 / bias60 / ret20 / pct / circ_mv / amp / turnover_rate / close_pos
  + 8 因子等权 zscore 组合
  + 随机(500 次)
输出 output/w235b_scores.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "output" / "feat234"
TRADES = ROOT / "output" / "w232_trades.csv"
OUT = ROOT / "output"

COLS = ["ts_code", "trade_date", "y", "n1_ok", "pct", "close_pos", "circ_mv",
        "close", "turnover_rate", "amt_yi", "amp", "bias60", "ret20", "dist_high20", "bias20"]

FACTORS = ["dist_high20", "bias60", "ret20", "pct", "circ_mv", "amp",
           "turnover_rate", "close_pos"]


def main() -> None:
    tr = pd.read_csv(TRADES).rename(columns={"signal": "trade_date", "code": "ts_code"})
    tr["trade_date"] = tr["trade_date"].astype(int)
    days = set(tr["trade_date"].unique())
    print(f"成交 {len(tr)} 笔 / {len(days)} 个信号日")

    parts = []
    for p in sorted(FEAT.glob("f*.parquet")):
        d = pd.read_parquet(p, columns=COLS)
        d = d[d["trade_date"].isin(days)]
        if len(d):
            parts.append(d)
    d = pd.concat(parts, ignore_index=True)

    base = d[(d["n1_ok"] == True) & d["pct"].between(-6.0, -3.0)]  # noqa: E712
    pool = base[(base["close_pos"] <= 0.32)
                & base["close"].between(3, 150)
                & base["circ_mv"].between(0, 600)
                & base["turnover_rate"].between(2, 12)].copy()

    # 只保留"真实选中股在池内"的天, 且池内至少 10 只(否则 top1 无意义)
    yt = d.set_index(["trade_date", "ts_code"])["y"]
    tr["y_true"] = [yt.get(k, np.nan) for k in zip(tr["trade_date"], tr["ts_code"])]
    good = tr.dropna(subset=["y_true"])
    keep = set(good["trade_date"])
    pool = pool[pool["trade_date"].isin(keep)]
    cnt = pool.groupby("trade_date").size()
    keep = set(cnt[cnt >= 10].index)
    pool = pool[pool["trade_date"].isin(keep)]
    tr2 = good[good["trade_date"].isin(keep)]
    print(f"可比天数 {len(keep)} / strict 池 {len(pool):,} 行 / 池大小中位 "
          f"{cnt[cnt.index.isin(keep)].median():.0f}")

    # 因子在池内按日 rank (0~1)
    for f in FACTORS:
        pool[f + "_r"] = pool.groupby("trade_date")[f].rank(pct=True)

    res = {}

    # 真实策略
    res["★现行评分(实际成交)"] = tr2["y_true"].mean()

    # 单因子, 正反两向
    for f in FACTORS:
        best = (-9e9, None)
        for sign, tag in ((1, "高"), (-1, "低")):
            v = []
            for _, sub in pool.groupby("trade_date"):
                s2 = sub.dropna(subset=[f + "_r"])
                if not len(s2):
                    continue
                v.append(s2.loc[(sign * s2[f + "_r"]).idxmax(), "y"])
            m = float(np.mean(v))
            res[f"{f} {'越大越好' if sign == 1 else '越小越好'}"] = m
            if m > best[0]:
                best = (m, tag)
        res[f"[最佳]{f}"] = best[0]

    # 等权 zscore 组合(池内 rank 去均值后等权, 方向取 234 轮归纳的符号)
    sign_map = {"dist_high20": 1, "bias60": 1, "ret20": 1, "pct": 1,
                "circ_mv": -1, "amp": 1, "turnover_rate": -1, "close_pos": -1}
    pool["combo"] = sum(sign_map[f] * (pool[f + "_r"] - 0.5) for f in sign_map)
    v = [s2.loc[s2["combo"].idxmax(), "y"] for _, sub in pool.groupby("trade_date")
         for s2 in [sub.dropna(subset=["combo"])] if len(s2)]
    res["8因子等权(方向按234轮)"] = float(np.mean(v))

    # 随机
    rng = np.random.default_rng(235)
    sim = []
    for _ in range(500):
        v = [sub["y"].values[rng.integers(len(sub))] for _, sub in pool.groupby("trade_date")]
        sim.append(np.mean(v))
    res["随机选一只(500次均值)"] = float(np.mean(sim))
    res["随机选一只(500次最好)"] = float(np.max(sim))

    out = pd.DataFrame({"评分": list(res), "单笔%": list(res.values())})
    out["vs池均值"] = out["单笔%"] - res["随机选一只(500次均值)"]
    out = out.sort_values("单笔%", ascending=False)
    print("\n" + out.to_string(index=False))
    out.to_csv(OUT / "w235b_scores.csv", index=False, encoding="utf-8-sig")
    print(f"\n-> {OUT / 'w235b_scores.csv'}")


if __name__ == "__main__":
    main()
