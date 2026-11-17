"""第244轮 A: 构造多排序窗口的候选池缓存

243 轮把排序窗口定在 250 日, 但那是**在 11 年全样本上从 9 个窗口里挑卡玛最高**的
(5.39 vs 5.05/5.44/5.08...), 仓位 70% 也是从 6 档里挑的, 冷却 20 日同理。
这是 239 轮已经踩过一次的坑: 全史选参数 vs 纯样本外, 卡玛差 1.45。

本轮先把 9 个窗口 × 5 档冷却的选股序列全部算出来缓存, 再做:
  ① 参数联合面(热力图): 250/70/20 到底落在平台上还是孤岛上
  ② 滚动 walk-forward: 每年初只用过去数据选参数 -> 该年应用 -> 拼接真样本外净值
  ③ 随机参数对照: 每年随机选 -> 分布, 看"用历史选参数"是否真有信息量

产出 output/w244_pool/W{win}.parquet: trade_date, ts_code, dh, y, amt_yi
"""
from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data" / "panel"
FEAT = ROOT / "output" / "feat234"
OUT = ROOT / "output"
POOL = OUT / "w244_pool"

WINS = (60, 90, 120, 150, 180, 200, 250, 300, 360)
COLS = ["ts_code", "trade_date", "y", "n1_ok", "pct", "idx_dist", "dist_high60", "amt_yi",
        "is_paused", "is_st", "listed"]
IDX_DIST_MAX = -3.51
DIST_HIGH_MAX = -24.46


def main() -> None:
    POOL.mkdir(parents=True, exist_ok=True)

    # ---------- 1. 信号日 + 当日候选票 (与 243L 同口径) ----------
    parts = []
    for p in sorted(FEAT.glob("f*.parquet")):
        d = pd.read_parquet(p, columns=COLS)
        d = d[(d.is_paused == 0) & (d.is_st == 0) & (d.listed >= 250)]
        d = d[d.pct.between(-11.5, 11.5) & d.y.between(-40, 60)]
        d = d[d.ts_code.str[:2].isin(["60", "00"])]
        d = d[d.n1_ok == True]                      # noqa: E712
        if len(d):
            d["trade_date"] = d["trade_date"].astype("int64")
            parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    del parts
    gc.collect()
    print(f"面板(清洗后) {d.trade_date.min()}~{d.trade_date.max()} {len(d):,} 行")

    # 主策略占用日 -> 补位只能用 free 日
    tr = pd.read_csv(OUT / "w232_trades.csv")
    cal = np.sort(d.trade_date.unique())
    cal = cal[cal >= tr.signal.min()]
    tr = tr[tr.signal.isin(cal)]
    busy = set()
    for _, r in tr.iterrows():
        busy.add(r.buy)
        busy.add(r.sell)
    sig = set(tr.signal)
    free = set(int(x) for x in cal if x not in busy and x not in sig)

    trig = d[(d.idx_dist <= IDX_DIST_MAX) & (d.dist_high60 <= DIST_HIGH_MAX)]
    days = sorted(int(x) for x in trig.trade_date.unique() if int(x) in free)
    print(f"触发日(可用) {len(days)} 个   {days[0]}~{days[-1]}")

    day_set = set(days)
    keep = d[d.trade_date.isin(day_set)][["ts_code", "trade_date", "y", "amt_yi"]].copy()
    del d, trig
    gc.collect()
    print(f"候选票 {len(keep):,} 行 / {len(day_set)} 天")

    # ---------- 2. 一次读 panel, 逐窗口算"距前 W 日最高"(不含当日) ----------
    px_parts = []
    for p in sorted(PANEL.glob("*.csv.gz")):
        try:
            x = pd.read_csv(p, usecols=["ts_code", "trade_date", "high", "close", "factor"])
        except Exception:                            # noqa: BLE001
            continue
        if not len(x):
            continue
        f = x["factor"].fillna(1.0)
        px_parts.append(pd.DataFrame({
            "ts_code": x["ts_code"].astype("category"),
            "trade_date": x["trade_date"].astype("int64"),
            "ah": (x["high"] * f).astype("float32"),
            "ac": (x["close"] * f).astype("float32")}))
    px = pd.concat(px_parts, ignore_index=True)
    del px_parts
    gc.collect()
    px = px.sort_values(["ts_code", "trade_date"], kind="mergesort").reset_index(drop=True)
    print(f"价格表 {len(px):,} 行")

    # 只保留信号日, 减少后续 merge 成本(rolling 仍需全序列, 但求完即丢)
    mask = px["trade_date"].isin(day_set).values
    ah_shift = px.groupby("ts_code", sort=False)["ah"].shift(1)

    for W in WINS:
        hm = ah_shift.groupby(px["ts_code"], sort=False).rolling(
            W, min_periods=max(W // 3, 40)).max()
        dh = ((px["ac"] / hm.reset_index(level=0, drop=True).values - 1) * 100).astype("float32")
        sub = px.loc[mask, ["ts_code", "trade_date"]].copy()
        sub["dh"] = dh[mask]
        sub["ts_code"] = sub["ts_code"].astype(str)
        sub = sub.dropna(subset=["dh"])
        sub.to_parquet(POOL / f"W{W}.parquet", index=False)
        print(f"  W={W:>3}: {len(sub):,} 行")
        del hm, dh, sub
        gc.collect()

    keep.to_parquet(POOL / "signals.parquet", index=False)
    print("完成")


if __name__ == "__main__":
    main()
