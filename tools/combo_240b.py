"""第二百四十轮 B: 宽覆盖条件能不能天天补位 —— 这才是"可用"与"彩票"的分界

240 在基准空仓日上挖, 除了 4 个窄条件(idx_dist<-3.5 的暴跌日), 还冒出几个**覆盖 50~95% 空仓日**
的宽条件: intraday 最弱 + 相对抗跌, 测试段 568 日、单笔 1.83%、胜率 61%。
如果它成立, 补位就不是"一年 6 次的彩票", 而是"几乎每天都有事干"。

本轮只算一件事: 这些条件在 2021+ 的**测试段**补位后, 年化/回撤/卡玛是多少。
选股规则同时扫三种(等权全买 / 市值最小 3 只 / 市值最小 1 只), 因为 3 万本金下
等权买 15 只每只 2000 元, 佣金最低 5 元 = 0.25% 摩擦, 不可忽略。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "output" / "feat234"
OUT = ROOT / "output"
TRADES = OUT / "w232_trades.csv"

COST = 0.10
COLS = ["ts_code", "trade_date", "y", "n1_ok", "pct", "rel_pct", "bias10", "bias20",
        "ret3", "ret5", "ret10", "ret20", "vol20", "maxdd20", "is_paused", "is_st",
        "listed", "circ_mv", "intraday", "close_pos", "dn_streak", "amp",
        "dist_high60", "rs20", "idx_dist", "low_tail"]
TRAIN_END, TEST_START = 2020, 2021


def load() -> pd.DataFrame:
    parts = []
    for p in sorted(FEAT.glob("f*.parquet")):
        d = pd.read_parquet(p, columns=COLS)
        d = d[(d["is_paused"] == 0) & (d["is_st"] == 0) & (d["n1_ok"] == True)]   # noqa: E712
        d = d[d["pct"].between(-11.5, 11.5) & d["y"].between(-40, 60) & (d["listed"] >= 250)]
        if len(d):
            parts.append(d)
    return pd.concat(parts, ignore_index=True)


def metrics(day_ret: pd.Series) -> tuple[float, float, float]:
    nav = (1 + day_ret / 100).cumprod()
    yrs = len(day_ret) / 244.0
    ann = nav.iloc[-1] ** (1 / yrs) - 1
    dd = (nav / nav.cummax() - 1).min()
    return round(ann * 100, 2), round(dd * 100, 2), round(ann * 100 / abs(dd * 100), 2)


def main() -> None:
    d = load()
    d["dn_x_amp"] = d["dn_streak"] * d["amp"]
    d["vol_x_pct"] = d["vol20"] * d["pct"]
    cal = np.sort(d["trade_date"].unique())
    cal = cal[cal >= 20160101]

    tr = pd.read_csv(TRADES)
    tr = tr[tr["signal"].isin(cal)]
    busy, sig = set(), set(tr["signal"])
    for _, r in tr.iterrows():
        busy.add(r["buy"]); busy.add(r["sell"])
    free = [x for x in cal if x not in busy and x not in sig]
    d = d[d["trade_date"].isin(free)].copy()

    # 分位只用训练段(2016-2020 空仓日)
    yr = (d["trade_date"] // 10000).astype(int)
    q = {}
    for c in ["intraday", "rel_pct", "ret3", "ret5", "close_pos", "vol_x_pct",
              "vol20", "idx_dist", "dn_x_amp", "rs20", "dist_high60", "circ_mv", "amp"]:
        v = d.loc[yr <= TRAIN_END, c].to_numpy("float32")
        q[c] = np.nanquantile(v[np.isfinite(v)], [0.2, 0.8])
    print("训练段分位(20%/80%):")
    for k, v in q.items():
        print(f"  {k}: {v[0]:.2f} / {v[1]:.2f}")

    C = {
        "W1 日内最弱+抗跌": lambda x: (x.intraday < q["intraday"][0]) & (x.rel_pct > q["rel_pct"][1]),
        "W2 3日强/5日弱": lambda x: (x.ret3 > q["ret3"][1]) & (x.ret5 < q["ret5"][0]),
        "W3 收在低点+高波动杀": lambda x: (x.close_pos < q["close_pos"][0]) & (x.vol_x_pct > q["vol_x_pct"][1]),
        "W4 高波动+大盘破位": lambda x: (x.vol20 > q["vol20"][1]) & (x.idx_dist < q["idx_dist"][0]),
        "W5 抗跌+跌停带量": lambda x: (x.rel_pct > q["rel_pct"][1]) & (x.dn_x_amp > q["dn_x_amp"][0]),
        "W6 20日弱+大盘破位": lambda x: (x.rs20 < q["rs20"][0]) & (x.idx_dist < q["idx_dist"][0]),
        "W7 距60日高最远+破位": lambda x: (x.dist_high60 < q["dist_high60"][0]) & (x.idx_dist < q["idx_dist"][0]),
    }

    # 基准(测试段)
    tr_te = tr[tr["signal"] >= TEST_START * 10000 + 101]
    tcal = [x for x in cal if x >= TEST_START * 10000 + 101]
    base = pd.Series(0.0, index=tcal)
    for dt, g in tr_te.groupby("signal"):
        base[dt] = g["ret"].mean()
    print(f"\n测试段 {tcal[0]}~{tcal[-1]} {len(tcal)} 日  基准 {len(tr_te)} 笔 单笔 {tr_te['ret'].mean():.3f}%")
    print(f"基准 年化{metrics(base)[0]}% 回撤{metrics(base)[1]}% 卡玛{metrics(base)[2]}")

    rows = []
    for name, fn in C.items():
        m = fn(d)
        s = d[m]
        s = s[s["trade_date"] >= TEST_START * 10000 + 101]
        if not len(s):
            continue
        for k, tag in ((0, "等权全买"), (3, "市值最小3只"), (1, "市值最小1只")):
            v = {}
            for dt, g in s.groupby("trade_date"):
                if k in (1, 3):
                    g2 = g.nsmallest(min(k, len(g)), "circ_mv")
                else:
                    g2 = g
                v[dt] = g2["y"].mean() - COST
            v = pd.Series(v)
            sim = base.copy()
            sim[v.index] = v.values
            ann, dd, cml = metrics(sim)
            rows.append({"条件": name, "选股": tag, "补位日": len(v),
                         "覆盖率%": round(len(v) / len(tcal) * 100, 0),
                         "单笔": round(v.mean(), 3), "年化": ann, "回撤": dd, "卡玛": cml})

    r = pd.DataFrame(rows)
    print("\n" + r.to_string(index=False))
    r.to_csv(OUT / "w240b_combo.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
