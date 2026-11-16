"""第二百三十七轮 C: 给挖出来的条件扣掉市场 beta

第 237 轮挖出 6 个条件, 单笔 1.3~2.4%, 但它们可能只是在"大盘暴跌日"买入,
赚的是市场反弹(=beta), 不是选股。这一版把每笔的当日全市场均值扣掉(y_ex),
看还剩多少。同时给分年、与现行策略重叠、条件间重叠。

y_ex = y - 当日 817 万样本的 y 均值。这等价于"同一天随便买一只的收益"。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "output" / "feat234"
OUT = ROOT / "output"
TRADES = OUT / "w232_trades.csv"

COLS = ["ts_code", "trade_date", "y", "n1_ok", "pct", "rel_pct", "bias10", "bias20",
        "ret10", "maxdd20", "vol20", "circ_mv", "amt_yi", "turnover_rate", "close_pos",
        "amp", "is_paused", "is_st", "listed",
        "bias60", "ret5", "ret20", "dist_high20", "dist_high60", "dn_streak", "amt_ratio",
        "gap", "bias5"]


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
    d = pd.concat(parts, ignore_index=True)
    # 与 mine_237 完全一致的交互衍生特征
    d["mv_x_turn"] = d["circ_mv"] * d["turnover_rate"]
    d["amt_x_turn"] = d["amt_yi"] / d["turnover_rate"].clip(lower=0.1)
    d["b60_m_b20"] = d["bias60"] - d["bias20"]
    d["r5_m_r20"] = d["ret5"] - d["ret20"]
    d["h20_m_h60"] = d["dist_high20"] - d["dist_high60"]
    d["dd_m_pct"] = d["maxdd20"] - d["pct"]
    d["pos_x_tail"] = d["close_pos"] * d["close_pos"]
    d["vol_x_pct"] = d["vol20"] * d["pct"]
    d["dn_x_amp"] = d["dn_streak"] * d["amp"]
    d["amp_x_turn"] = d["amp"] * d["turnover_rate"]
    d["gap_x_vr"] = d["gap"] / (d["amt_ratio"].abs().clip(lower=0.2))
    d["ln_mv"] = np.log(d["circ_mv"].clip(lower=1))
    d["y_ex"] = d["y"] - d.groupby("trade_date")["y"].transform("mean")
    d["year"] = (d["trade_date"] // 10000).astype(int)
    d["key"] = d["ts_code"] + "_" + d["trade_date"].astype(str)
    return d


def mask_of(d: pd.DataFrame, c: dict) -> pd.Series:
    m = pd.Series(True, index=d.index)
    for f, lo, hi in ((c["f1"], c["lo1"], c["hi1"]), (c["f2"], c["lo2"], c["hi2"])):
        v = d[f]
        if lo is not None:
            m &= v >= lo
        if hi is not None:
            m &= v < hi
    return m


def main() -> None:
    conds = json.loads((OUT / "w237_top.json").read_text(encoding="utf-8"))
    d = load()
    print(f"可执行样本 {len(d):,}  全样本均值 {d['y'].mean():.3f}%  "
          f"y_ex 标准差 {d['y_ex'].std():.2f}%\n")

    tr = pd.read_csv(TRADES).rename(columns={"signal": "trade_date", "code": "ts_code"})
    tr["key"] = tr["ts_code"] + "_" + tr["trade_date"].astype(str)
    tk = set(tr["key"])

    rows = []
    keep_masks = {}
    for i, c in enumerate(conds, 1):
        name = f"C{i} {c['f1']} & {c['f2']}"
        m = mask_of(d, c)
        s = d[m]
        if len(s) < 100:
            continue
        keep_masks[name] = m
        ny = s.groupby("year")["y_ex"].agg(["size", "mean"])
        ny = ny[ny["size"] >= 40]
        # 市场环境: 条件日的当日均值(负=大盘跌)
        env = s.groupby("trade_date")["y"].mean()
        # 只买一只: 按条件内最"抗跌"(rel_pct 最大)
        one = s.loc[s.groupby("trade_date")["rel_pct"].idxmax()]
        rows.append({
            "cond": name, "n": len(s), "days": s["trade_date"].nunique(),
            "per_day": round(len(s) / s["trade_date"].nunique(), 1),
            "y": round(s["y"].mean(), 3), "y_ex": round(s["y_ex"].mean(), 3),
            "beta_share": round(1 - s["y_ex"].mean() / s["y"].mean(), 2) if s["y"].mean() else np.nan,
            "win": round((s["y"] > 0).mean() * 100, 1),
            "win_ex": round((s["y_ex"] > 0).mean() * 100, 1),
            "yrs_pos": f"{int((ny['mean'] > 0).sum())}/{len(ny)}",
            "worst_yr": round(float(ny["mean"].min()), 2) if len(ny) else np.nan,
            "overlap559": len(set(s["key"]) & tk),
            "one_y": round(one["y"].mean(), 3), "one_yex": round(one["y_ex"].mean(), 3),
            "mkt_env": round(float(env.mean()), 2),
        })
        ny.to_csv(OUT / f"w237c_C{i}.csv", encoding="utf-8-sig")

    r = pd.DataFrame(rows)
    pd.set_option("display.width", 260)
    pd.set_option("display.max_colwidth", 55)
    print(r.to_string(index=False))
    r.to_csv(OUT / "w237c_summary.csv", index=False, encoding="utf-8-sig")

    # 条件两两重叠
    if len(keep_masks) > 1:
        print("\n=== 条件两两重叠(Jaccard)")
        ks = list(keep_masks)
        jac = pd.DataFrame(index=ks, columns=ks, dtype=float)
        for a in ks:
            for b in ks:
                ia, ib = keep_masks[a].values, keep_masks[b].values
                inter = (ia & ib).sum()
                jac.loc[a, b] = round(inter / max((ia | ib).sum(), 1), 2)
        print(jac.to_string())


if __name__ == "__main__":
    main()
