"""第二百三十九轮 B: 样本内条件 vs 样本外条件, 在同一段(2019+)上补位对打

对照组设计(这是本轮的核心):
  A 组 = 第 237 轮那 6 个条件     —— 在**全史**上挖的, 对 2019+ 属于样本内
  B 组 = 第 239 轮挖出并存活的条件 —— 只用 2009~2018 挖, 对 2019+ 是纯样本外
两条都在同一段行情、同一个资金池、同一条"避开基准持仓日"的规则下补位。
如果 A 组明显赢 B 组, 说明 237 轮挖出来的东西是拟合; 若 B 组不输, 说明方向真实。
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

COST = 0.10
COLS = ["ts_code", "trade_date", "y", "n1_ok", "pct", "rel_pct", "bias5", "bias10",
        "bias20", "ret10", "maxdd20", "vol20", "is_paused", "is_st", "listed", "circ_mv"]

# A 组: 第 237 轮(全史挖)的 6 个
A_GROUP = {
    "A1 反弹中继": lambda d: (d.bias10 > 3.148) & (d.bias20 < -4.582),
    "A2 急杀后首日反弹": lambda d: (d.bias10 > 3.148) & (d.ret10 < -5.874),
    "A3 暴跌日抗跌": lambda d: (d.pct < -1.816) & (d.rel_pct > 1.287),
    "A4 抗跌+跌穿20日回撤": lambda d: (d.rel_pct > 1.287) & ((d.maxdd20 - d.pct) > 1.707),
    "A5 抗跌+高波动下跌": lambda d: (d.rel_pct > 1.287) & ((d.vol20 * d.pct) < -4.439),
    "A6 20日弱+10日涨": lambda d: (d.bias20 < -4.582) & d.ret10.between(-5.874, -1.72),
}
# B 组: 第 239 轮(只用 2009~2018 挖, 2019+ 样本外存活)
B_GROUP = {
    "B1 抗跌+高波动下跌": lambda d: (d.rel_pct > 1.26) & ((d.vol20 * d.pct) < -4.37),
    "B2 20日线附近+10日强": lambda d: d.bias20.between(-4.59, -1.35) & (d.ret10 > 6.60),
    "B3 5日弱+10日强": lambda d: (d.bias5 < -1.98) & (d.bias10 > 3.29),
}


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
    cal = np.sort(d["trade_date"].unique())
    cal = cal[cal >= 20190101]
    nxt = {cal[i]: cal[i + 1] for i in range(len(cal) - 1)}
    nxt[np.int64(cal[-1])] = np.int64(cal[-1])
    d = d[d["trade_date"].isin(cal)].copy()
    d["y_ex"] = d["y"] - d.groupby("trade_date")["y"].transform("mean")
    print(f"测试段 {cal[0]} ~ {cal[-1]}  {len(cal)} 交易日  样本 {len(d):,}")

    tr = pd.read_csv(TRADES)
    tr = tr[tr["signal"].isin(cal)]
    base = pd.Series(0.0, index=cal)
    for dt, g in tr.groupby("signal"):
        base[dt] = g["ret"].mean()
    busy = set()
    for _, r in tr.iterrows():
        busy.add(r["buy"]); busy.add(r["sell"])
    b_ann, b_dd, b_cml = metrics(base)
    print(f"基准 {len(tr)} 笔 单笔 {tr['ret'].mean():.3f}%  年化 {b_ann}%  回撤 {b_dd}%  卡玛 {b_cml}\n")

    rows = []
    for grp, conds in (("A 样本内(237)", A_GROUP), ("B 样本外(239)", B_GROUP)):
        for name, fn in conds.items():
            m = fn(d)
            s = d[m]
            if not len(s):
                continue
            # 条件自身(不补位, 直接看)
            own = {"组": grp, "条件": name, "条件内日": s["trade_date"].nunique(),
                   "条件内单笔": round(s["y"].mean(), 3),
                   "条件内alpha": round(s["y_ex"].mean(), 3)}
            # 补位
            pick_days = {}
            for dt in s["trade_date"].unique():
                if dt in busy:
                    continue
                bt, sl = nxt[dt], nxt[nxt[dt]]
                if bt in busy or sl in busy:
                    continue
                g = s[s["trade_date"] == dt]
                pick_days[dt] = g.loc[g["circ_mv"].idxmin(), "y"] - COST
            if not pick_days:
                continue
            sim = base.copy()
            ps = pd.Series(pick_days)
            # 时间上互斥: 同一时间段只保留一条(按日期顺序, 谁先命中谁占)
            sim[ps.index] = ps.values
            ann, dd, cml = metrics(sim)
            # 同一批补位日, 用等权代替市值最小
            pick_ew = {}
            for dt in pick_days:
                g = s[s["trade_date"] == dt]
                pick_ew[dt] = g["y"].mean() - COST
            sim2 = base.copy()
            p2 = pd.Series(pick_ew)
            sim2[p2.index] = p2.values
            ann2, dd2, cml2 = metrics(sim2)
            rows.append({**own, "补位笔": len(ps), "补位单笔": round(ps.mean(), 3),
                         "年化": ann, "回撤": dd, "卡玛": cml,
                         "等权年化": ann2, "等权回撤": dd2, "等权卡玛": cml2})

    r = pd.DataFrame(rows)
    print(r.to_string(index=False))
    r.to_csv(OUT / "w239b_combo.csv", index=False, encoding="utf-8-sig")

    print("\n基准:", b_ann, b_dd, b_cml)
    a = r[r["组"].str.startswith("A")]
    b = r[r["组"].str.startswith("B")]
    print(f"\nA 组(样本内) 补位单笔均值 {a['补位单笔'].mean():+.3f}%  卡玛均值 {a['卡玛'].mean():.2f}")
    print(f"B 组(样本外) 补位单笔均值 {b['补位单笔'].mean():+.3f}%  卡玛均值 {b['卡玛'].mean():.2f}")
    print(f"差: 单笔 {b['补位单笔'].mean()-a['补位单笔'].mean():+.3f}pp  "
          f"卡玛 {b['卡玛'].mean()-a['卡玛'].mean():+.2f}")


if __name__ == "__main__":
    main()
