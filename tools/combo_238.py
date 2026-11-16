"""第二百三十八轮: 补位回测 —— 现行空仓日能不能用 237 轮挖出来的条件填上

第 237 轮挖出 6 个条件, 但它们单独只有 1140~1670 天, 现行策略只占 4015 天的 14%。
本轮不换策略, 只回答一个问题: **在那 86% 的空仓日里补位, 年化/回撤/卡玛怎么变。**

口径与引擎一致:
  - 单只满仓, T 日收盘出信号 -> T+1 开盘买 -> T+2 收盘卖
  - 持仓期占用资金, 期间不能开新仓 (config: sell_open_frac=0 即收盘卖)
  - 成本: 佣金万2.5双边 + 印花万5 = 0.1%; 滑点 0 (第227轮机制事实)

基线不是重跑引擎, 而是直接重放 559 笔实际成交 (含 signal/buy/sell), 先验证
重放口径能对上官方年化, 再在空档里插补位票。
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

WINDOW = 2500
COST = 0.10  # % 往返

COLS = ["ts_code", "trade_date", "y", "n1_ok", "pct", "rel_pct", "bias10", "bias20",
        "ret10", "maxdd20", "vol20", "is_paused", "is_st", "listed", "circ_mv",
        "turnover_rate", "amp", "amt_yi"]

# 与第 237 轮挖掘结果完全一致的 6 个格子
CONDS = {
    "C1": lambda d: (d.bias10 > 3.148) & (d.bias20 < -4.582),
    "C2": lambda d: (d.bias10 > 3.148) & (d.ret10 < -5.874),
    "C3": lambda d: (d.pct < -1.816) & (d.rel_pct > 1.287),
    "C4": lambda d: (d.rel_pct > 1.287) & ((d.maxdd20 - d.pct) > 1.707),
    "C5": lambda d: (d.rel_pct > 1.287) & ((d.vol20 * d.pct) < -4.439),
    "C6": lambda d: (d.bias20 < -4.582) & d.ret10.between(-5.874, -1.72),
}


def load_all() -> pd.DataFrame:
    parts = []
    for p in sorted(FEAT.glob("f*.parquet")):
        d = pd.read_parquet(p, columns=COLS)
        d = d[(d["is_paused"] == 0) & (d["is_st"] == 0) & (d["n1_ok"] == True)]   # noqa: E712
        d = d[d["pct"].between(-11.5, 11.5) & d["y"].between(-40, 60) & (d["listed"] >= 250)]
        if len(d):
            parts.append(d)
    return pd.concat(parts, ignore_index=True)


def metrics(day_ret: pd.Series, label: str) -> dict:
    nav = (1 + day_ret / 100).cumprod()
    yrs = len(day_ret) / 244.0
    ann = nav.iloc[-1] ** (1 / yrs) - 1 if yrs > 0 else 0
    dd = (nav / nav.cummax() - 1).min()
    return {"方案": label, "年化%": round(ann * 100, 2), "回撤%": round(dd * 100, 2),
            "卡玛": round(ann * 100 / abs(dd * 100), 2) if dd else np.nan,
            "净值": round(nav.iloc[-1], 2), "交易日": len(day_ret)}


def main() -> None:
    d = load_all()
    cal = np.sort(d["trade_date"].unique())
    cal = cal[-WINDOW:]
    nxt = {cal[i]: cal[i + 1] for i in range(len(cal) - 1)}
    nxt[np.int64(cal[-1])] = np.int64(cal[-1])
    print(f"窗口 {cal[0]} ~ {cal[-1]}  交易日 {len(cal)}")

    d = d[d["trade_date"].isin(cal)].copy()
    print(f"窗口内可执行样本 {len(d):,}")

    tr = pd.read_csv(TRADES)
    tr = tr[tr["signal"].isin(cal)].copy()
    print(f"现行成交 {len(tr)} 笔  单笔 {tr['ret'].mean():.3f}%")

    # 基准: 逐日复利, 有仓日取 ret, 空仓日 0
    base_r = pd.Series(0.0, index=cal)
    for dt, g in tr.groupby("signal"):
        base_r[dt] = g["ret"].mean()
    # 占用日历: buy..sell 之间不可再开仓(引擎口径)
    busy = set()
    for _, r in tr.iterrows():
        busy.add(r["buy"]); busy.add(r["sell"])
    print(f"基准占用交易日 {len(busy)} / {len(cal)}")

    rows = [metrics(base_r, "基准(现行 559 笔)")]
    print(f"\n基准重放: 年化 {rows[0]['年化%']}%  回撤 {rows[0]['回撤%']}%  卡玛 {rows[0]['卡玛']}")

    # 逐条件构造补位候选
    for cname, fn in CONDS.items():
        m = fn(d)
        s = d[m]
        if not len(s):
            continue
        cand_days = [x for x in s["trade_date"].unique() if x in cal]
        # 补位日: 该日非基准有仓日, 且 T+1 与 T+2 未被基准占用
        ok_days = []
        for dt in cand_days:
            if dt in busy:
                continue
            b = nxt[dt]; sl = nxt.get(b, b)
            if b in busy or sl in busy:
                continue
            ok_days.append(dt)
        sub = s[s["trade_date"].isin(ok_days)]
        if not len(sub):
            print(f"{cname}: 无可用补位日")
            continue

        for mode in ("ew", "mv", "rel", "k3"):
            if mode == "ew":
                pick = sub.groupby("trade_date")["y"].mean()
                tag = f"+{cname} 全买等权"
            elif mode == "mv":
                pick = sub.loc[sub.groupby("trade_date")["circ_mv"].idxmin()].set_index("trade_date")["y"]
                tag = f"+{cname} 市值最小 1 只"
            elif mode == "rel":
                idx = sub.assign(a=sub["rel_pct"].abs()).groupby("trade_date")["a"].idxmax()
                pick = sub.loc[idx].set_index("trade_date")["y"]
                tag = f"+{cname} |rel| 最大 1 只"
            else:
                pick = sub.groupby("trade_date").apply(
                    lambda g: g.nsmallest(min(3, len(g)), "circ_mv")["y"].mean())
                tag = f"+{cname} 市值最小3只等权"
            add = pick.reindex(cal).fillna(np.nan) - COST
            sim = base_r.copy()
            fill = add.dropna()
            sim[fill.index] = fill.values
            r = metrics(sim, tag)
            r.update({"补位笔": len(fill), "补位单笔": round(fill.mean(), 3),
                      "补位总贡献pp": round((sim - base_r).sum(), 1)})
            rows.append(r)

    res = pd.DataFrame(rows)
    print("\n" + res.to_string(index=False))
    res.to_csv(OUT / "w238_combo.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
