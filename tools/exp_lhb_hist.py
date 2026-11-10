# -*- coding: utf-8 -*-
"""第一百八十八轮: 把龙虎榜补到 2016 年, 看"资金流"能不能做成独立的右侧腿。

为什么是这件事
--------------
第一百八十六/八十七轮把日线价量上的右侧(追突破/趋势回踩/涨停次日/放量/VCP)全测完了,
绝对天花板 22.5x / -62.6%, 全史单笔 0.27% —— 右侧的 edge 不是"买贵了", 是信号本身没期望。
第一百六十五轮已经用 2025-01 起的 1.5 年龙虎榜做过一次否证(29 个维度全是负的), 但那次样本
太短, 无法区分"龙虎榜没用"和"2025 年以后龙虎榜失效"。本轮把历史补到 2016-01(2400 个交易日,
data/lhb_hist), 用十年样本重做同一件事。

口径
----
T 日盘后交易所公布龙虎榜 → T+1 开盘买 → T+1+H 收盘卖(H 个交易日)。
T+1 开盘 ≥ 涨停价(一字)视为买不到; 停牌/缺价剔除; 往返成本 0.32%(滑点千1x2+佣金万2.5x2+
印花税万5, 2023-08 前略高, 忽略差异)。
因子全部只用 T 日盘后可知信息: 涨跌幅、净买额/成交额、龙虎榜成交额占比、换手、流通市值、上榜原因。

用法: python3 tools/exp_lhb_hist.py [--stage 1|2]
  stage1: 单维度分箱(找方向)
  stage2: 组合网格(找可交易子集)
"""
from __future__ import annotations

import glob
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COST = 0.32  # 往返成本 %


# ---------------------------------------------------------------- 数据
def load_lhb() -> pd.DataFrame:
    """两源合并: data/lhb_hist.parquet(2016~2025.06, 东财) + data/lhb(2025.05~, tushare)。

    东财源以单个 parquet 入库(2304 个日文件逐个推 git 会超时, 见 fetch_lhb_hist.py --pack),
    若 parquet 不存在则回退到目录读取。
    """
    rows = []
    pq = os.path.join(ROOT, "data", "lhb_hist.parquet")
    if os.path.exists(pq):
        a = pd.read_parquet(pq)
        a["code"] = a["code"].astype(str).str.zfill(6)
        a["date"] = a["date"].astype(str).str[:8]
        a = a.assign(src="em")
    else:
        for f in sorted(glob.glob(os.path.join(ROOT, "data", "lhb_hist", "*.csv.gz"))):
            d = pd.read_csv(f, dtype={"code": str})
            d["code"] = d["code"].str.zfill(6)
            rows.append(d[["code", "date", "pct", "net_buy", "lhb_amt", "mkt_amt",
                           "turn", "circ_mv", "reason"]].assign(src="em"))
        a = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
            columns=["code", "date", "pct", "net_buy", "lhb_amt", "mkt_amt",
                     "turn", "circ_mv", "reason", "src"])
    # tushare 源
    rows = []
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "lhb", "*.csv.gz"))):
        d = pd.read_csv(f, dtype={"stock_code": str, "trade_date": str})
        rows.append(pd.DataFrame({
            "code": d["stock_code"].str.zfill(6),
            "date": d["trade_date"].str[:8],
            "pct": pd.to_numeric(d.get("quote_change"), errors="coerce"),
            "net_buy": pd.to_numeric(d.get("buy_in"), errors="coerce"),
            "lhb_amt": np.nan,
            "mkt_amt": np.nan,
            "turn": pd.to_numeric(d.get("turnover_ratio"), errors="coerce"),
            "circ_mv": pd.to_numeric(d.get("capitalization"), errors="coerce"),
            "reason": d.get("up_reason"),
            "src": "ts",
        }))
    if rows:
        b = pd.concat(rows, ignore_index=True)
        a = pd.concat([a, b], ignore_index=True)
    a = a[a["code"].str.len() == 6]
    a["date"] = a["date"].astype(str).str.replace(r"\.0$", "", regex=True).str[:8]
    a = a.drop_duplicates(["code", "date"])
    a["ts_code"] = np.where(a["code"].str[0].isin(["6"]),
                            a["code"] + ".SH",
                            np.where(a["code"].str[:2].isin(["83", "87", "43", "92"]),
                                     a["code"] + ".BJ", a["code"] + ".SZ"))
    return a.reset_index(drop=True)


def price_matrices():
    fs = sorted(glob.glob(os.path.join(ROOT, "data", "panel", "*.csv.gz")))
    cols = ["ts_code", "trade_date", "open", "close", "high_limit", "low_limit",
            "amount", "is_paused"]
    frames = []
    for f in fs:
        d = pd.read_csv(f, dtype={"ts_code": str}, usecols=cols)
        frames.append(d)
    px = pd.concat(frames, ignore_index=True)
    px["trade_date"] = px["trade_date"].astype(str).str[:8]
    out = {}
    for c in ("open", "close", "high_limit", "low_limit", "amount", "is_paused"):
        out[c] = px.pivot_table(index="trade_date", columns="ts_code", values=c,
                                aggfunc="first").astype("float32")
    return out


# ---------------------------------------------------------------- 回测
def simulate(sig: pd.DataFrame, M: dict, H: int, at_open: bool = True) -> pd.DataFrame:
    """sig 需要 code / date / ts_code; 返回每笔净收益(%)。

    at_open=True  : T+1 开盘买 → T+1+H 开盘卖 (与主策略同口径, 默认 sell_at_open)
    at_open=False : T+1 开盘买 → T+1+H 收盘卖
    买不到: T+1 开盘一字涨停 / 停牌;  卖不掉: 卖出时点一字跌停 / 停牌 → 该笔剔除(保守)。
    """
    di = M["open"].index
    ci = M["open"].columns
    pos = pd.Series(np.arange(len(di)), index=di)
    cpos = pd.Series(np.arange(len(ci)), index=ci)
    d_i = pos.reindex(sig["date"]).to_numpy()
    c_i = cpos.reindex(sig["ts_code"]).to_numpy()
    ok = np.isfinite(d_i) & np.isfinite(c_i)
    d_i = np.where(ok, d_i, 0).astype(int)
    c_i = np.where(ok, c_i, 0).astype(int)
    n = len(di)
    ok &= (d_i + 1 + H) < n
    d_i = np.where(ok, d_i, 0)

    op = M["open"].to_numpy()
    cl = M["close"].to_numpy()
    hl = M["high_limit"].to_numpy()
    ll = M["low_limit"].to_numpy()
    pa = M["is_paused"].to_numpy()

    b_i = np.minimum(d_i + 1, n - 1)
    s_i = np.minimum(d_i + 1 + H, n - 1)
    buy = op[b_i, c_i]
    sell = (op if at_open else cl)[s_i, c_i]
    pause = (pa[b_i, c_i] == 1) | (pa[s_i, c_i] == 1)
    buy_limit = np.isfinite(hl[b_i, c_i]) & (buy >= hl[b_i, c_i] - 1e-6)
    sell_ll = np.isfinite(ll[s_i, c_i]) & (np.isfinite(sell) & (sell <= ll[s_i, c_i] + 1e-6))
    good = ok & np.isfinite(buy) & np.isfinite(sell) & (buy > 0) & ~pause & ~buy_limit
    if at_open:
        good &= ~sell_ll          # 开盘跌停卖不掉
    ret = (sell / buy - 1.0) * 100.0 - COST
    o = sig.copy()
    o["ret"] = np.where(good, ret, np.nan)
    o["buy_date"] = di.to_numpy()[b_i]
    o["sell_date"] = di.to_numpy()[s_i]
    return o[np.isfinite(o["ret"].to_numpy())]


def stat(r: pd.Series) -> str:
    n = len(r)
    if n == 0:
        return "n=0"
    m = r.mean()
    win = (r > 0).mean() * 100
    t = m / (r.std(ddof=1) / np.sqrt(n)) if n > 1 and r.std(ddof=1) > 0 else 0
    eq = (1 + r / 100).prod()
    return f"n={n:5d} 单笔{m:+7.3f}% 胜率{win:5.1f}% t{t:+6.2f} 累计{eq:9.2f}x"


def bucket(x: pd.Series, q: int) -> pd.Series:
    try:
        return pd.qcut(x, q, labels=False, duplicates="drop")
    except Exception:
        return pd.Series(np.nan, index=x.index)


def main() -> None:
    stage = 2
    if "--stage" in sys.argv:
        stage = int(sys.argv[sys.argv.index("--stage") + 1])

    lhb = load_lhb()
    print(f"龙虎榜记录 {len(lhb)} 条, {lhb['date'].nunique()} 个交易日, "
          f"{lhb['date'].min()} ~ {lhb['date'].max()}")
    print(lhb.groupby(lhb["date"].str[:4]).size().to_string())

    M = price_matrices()
    dates = M["open"].index
    print(f"价格面板 {len(dates)} 个交易日 x {M['open'].shape[1]} 只")

    # 因子: 净买额占当日成交额比重 / 龙虎榜成交额占比
    amt = M["amount"].stack().rename("amt").reset_index()
    amt.columns = ["date", "ts_code", "amt"]
    lhb = lhb.merge(amt, on=["date", "ts_code"], how="left")
    lhb["net_ratio"] = lhb["net_buy"] / lhb["amt"] * 100.0     # 净买额 / 全日成交额 %
    lhb["amt_ratio"] = lhb["lhb_amt"] / lhb["amt"] * 100.0     # 龙虎榜成交额占比 %
    lhb["mv_yi"] = lhb["circ_mv"] / 1e8                        # 流通市值(亿)
    lhb["rcls"] = lhb["reason"].fillna("").map(
        lambda r: "跌幅榜" if "跌幅" in r else
        "涨幅榜" if ("涨幅" in r or "涨停" in r) else
        "换手榜" if "换手率" in r else
        "振幅榜" if "振幅" in r else "其他")
    lhb["y"] = lhb["date"].str[:4]

    print("\n" + "=" * 100)
    print("【A】基准: 全部龙虎榜记录, 不同持有期")
    print("=" * 100)
    base = {}
    for H in (1, 2, 3, 5, 10):
        r = simulate(lhb, M, H)
        base[f"H{H}"] = [round(float(r["ret"].mean()), 4), int(len(r))]
        print(f"H{H:<2d} {stat(r['ret'])}")

    if stage < 2:
        return

    print("\n" + "=" * 100)
    print("【B】单维度分箱 (H1 / H3 / H5), 单笔净收益%")
    print("=" * 100)
    dims = [
        ("net_ratio", "净买额占成交额%", 10),
        ("amt_ratio", "龙虎榜成交占全日成交%", 5),
        ("turn", "换手率%", 5),
        ("mv_yi", "流通市值(亿)", 5),
        ("pct", "当日涨跌幅%", 8),
    ]
    for col, name, q in dims:
        print(f"\n-- {name}")
        sub = lhb[np.isfinite(lhb[col])]
        sub = sub.assign(g=bucket(sub[col], q))
        for g, gg in sub.groupby("g"):
            lo, hi = gg[col].min(), gg[col].max()
            line = f"  [{lo:9.2f},{hi:9.2f}]"
            for H in (1, 3, 5):
                rr = simulate(gg, M, H)["ret"]
                line += f" | H{H} n={len(rr):4d} {rr.mean():+6.2f}%" if len(rr) else f" | H{H} -"
            print(line)

    print("\n-- 上榜原因分类")
    for g, gg in lhb.groupby("rcls"):
        line = f"  {g}"
        for H in (1, 3, 5):
            rr = simulate(gg, M, H)["ret"]
            line += f" | H{H} n={len(rr):5d} {rr.mean():+6.2f}%"
        print(line)

    print("\n" + "=" * 100)
    print("【C】按年度: 龙虎榜效应是不是消失了")
    print("=" * 100)
    year = {}
    for y, gg in lhb.groupby("y"):
        line = f"  {y}"
        rr1 = simulate(gg, M, 1)["ret"]
        year[y] = [round(float(rr1.mean()), 4), int(len(rr1))]
        for H in (1, 3, 5):
            rr = simulate(gg, M, H)["ret"]
            line += f" | H{H} n={len(rr):5d} {rr.mean():+6.2f}%" if len(rr) else f" | H{H} -"
        print(line)
    with open(os.path.join(ROOT, "output", "lhb_hist_188.json"), "w", encoding="utf-8") as f:
        json.dump({"base": base, "year_H1": year,
                   "span": [lhb["date"].min(), lhb["date"].max(), int(len(lhb))]},
                  f, ensure_ascii=False, indent=1)
    print("\n已写 output/lhb_hist_188.json")

    if stage < 3:
        return

    print("\n" + "=" * 100)
    print("【D】组合网格: 找出唯一的正期望子集 (H1, 按单笔排序)")
    print("=" * 100)
    pcts = [("涨>5", lhb["pct"] > 5), ("涨>9", lhb["pct"] > 9),
            ("跌<-3", lhb["pct"] < -3), ("跌<-5", lhb["pct"] < -5), ("跌<-9", lhb["pct"] < -9),
            ("全部", lhb["pct"].notna())]
    nets = [("净买>0", lhb["net_ratio"] > 0), ("净买>3", lhb["net_ratio"] > 3),
            ("净买<-3", lhb["net_ratio"] < -3), ("净买任意", lhb["net_ratio"].notna())]
    mvs = [("小30亿", lhb["mv_yi"] < 30), ("小50亿", lhb["mv_yi"] < 50), ("市值任意", lhb["mv_yi"].notna())]
    res = []
    for pn, pm in pcts:
        for nn, nm in nets:
            for mn, mm in mvs:
                sig = lhb[pm & nm & mm]
                if len(sig) < 300:
                    continue
                r = simulate(sig, M, 1)["ret"]
                if len(r) < 300:
                    continue
                res.append((r.mean(), len(r), (r > 0).mean() * 100, pn, nn, mn))
    res.sort(reverse=True)
    print(f"{'单笔':>8} {'笔数':>7} {'胜率':>6}  条件   [T+1开→T+2开, 与主策略同口径]")
    for m, n, w, pn, nn, mn in res[:16]:
        print(f"{m:+8.3f}% {n:7d} {w:5.1f}%  {pn} & {nn} & {mn}")
    print("  ... 最差 3 组")
    for m, n, w, pn, nn, mn in res[-3:]:
        print(f"{m:+8.3f}% {n:7d} {w:5.1f}%  {pn} & {nn} & {mn}")

    print("\n" + "=" * 100)
    print("【E】最优子集的年度稳定性 (跌<-9 & 净买>0 & 市值<30亿)")
    print("=" * 100)
    best = lhb[(lhb["pct"] < -9) & (lhb["net_ratio"] > 0) & (lhb["mv_yi"] < 30)]
    for at_open in (True, False):
        tag = "T+1开→T+2开" if at_open else "T+1开→T+2收"
        r = simulate(best, M, 1, at_open=at_open)
        print(f"\n{tag}: {stat(r['ret'])}")
        for y, gg in r.groupby(r["date"].str[:4]):
            rr = gg["ret"]
            print(f"  {y} n={len(rr):4d} 单笔{rr.mean():+6.2f}% 胜率{(rr > 0).mean() * 100:5.1f}% "
                  f"累计{(1 + rr / 100).prod():6.2f}x")


if __name__ == "__main__":
    main()
