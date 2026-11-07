# -*- coding: utf-8 -*-
"""第一百六十五轮: 龙虎榜能不能当"新 alpha 源"? —— 29,279 个(日,票)的全维度否证。

背景
----
第一百六十四轮的结论是"复利已到顶, 要再上台阶只能换 alpha 源(新数据/新市场)"。
项目里唯一买了却从未进入选股/回测的数据是 data/lhb(龙虎榜, 446 个交易日, 2025-01~2026-11),
本脚本把它一次性验完。

口径(无未来函数)
----------------
T 日收盘后交易所发布龙虎榜 → T+1 开盘买入 → T+2 开盘卖出(持有 1 个交易日)。
T+1 开盘一字涨停(开盘价 >= 涨停价)视为买不到, 剔除; 停牌/缺价剔除。

结论(见 CHANGELOG 第一百六十五轮)
----------------------------------
基准 -0.535%/笔; 29 个维度分组(上榜类型/涨跌幅 12 档/换手 5 档/机构/北向/游资/量化净额方向)
**全部为负**, 最好的一组 -0.285%, 无一能覆盖双边 0.13% 的成本。
唯一为正的子集是"北向净买 + 当日跌 >=5%"(n=1068, +0.163%), 扣成本后归零, 且 2026 已衰减到 +0.055%。

用法: python tools/exp_lhb.py
"""
import ast
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_seats(s):
    if not isinstance(s, str) or not s.strip():
        return []
    try:
        v = ast.literal_eval(s)
        return v if isinstance(v, list) else []
    except Exception:
        return []


def load_px(dates):
    px = {}
    for d in dates:
        f = os.path.join(ROOT, "data", "panel", f"{d}.csv.gz")
        if not os.path.exists(f):
            continue
        df = pd.read_csv(f, dtype={"ts_code": str})
        for row in df.itertuples(index=False):
            px[(row.ts_code, d)] = (row.open, row.close, row.high_limit)
    return px


def _tag(df):
    if "rcls" not in df.columns:
        df["rcls"] = df["reason"].map(lambda r: "跌幅榜" if "跌幅" in r else
                                      "涨幅榜" if ("涨幅" in r or "涨停" in r) else
                                      "换手榜" if "换手率" in r else
                                      "振幅榜" if "振幅" in r else "其他")
    if "y" not in df.columns:
        df["y"] = df["d"].str[:4]
    return df


def build():
    cache = os.path.join(ROOT, "data", "cache", "lhb_probe.pkl")
    if os.path.exists(cache):
        df = pd.read_pickle(cache)
        return _tag(df)
    fs = sorted(glob.glob(os.path.join(ROOT, "data", "lhb", "*.csv.gz")))
    lhb = pd.concat([pd.read_csv(f, dtype={"stock_code": str, "ts_code": str, "trade_date": str})
                     for f in fs], ignore_index=True)
    lhb = lhb[lhb["ts_code"].notna()]
    agg = defaultdict(lambda: dict(net=0.0, inst_b=0.0, inst_s=0.0, north_b=0.0, north_s=0.0,
                                   youzi_b=0.0, youzi_s=0.0, quant_b=0.0, quant_s=0.0,
                                   reasons=[], chg=None, turn=None))
    for row in lhb.itertuples(index=False):
        k = (row.trade_date, row.ts_code)
        a = agg[k]
        a["net"] += float(row.buy_in) if pd.notna(row.buy_in) else 0.0
        a["chg"], a["turn"] = row.quote_change, row.turnover_ratio
        a["reasons"].append(str(row.up_reason))
        for side, key in ((row.buy_group_icons, "b"), (row.sell_group_icons, "s")):
            for st in parse_seats(side):
                nm, tags = str(st.get("name") or ""), str(st.get("tags") or "")
                amt = float(st.get("amount") or 0)
                if "机构" in nm:
                    a[f"inst_{key}"] += amt
                elif "股通" in nm:
                    a[f"north_{key}"] += amt
                elif "量化" in nm:
                    a[f"quant_{key}"] += amt
                elif "游资" in tags or st.get("youzi_icon"):
                    a[f"youzi_{key}"] += amt

    all_dates = sorted({os.path.basename(f)[:8]
                        for f in glob.glob(os.path.join(ROOT, "data", "panel", "*.csv.gz"))})
    idx = {d: i for i, d in enumerate(all_dates)}
    px = load_px(sorted({d for d, _ in agg}))
    recs = []
    for (d, code), a in agg.items():
        i = idx.get(d)
        if i is None or i + 3 >= len(all_dates):
            continue
        d1, d2, d3 = all_dates[i + 1], all_dates[i + 2], all_dates[i + 3]
        p0, p1, p2, p3 = px.get((code, d)), px.get((code, d1)), px.get((code, d2)), px.get((code, d3))
        if not all([p0, p1, p2, p3]):
            continue
        o1 = p1[0]
        if not np.isfinite(o1) or o1 <= 0:
            continue
        if p1[2] and o1 >= p1[2] * 0.999:      # T+1 一字涨停, 买不到
            continue
        recs.append(dict(d=d, code=code, chg=a["chg"], turn=a["turn"], net=a["net"],
                         inst=a["inst_b"] - a["inst_s"], north=a["north_b"] - a["north_s"],
                         youzi=a["youzi_b"] - a["youzi_s"], quant=a["quant_b"] - a["quant_s"],
                         reason=";".join(a["reasons"]),
                         r1=p1[1] / o1 - 1, r2=p2[0] / o1 - 1, r3=p3[0] / o1 - 1))
    df = _tag(pd.DataFrame(recs))
    df.to_pickle(cache)
    return df


def main():
    df = build()
    print(f"样本 {len(df)} 个(日,票), {df['d'].nunique()} 个交易日, {df['y'].value_counts().to_dict()}")
    print(f"基准(T+1 开盘买 → T+2 开盘卖): 均 {df.r2.mean()*100:+.3f}%  "
          f"中位 {df.r2.median()*100:+.3f}%  胜率 {(df.r2>0).mean()*100:.1f}%")
    for c in ["r1", "r3"]:
        print(f"  {c}: 均 {df[c].mean()*100:+.3f}%  (仅对照, r1 当日买卖不可交易)")

    out = []
    def rep(label, s):
        n = len(s)
        if n < 30:
            return
        m, se = s.mean(), s.std(ddof=1) / np.sqrt(n)
        out.append(dict(group=label, n=n, mean=round(m * 100, 3),
                        med=round(s.median() * 100, 3), win=round((s > 0).mean() * 100, 1),
                        t=round(m / se, 2)))

    print("\n=== 上榜类型 ===")
    for k, g in df.groupby("rcls"):
        rep(f"类型:{k}", g.r2)
    print("\n=== 当日涨跌幅 12 档 ===")
    df["cb"] = pd.cut(df.chg, [-100, -9.8, -9.5, -7, -5, -3, 0, 3, 5, 7, 9.5, 9.8, 100],
                      labels=["≤-9.8", "-9.8~-9.5", "-9.5~-7", "-7~-5", "-5~-3", "-3~0",
                              "0~3", "3~5", "5~7", "7~9.5", "9.5~9.8", "≥9.8"])
    for k, g in df.groupby("cb", observed=True):
        rep(f"涨跌:{k}", g.r2)
    print("\n=== 换手率 5 档 ===")
    df["tb"] = pd.qcut(df.turn, 5, duplicates="drop")
    for k, g in df.groupby("tb", observed=True):
        rep(f"换手:{str(k)}", g.r2)
    print("\n=== 席位净额方向 ===")
    for k, name in [("inst", "机构"), ("north", "北向"), ("youzi", "游资"), ("quant", "量化")]:
        for sg, lab in [(1, "净买"), (-1, "净卖")]:
            rep(f"{name}:{lab}", df.loc[np.sign(df[k]) == sg, "r2"])

    t = pd.DataFrame(out)
    print(t.to_string(index=False))
    print(f"\n共 {len(t)} 组, 正期望 {int((t['mean'] > 0).sum())} 组, "
          f"扣双边成本 0.13% 后为正 {int((t['mean'] > 0.13).sum())} 组")
    json.dump(out, open(os.path.join(ROOT, "output", "exp_lhb.json"), "w"),
              ensure_ascii=False, indent=1)
    print("saved output/exp_lhb.json")


if __name__ == "__main__":
    main()
