"""第261轮b: 702 个"开盘没跌到位"的日子, 盘中到底给没给第二次机会

现状: 3200 天里 1140 天(42%)有候选, 但 T+1 开盘只跌不到 -2.5%, 不买。
这 1140 天是最大的空仓来源, 比"完全没候选"(901 天)还大。

一直以来的默认假设是"开盘没跌到位 = 今天没机会"。但开盘价只是第一笔,
盘中完全可能继续砸下去。若盘中最低价跌破 -2.5%, 就等于给了一张迟到的车票。

用日线 low 做上界估计(不抓分钟线):
  · 只统计 T+1 的 low 相对 T 日收盘 <= -2.5% 的日子(盘中触及过承接线)
  · 假设在 -2.5% 挂限价单成交(最乐观: 触及即成交, 无滑点)
  · 卖出仍是 T+2 收盘(与主策略一致)
这个上界如果都不赚钱, 这条路上限就是负的, 直接否掉; 若上界明显为正, 再抓分钟线验真实成交率。
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data/panel"


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    g = pickle.load(open(ROOT / f"output/w261_multi_{n}.pkl", "rb"))
    A = [x for x in g if x["name"].startswith("A ")][0]["trades"]
    sk = A[A.ret_pct.isna()].copy()
    sk["why"] = np.where(sk.note.fillna("").str.contains("缺口"), "gap",
                         np.where(sk.note.fillna("").str.contains("评分"), "score", "other"))
    sk = sk[sk.why.isin(["gap", "score"])]

    days = sorted(p.stem.replace(".csv", "") for p in PANEL.glob("*.csv.gz"))
    pos = {d: i for i, d in enumerate(days)}

    cache: dict[str, pd.DataFrame] = {}

    def load(d: str) -> pd.DataFrame | None:
        if d not in cache:
            f = PANEL / f"{d}.csv.gz"
            if not f.exists():
                cache[d] = None
            else:
                cache[d] = pd.read_csv(f, usecols=["ts_code", "open", "low", "high", "close",
                                                   "pre_close", "factor", "high_limit"])
        return cache[d]

    rows = []
    for r in sk.itertuples():
        bd = str(r.buy_date)
        i = pos.get(bd)
        if i is None or i + 1 >= len(days):
            continue
        d1, d2 = load(bd), load(days[i + 1])
        if d1 is None or d2 is None:
            continue
        m1 = d1[d1.ts_code == r.code]
        m2 = d2[d2.ts_code == r.code]
        if not len(m1) or not len(m2):
            continue
        a, b = m1.iloc[0], m2.iloc[0]
        pc = float(a.pre_close)
        if not pc or np.isnan(pc):
            continue
        gap = (float(a.open) / pc - 1) * 100
        lowp = (float(a.low) / pc - 1) * 100
        # 一字跌停: 买不进, 排除(与主策略口径一致)
        lim = float(a.high_limit) if not pd.isna(a.high_limit) else np.inf
        if float(a.open) >= lim - 1e-6 and float(a.high) <= lim + 1e-6:
            continue
        # 盘中 -2.5% 限价买入; 若开盘就低于 -2.5% 说明已属现状成交口径, 不算"第二次机会"
        if gap <= -2.5:
            continue
        buy_px = pc * 0.975
        # 卖出 T+2 收盘, 用 factor 对齐复权
        ret = (float(b.close) * float(b.factor) / (buy_px * float(a.factor)) - 1) * 100
        rows.append(dict(why=r.why, code=r.code, sd=r.signal_date, bd=bd,
                         gap=gap, low=lowp, ret=ret, score=r.score, pool_n=r.pool_n))

    x = pd.DataFrame(rows)
    print(f"样本: {len(x)} 条有候选但开盘未达线的记录 (总跳过 {len(sk)})")
    for k in ("gap", "score"):
        s = x[x.why == k]
        if not len(s):
            continue
        touch = s[s.low <= -2.5]
        print(f"\n== {k} 类  共 {len(s)} 条")
        print(f"   盘中触及 -2.5%: {len(touch)} 条 ({len(touch)/len(s)*100:.0f}%)")
        if len(touch):
            print(f"   若在 -2.5% 买入、T+2 收盘卖: 单笔 {touch.ret.mean():>6.2f}%  "
                  f"胜率 {(touch.ret>0).mean()*100:.0f}%  中位 {touch.ret.median():.2f}%")
            print(f"   分年: ", end="")
            touch = touch.assign(y=touch.bd.str[:4].astype(int))
            print(touch.groupby("y").ret.agg(["size", "mean"]).round(2).to_dict("index"))
        notouch = s[s.low > -2.5]
        if len(notouch):
            print(f"   盘中也没触及: {len(notouch)} 条  (这些才是真·没机会)")

    x.to_csv(ROOT / f"output/w261b_intraday_{n}.csv", index=False)


if __name__ == "__main__":
    main()
