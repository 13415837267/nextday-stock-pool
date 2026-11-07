"""第 168 轮(二): seg3(T+2 日内) 是不是结构性衰减 —— 决定"尾盘卖"还剩多少寿命。

seg1 = T+1 日内(制度锁死, 吃不到)
seg2 = 隔夜
seg3 = T+2 日内  <- 尾盘卖相对开盘卖的唯一来源
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "output")


def main():
    tr = pd.DataFrame(pd.read_pickle(os.path.join(OUT, "main_trades_latest.pkl")))
    tr = tr[tr["ret_pct"].notna()].copy()
    tr["d"] = tr["signal_date"].astype(str).str.replace("-", "").str[:8]
    tr = tr.sort_values("d").reset_index(drop=True)

    # 成交记录只有买卖价, 三段的中间节点要用 panel 重算
    p = pd.read_pickle(os.path.join(ROOT, "data", "cache", "panel_all.pkl"))
    p = p.loc[p["is_paused"] == 0, ["ts_code", "d", "open", "close", "factor"]].copy()
    p["d"] = p["d"].astype(str).str.replace("-", "").str[:8]
    p["px_o"] = (p["open"].astype("float64") * p["factor"]).astype("float32")
    p["px_c"] = (p["close"].astype("float64") * p["factor"]).astype("float32")
    mo = p.pivot_table(index="d", columns="ts_code", values="px_o", aggfunc="first")
    mc = p.pivot_table(index="d", columns="ts_code", values="px_c", aggfunc="first")
    days = list(mo.index)
    pos = {d: i for i, d in enumerate(days)}
    cols = {c: j for j, c in enumerate(mo.columns)}

    s1 = np.full(len(tr), np.nan)
    s2 = np.full(len(tr), np.nan)
    s3 = np.full(len(tr), np.nan)
    for k, (c, d) in enumerate(zip(tr["code"].astype(str), tr["d"])):
        i, j = pos.get(d), cols.get(c)
        if i is None or j is None or i + 2 >= len(days):
            continue
        o1, c1, o2, c2 = mo.iat[i + 1, j], mc.iat[i + 1, j], mo.iat[i + 2, j], mc.iat[i + 2, j]
        if o1 > 0 and c1 > 0 and o2 > 0 and c2 > 0:
            s1[k], s2[k], s3[k] = c1 / o1 - 1, o2 / c1 - 1, c2 / o2 - 1
    tr["s1"], tr["s2"], tr["s3"] = s1 * 100, s2 * 100, s3 * 100
    tr["y"] = tr["d"].str[:4]
    a = tr.dropna(subset=["s3"])

    print("=== 一、分年: 收盘卖 vs 开盘卖(毛, 每笔均值 %) ===")
    rows = []
    for y, g in a.groupby("y"):
        close_r = ((1 + g.s1 / 100) * (1 + g.s2 / 100) * (1 + g.s3 / 100) - 1).mean() * 100
        open_r = ((1 + g.s1 / 100) * (1 + g.s2 / 100) - 1).mean() * 100
        rows.append(dict(y=y, n=len(g), seg1=g.s1.mean(), seg2=g.s2.mean(), seg3=g.s3.mean(),
                         close=close_r, open=open_r, diff=close_r - open_r))
    df = pd.DataFrame(rows)
    print(df.round(3).to_string(index=False))
    print(f"\n全样本: 收盘 {a.apply(lambda g: (1+g.s1/100)*(1+g.s2/100)*(1+g.s3/100)-1, axis=1).mean()*100:.3f}%"
          f"  开盘 {a.apply(lambda g: (1+g.s1/100)*(1+g.s2/100)-1, axis=1).mean()*100:.3f}%")

    print("\n=== 二、seg3 的滚动均值(窗口 40 笔) ===")
    a = a.reset_index(drop=True)
    roll = a["s3"].rolling(40).mean()
    # 每 40 笔抽查一次
    for i in range(39, len(a), 40):
        print(f"  第{a.loc[i-39,'d']}~{a.loc[i,'d']}  seg3 {roll.iloc[i]:+.3f}%  "
              f"(累计均 {a.loc[:i,'s3'].mean():+.3f}%)")
    # 按日期每半年
    print("\n=== 三、按半年 ===")
    _m = a["d"].str[4:6].astype(int)
    a["h"] = a["d"].str[:4] + np.where(_m <= 6, "H1", "H2")
    h = a.groupby("h").agg(n=("s3", "size"), s1=("s1", "mean"), s2=("s2", "mean"), s3=("s3", "mean"))
    h["close"] = a.groupby("h").apply(lambda g: ((1 + g.s1 / 100) * (1 + g.s2 / 100) * (1 + g.s3 / 100) - 1).mean() * 100)
    h["open"] = a.groupby("h").apply(lambda g: ((1 + g.s1 / 100) * (1 + g.s2 / 100) - 1).mean() * 100)
    print(h.round(3).to_string())

    print("\n=== 四、近两年 vs 前四年, seg3 差异 bootstrap ===")
    early = a[a["y"] <= "2024"]["s3"].to_numpy()
    late = a[a["y"] >= "2025"]["s3"].to_numpy()
    print(f"  2021-2024: n={len(early)} 均值 {early.mean():+.3f}%")
    print(f"  2025-2026: n={len(late)} 均值 {late.mean():+.3f}%")
    obs = late.mean() - early.mean()
    rng = np.random.default_rng(168)
    pool = np.concatenate([early, late])
    cnt = 0
    B = 8000
    for _ in range(B):
        rng.shuffle(pool)
        d = pool[len(early):].mean() - pool[:len(early)].mean()
        if d <= obs:
            cnt += 1
    print(f"  差 {obs:+.3f}pp  P(打乱后更小)={cnt/B:.4f}")

    print("\n=== 五、如果 T+1 日内也吃得到(T+0 假想上限) ===")
    t0 = ((1 + a.s1 / 100) - 1).mean() * 100
    print(f"  每笔 T+1 日内 {t0:+.3f}%  (现口径整笔 {((1+a.s1/100)*(1+a.s2/100)*(1+a.s3/100)-1).mean()*100:+.3f}%)")
    print(f"  叠加后单笔约 {((1+a.s1/100)*(1+a.s2/100)*(1+a.s3/100)-1).mean()*100 + t0:+.3f}%"
          f"  ≈ 现口径的 {(1 + t0/100):.2f} 倍")

    df.to_csv(os.path.join(OUT, "exp_seg_trend.csv"), index=False, encoding="utf-8-sig")
    h.to_csv(os.path.join(OUT, "exp_seg_half.csv"), encoding="utf-8-sig")
    print("\nsaved output/exp_seg_trend.csv, exp_seg_half.csv")


if __name__ == "__main__":
    main()
