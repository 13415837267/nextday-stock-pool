"""第 168 轮: 把 104% 拆开 —— 三段收益分解 + 双差 placebo(选股 vs 择时)。

口径与主策略逐位一致: T 日信号 -> T+1 开盘买 -> T+2 收盘卖, 全复权。
输出三段: seg1=T+1 日内, seg2=隔夜(T+1收->T+2开), seg3=T+2 日内。
placebo 四格:
  A 真实(原日期 x 原票)
  B 原日期 x 全市场等权      -> 日期本身的 beta
  C 原票 x 随机日期          -> 票本身(选股)
  D 随机日期 x 全市场等权    -> 全样本市场平均
双差: 择时 = B - D; 选股 = C - D; 交互 = A - B - C + D
"""
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "data", "cache")
OUT = os.path.join(ROOT, "output")
RNG = np.random.default_rng(20261107)


def norm(d):
    return str(d).replace("-", "")[:8]


def build_matrix():
    t = time.time()
    p = pd.read_pickle(os.path.join(CACHE, "panel_all.pkl"))
    p = p.loc[p["is_paused"] == 0, ["ts_code", "d", "open", "close", "factor"]].copy()
    p["d"] = p["d"].map(norm)
    p["px_o"] = (p["open"].astype("float64") * p["factor"]).astype("float32")
    p["px_c"] = (p["close"].astype("float64") * p["factor"]).astype("float32")
    mo = p.pivot_table(index="d", columns="ts_code", values="px_o", aggfunc="first")
    mc = p.pivot_table(index="d", columns="ts_code", values="px_c", aggfunc="first")
    print(f"矩阵 {mo.shape} 用时 {time.time()-t:.0f}s", flush=True)
    return mo, mc


def segments(mo, mc, days, pos, cols, codes, ds):
    """返回每笔的 seg1/seg2/seg3 (小数)。"""
    n = len(codes)
    s1 = np.full(n, np.nan)
    s2 = np.full(n, np.nan)
    s3 = np.full(n, np.nan)
    o, c = mo, mc
    for k, (code, d) in enumerate(zip(codes, ds)):
        i = pos.get(d)
        j = cols.get(code)
        if i is None or j is None or i + 2 >= len(days):
            continue
        o1, c1 = o.iat[i + 1, j], c.iat[i + 1, j]
        o2, c2 = o.iat[i + 2, j], c.iat[i + 2, j]
        if not (o1 > 0 and c1 > 0 and o2 > 0 and c2 > 0):
            continue
        s1[k] = c1 / o1 - 1
        s2[k] = o2 / c1 - 1
        s3[k] = c2 / o2 - 1
    return s1, s2, s3


def fwd_of(mo, mc, days, pos, cols, codes, ds):
    o, c = mo, mc
    n = len(codes)
    r = np.full(n, np.nan)
    for k, (code, d) in enumerate(zip(codes, ds)):
        i = pos.get(d)
        j = cols.get(code)
        if i is None or j is None or i + 2 >= len(days):
            continue
        o1, c2 = o.iat[i + 1, j], c.iat[i + 2, j]
        if o1 > 0 and c2 > 0:
            r[k] = c2 / o1 - 1
    return r


def main():
    mo, mc = build_matrix()
    days = list(mo.index)
    pos = {d: i for i, d in enumerate(days)}
    cols = {c: j for j, c in enumerate(mo.columns)}

    tr = pd.DataFrame(pd.read_pickle(os.path.join(OUT, "main_trades_latest.pkl")))
    tr = tr[tr["ret_pct"].notna()].copy()
    tr["d"] = tr["signal_date"].map(norm)
    print(f"成交 {len(tr)} 笔  {tr['d'].min()} ~ {tr['d'].max()}", flush=True)

    codes = tr["code"].astype(str).tolist()
    ds = tr["d"].tolist()
    s1, s2, s3 = segments(mo, mc, days, pos, cols, codes, ds)
    ok = ~np.isnan(s1)
    print(f"三段可算 {ok.sum()} / {len(tr)}", flush=True)

    total = (1 + s1) * (1 + s2) * (1 + s3) - 1
    print("\n=== 一、三段分解(毛, 未扣费) ===")
    print(f"合计      均值 {np.nanmean(total)*100:+.3f}%  中位 {np.nanmedian(total)*100:+.3f}%")
    print(f"  seg1 T+1日内  均值 {np.nanmean(s1)*100:+.3f}%  中位 {np.nanmedian(s1)*100:+.3f}%  胜率 {(s1[ok]>0).mean()*100:.1f}%")
    print(f"  seg2 隔夜     均值 {np.nanmean(s2)*100:+.3f}%  中位 {np.nanmedian(s2)*100:+.3f}%  胜率 {(s2[ok]>0).mean()*100:.1f}%")
    print(f"  seg3 T+2日内  均值 {np.nanmean(s3)*100:+.3f}%  中位 {np.nanmedian(s3)*100:+.3f}%  胜率 {(s3[ok]>0).mean()*100:.1f}%")
    # 对数分解(可加)
    l1, l2, l3 = np.log1p(s1[ok]), np.log1p(s2[ok]), np.log1p(s3[ok])
    tot = l1 + l2 + l3
    print(f"  对数份额: seg1 {l1.sum()/tot.sum()*100:.1f}%  seg2 {l2.sum()/tot.sum()*100:.1f}%  seg3 {l3.sum()/tot.sum()*100:.1f}%")

    print("\n=== 二、三段的分年(均值 %) ===")
    yr = pd.Series([d[:4] for d in ds])[ok]
    tab = pd.DataFrame({"seg1": s1[ok], "seg2": s2[ok], "seg3": s3[ok], "tot": total[ok], "y": yr.values})
    g = tab.groupby("y")[["seg1", "seg2", "seg3", "tot"]].mean() * 100
    g["n"] = tab.groupby("y").size()
    print(g.round(3).to_string())

    print("\n=== 三、placebo: 原日期 x 全市场等权 (B) ===")
    # 每日全市场等权 fwd (同一口径)
    r_all = mc.shift(-2) / mo.shift(-1) - 1.0
    r_all = r_all.where((r_all.abs() <= 0.6) & (mo.shift(-1) > 0) & (mc.shift(-2) > 0))
    daily_mkt = r_all.mean(axis=1)
    mkt_by_d = daily_mkt.to_dict()
    b = np.array([mkt_by_d.get(d, np.nan) for d in ds])
    print(f"  当日全市场等权 均值 {np.nanmean(b)*100:+.3f}%  (n={np.sum(~np.isnan(b))})")

    print("\n=== 四、placebo: 原票 x 随机日期 (C) ===")
    # 对该票在样本期内随机取一个可算日
    o_arr, c_arr = mo.to_numpy(), mc.to_numpy()
    col_of = np.array([cols.get(c, -1) for c in codes])
    trials = 200
    crand = np.full((trials, len(codes)), np.nan, dtype=np.float32)
    for t in range(trials):
        # 每笔独立抽一个 i, 要求 i+2 < len(days)
        ii = RNG.integers(0, len(days) - 2, size=len(codes))
        oo = o_arr[ii + 1, np.clip(col_of, 0, None)]
        oc = c_arr[ii + 1, np.clip(col_of, 0, None)]
        cc = c_arr[ii + 2, np.clip(col_of, 0, None)]
        r = np.where((oo > 0) & (oc > 0) & (cc > 0), cc / oo - 1, np.nan)
        crand[t] = np.where(col_of >= 0, r, np.nan).astype(np.float32)
    c_mean_per_trial = np.nanmean(crand, axis=1)
    print(f"  200 次重采样, 每次均值分布: 中位 {np.median(c_mean_per_trial)*100:+.3f}%  "
          f"5%~95% [{np.percentile(c_mean_per_trial,5)*100:+.3f}%, {np.percentile(c_mean_per_trial,95)*100:+.3f}%]")

    print("\n=== 五、双差分解 ===")
    A = np.nanmean(total) * 100
    Bm = np.nanmean(b) * 100
    Cm = float(np.median(c_mean_per_trial)) * 100
    D = np.nanmean(daily_mkt.values) * 100
    print(f"  A 真实(原日期x原票)      {A:+.3f}%")
    print(f"  B 原日期 x 全市场等权    {Bm:+.3f}%")
    print(f"  C 原票 x 随机日期        {Cm:+.3f}%")
    print(f"  D 随机 x 随机(市场平均)  {D:+.3f}%")
    print(f"  --> 择时(日期) = B - D   {Bm-D:+.3f}pp")
    print(f"  --> 选股(票本身) = C - D {Cm-D:+.3f}pp")
    print(f"  --> 交互(日期x票)        {A-Bm-Cm+D:+.3f}pp")
    print(f"  --> 合计超额 = A - D     {A-D:+.3f}pp")

    res = {
        "n": int(ok.sum()), "seg1": float(np.nanmean(s1) * 100), "seg2": float(np.nanmean(s2) * 100),
        "seg3": float(np.nanmean(s3) * 100), "total": A,
        "log_share": {"s1": float(l1.sum() / tot.sum()), "s2": float(l2.sum() / tot.sum()),
                      "s3": float(l3.sum() / tot.sum())},
        "placebo": {"A_true": A, "B_date_mkt": Bm, "C_stock_rand": Cm, "D_mkt": D,
                    "timing": Bm - D, "selection": Cm - D, "interaction": A - Bm - Cm + D,
                    "excess": A - D},
        "c_trials_p5": float(np.percentile(c_mean_per_trial, 5) * 100),
        "c_trials_p95": float(np.percentile(c_mean_per_trial, 95) * 100),
        "yearly": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in g.round(4).to_dict("index").items()},
    }
    import json
    json.dump(res, open(os.path.join(OUT, "exp_decomp.json"), "w"), ensure_ascii=False, indent=1)
    print("\nsaved output/exp_decomp.json")


if __name__ == "__main__":
    main()
