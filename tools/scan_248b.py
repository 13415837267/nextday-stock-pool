"""第248轮 B: 生成补位「全量候选」—— 不做任何资金排他

247 的口径是 busy = {signal, buy, sell} 三天全排, 补位只能捡三天都不碰的日子。
但主策略是 T+1 开盘买、T+2 收盘卖, 资金占用区间只有 [buy 开盘, sell 收盘]。
- sell 日: 收盘卖出, 资金 T+1 开盘即可用 → 不该排 (补位正好 T+1 开盘买)
- 只有 [buy, sell] 与补位 [T+1, T+2] 有交集时才真冲突

这一步先不谈调度, 把「如果钱随便给」的全部候选铺出来, 之后的调度实验共用。
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
WINS = (60, 90, 120, 150, 180, 200, 250, 300, 360)
DHC = [f"dh{W}" for W in WINS]
DH = -35.0
N_DEEP = 10


def main() -> None:
    print("=" * 78)
    print("第248轮 B: 全量候选")
    print("=" * 78)

    d = pd.read_parquet(OUT / "w246b_all.parquet",
                        columns=["ts_code", "trade_date", "listed", "n1_gap", "y", "amt_yi"] + DHC)
    d = d[d.listed >= 250]
    d["trade_date"] = d["trade_date"].astype("int64")
    mk = pd.read_csv(OUT / "feat234" / "market.csv", usecols=["trade_date", "idx_ma20_dist"])
    d = d.merge(mk, on="trade_date", how="inner")
    d = d.dropna(subset=DHC + ["n1_gap", "y", "idx_ma20_dist"])
    print(f"候选池 {len(d):,} 行  {d.trade_date.nunique()} 天")

    parts = []
    for _yr, g in d.groupby(d.trade_date // 10000, sort=True):
        den = g[DHC].notna().sum()
        mr = (g[DHC].rank(ascending=True) / den).mean(axis=1, skipna=True)
        parts.append(g.assign(mr=mr))
    full = pd.concat(parts, ignore_index=True)
    del parts, d
    full = full.sort_values(["trade_date", "mr", "amt_yi"], ascending=[True, True, False])
    full["rk"] = full.groupby("trade_date").cumcount() + 1
    gb = {int(k): g for k, g in full.groupby("trade_date")}
    print(f"有票日 {len(gb)}")

    rows = []
    state = []
    for dt in sorted(gb):
        g = gb[dt]
        t1 = g.iloc[0]
        broke = bool(t1.idx_ma20_dist <= -3.51 and t1.dh60 <= DH)
        state.append({"trade_date": dt, "idx_ma20_dist": float(t1.idx_ma20_dist),
                      "top1_dh60": float(t1.dh60), "broken": broke,
                      "n_pool": int(len(g)), "med_dh60": float(g.dh60.median())})
        if broke:
            cc = g[g.n1_gap <= 0.05]
            path = "破位"
        else:
            cc = g[(g.n1_gap > -9.5) & (g.n1_gap <= -5.0) & (g.rk <= N_DEEP)]
            path = "深低开"
        if not len(cc):
            continue
        p = cc.iloc[0]
        chain = "|".join(f"{a}:{b:.4f}" for a, b in zip(cc.ts_code.head(8), cc.y.head(8)))
        rows.append({"trade_date": dt, "ts_code": p.ts_code, "y": p.y,
                     "n1_gap": float(p.n1_gap), "brk": broke, "path": path,
                     "idx": float(t1.idx_ma20_dist), "dh60": float(t1.dh60),
                     "rk": int(p.rk), "alt_n": int(len(cc)),
                     "alt_y": float(cc.y.mean()) if len(cc) > 1 else np.nan,
                     "chain": chain})
    c = pd.DataFrame(rows)
    c["yr"] = c.trade_date // 10000
    c.to_csv(OUT / "w248_cands.csv", index=False, encoding="utf-8-sig")
    stt = pd.DataFrame(state)
    stt.to_csv(OUT / "w248_daily_state.csv", index=False, encoding="utf-8-sig")
    print(f"破位日全集 {int(stt.broken.sum())} / {len(stt)} ({stt.broken.mean()*100:.1f}%)")
    print(f"全量候选 {len(c)} 天 (破位 {int(c.brk.sum())} / 深低开 {int((~c.brk).sum())})")
    print(f"  破位   单笔 {(c[c.brk].y - 0.30).mean():.2f}%")
    print(f"  深低开 单笔 {(c[~c.brk].y - 0.30).mean():.2f}%")

    # 与主策略的占用冲突
    tr = pd.read_csv(OUT / "w232_trades.csv")
    cal = sorted(int(f.split("/")[-1].replace(".csv.gz", ""))
                 for f in glob.glob(str(ROOT / "data" / "panel" / "*.csv.gz")))
    idx = {d_: i for i, d_ in enumerate(cal)}
    tr["bi"] = tr.buy.map(idx)
    tr["si"] = tr.sell.map(idx)
    tr = tr.dropna(subset=["bi", "si"])
    print(f"\n主策略 {len(tr)} 笔, 平均持仓 {int((tr.si - tr.bi).mean())} 个交易日")
    # 冲突: 补位 [T+1, T+2] 与主策略 [buy, sell] 有交集
    c["ci"] = c.trade_date.map(idx)
    c = c.dropna(subset=["ci"])
    c["ci"] = c.ci.astype(int)
    busy_signal = set(tr.signal)
    busy_buy = set(tr.buy)
    busy_sell = set(tr.sell)
    c["hit_sig"] = c.trade_date.isin(busy_signal)
    c["hit_buy"] = c.trade_date.isin(busy_buy)
    c["hit_sell"] = c.trade_date.isin(busy_sell)
    # 精确冲突: 补位买入日(ci+1) 落在主策略 [bi, si] 内
    lo = np.full(len(cal), 10**9)
    for b, s in zip(tr.bi.astype(int), tr.si.astype(int)):
        lo[b] = min(lo[b], b)
    occ = np.zeros(len(cal), dtype=bool)
    for b, s in zip(tr.bi.astype(int), tr.si.astype(int)):
        occ[b:s + 1] = True
    c["conflict"] = [occ[min(x + 1, len(cal) - 1)] for x in c.ci]
    print(f"  候选日命中 signal {int(c.hit_sig.sum())} / buy {int(c.hit_buy.sum())} / "
          f"sell {int(c.hit_sell.sum())}")
    print(f"  按 247 粗口径(三天全排) 剩余 {int((~(c.hit_sig|c.hit_buy|c.hit_sell)).sum())}")
    print(f"  按精确口径(买入日落在持仓区间) 剩余 {int((~c.conflict).sum())}  → 多释放 "
          f"{int((~c.conflict).sum()) - int((~(c.hit_sig|c.hit_buy|c.hit_sell)).sum())} 天")
    c.to_csv(OUT / "w248_cands.csv", index=False, encoding="utf-8-sig")

    print("\n【重叠日: 主策略 vs 补位】")
    mr = tr.set_index("signal")[["ret", "code"]]
    ov = c[c.hit_sig].copy()
    ov["m_ret"] = ov.trade_date.map(mr.ret)
    ov["m_code"] = ov.trade_date.map(mr.code)
    ov["m_c"] = ov.m_ret - 0.30
    ov["p_c"] = ov.y - 0.30
    print(f"  同日都想要的 {len(ov)} 天")
    print(f"    主策略 单笔 {ov.m_c.mean():>5.2f}%  胜率 {(ov.m_c>0).mean()*100:.0f}%  "
          f"合计 {ov.m_c.sum():>7.1f}pp")
    print(f"    补位   单笔 {ov.p_c.mean():>5.2f}%  胜率 {(ov.p_c>0).mean()*100:.0f}%  "
          f"合计 {ov.p_c.sum():>7.1f}pp")
    print(f"    补位更强 {(ov.p_c > ov.m_c).mean()*100:.0f}% 的天; 差均值 {(ov.p_c-ov.m_c).mean():+.2f}pp")
    ov.to_csv(OUT / "w248_overlap.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
