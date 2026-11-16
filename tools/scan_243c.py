"""第243轮 C: 合规最短持有 —— T+1 开盘买、T+2 开盘卖（只过一晚）

★ 必须先纠正 243A 的错误:
  243A 里"S1 T+1 收盘卖"用的是 y_1d = (T+1 收盘 / T+1 开盘 - 1),
  那是**当天买当天卖**, A 股 T+1 制度不允许。那个 5.11 的卡玛不可执行, 作废。

合规的时点组合只有:
  H1  T+1 开盘买 -> T+2 开盘卖   持 1 晚, 资金次日开盘即回笼(每天可滚动)
  H2  T+1 开盘买 -> T+2 收盘卖   持 1 天 1 晚 (= 现行口径, config sell_open_frac=0)
  H3  T+1 开盘买 -> T+3 开盘卖   持 2 晚
  H4  T+1 收盘买 -> T+2 收盘卖   持 1 天(不在集合竞价买)

本脚本用 data/panel 的 factor 重算 n2_open / n2_close, 对
  ① 主策略 559 笔  ② 补位 192 次
分别算四个时点的单笔/年化/回撤/卡玛。
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data" / "panel"
FEAT = ROOT / "output" / "feat234"
OUT = ROOT / "output"
COST = 0.10
CACHE = OUT / "w243c_px.parquet"


def build_px() -> pd.DataFrame:
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    parts = []
    for p in sorted(PANEL.glob("*.csv.gz")):
        try:
            d = pd.read_csv(p, usecols=["ts_code", "trade_date", "open", "close", "factor"])
        except Exception:
            continue
        if not len(d):
            continue
        d["f"] = d["factor"].fillna(1.0)
        d["ao"] = d["open"] * d["f"]
        d["ac"] = d["close"] * d["f"]
        parts.append(d[["ts_code", "trade_date", "ao", "ac"]])
    px = pd.concat(parts, ignore_index=True)
    px = px.sort_values(["ts_code", "trade_date"], kind="mergesort").reset_index(drop=True)
    g = px.groupby("ts_code", sort=False)
    px["ao1"] = g["ao"].shift(-1)
    px["ao2"] = g["ao"].shift(-2)
    px["ao3"] = g["ao"].shift(-3)
    px["ac1"] = g["ac"].shift(-1)
    px["ac2"] = g["ac"].shift(-2)
    px["ac3"] = g["ac"].shift(-3)
    px = px[["ts_code", "trade_date", "ao1", "ac1", "ao2", "ac2", "ao3", "ac3"]]
    px.to_parquet(CACHE, index=False, compression="zstd")
    print(f"价格缓存 -> {CACHE}  {len(px):,} 行")
    return px


def M(r: pd.Series):
    nav = (1 + r / 100).cumprod()
    ann = nav.iloc[-1] ** (244 / len(r)) - 1
    dd = (nav / nav.cummax() - 1).min()
    return round(ann * 100, 2), round(dd * 100, 2), round(ann * 100 / abs(dd * 100), 2)


def labels(px: pd.DataFrame, code: str, sig: int):
    """T=sig 日收盘选股; 返回四个合规时点的收益(%)"""
    r = px[(px.ts_code == code) & (px.trade_date == sig)]
    if not len(r):
        return None
    r = r.iloc[0]
    a1, c1, a2, c2, a3, c3 = r.ao1, r.ac1, r.ao2, r.ac2, r.ao3, r.ac3
    if not all(pd.notna(x) and x > 0 for x in (a1, a2)):
        return None
    out = {
        "H1_开1开2": (a2 / a1 - 1) * 100,
        "H2_开1收2": (c2 / a1 - 1) * 100 if pd.notna(c2) and c2 > 0 else np.nan,
        "H3_开1开3": (a3 / a1 - 1) * 100 if pd.notna(a3) and a3 > 0 else np.nan,
    }
    if pd.notna(c1) and c1 > 0 and pd.notna(c2) and c2 > 0:
        out["H4_收1收2"] = (c2 / c1 - 1) * 100
    else:
        out["H4_收1收2"] = np.nan
    return out


def main():
    px = build_px()
    print(f"价格表 {len(px):,} 行\n")

    # ---------- ① 主策略 559 笔 ----------
    tr = pd.read_csv(OUT / "w232_trades.csv")
    rows = []
    for r in tr.itertuples():
        L = labels(px, r.code, int(r.signal))
        if L:
            L["signal"] = int(r.signal)
            L["ret_csv"] = r.ret
            rows.append(L)
    T = pd.DataFrame(rows)
    print(f"主策略 559 笔, 能取到全部时点 {len(T)} 笔")
    cal = np.sort(T.signal.unique())
    print(f"  校验: 用 panel 重算 H2 单笔 {T['H2_开1收2'].mean():.3f}%  vs  成交明细 ret {T['ret_csv'].mean() * 100:.3f}%")

    print("\n" + "=" * 92)
    print("① 主策略 559 笔: 四个合规时点 (笔数不变, 每天一笔)")
    rr = []
    for k in ["H1_开1开2", "H2_开1收2", "H3_开1开3", "H4_收1收2"]:
        s = T.dropna(subset=[k])
        # 净值: 只在有成交的日子记收益, 其余日 0
        ser = pd.Series(0.0, index=np.arange(len(s)))
        ser[:] = s[k].values - COST
        a, dd, cm = M(ser)
        rr.append({"时点": k, "笔数": len(s), "单笔": round(ser.mean(), 3),
                   "胜率%": round((ser > 0).mean() * 100, 1),
                   "最差": round(ser.min(), 1), "年化": a, "回撤": dd, "卡玛": cm})
    print(pd.DataFrame(rr).to_string(index=False))

    # ---------- ② 补位 192 次 ----------
    print("\n" + "=" * 92)
    print("② 补位 192 次: 四个合规时点 (只看补位本身的单笔与风险)")
    picks = pd.read_csv(OUT / "w242b_picks.csv") if (OUT / "w242b_picks.csv").exists() else None
    # 重算补位选股(与 242 轮同规则), 需要 code
    FEAT_COLS = ["ts_code", "trade_date", "y", "n1_ok", "pct", "idx_dist", "dist_high60",
                 "amt_yi", "is_paused", "is_st", "listed"]
    parts = []
    for p in sorted(FEAT.glob("f*.parquet")):
        d = pd.read_parquet(p, columns=FEAT_COLS)
        d = d[(d.is_paused == 0) & (d.is_st == 0) & (d.listed >= 250)]
        d = d[d.pct.between(-11.5, 11.5) & d.y.between(-40, 60)]
        d = d[d.ts_code.str[:2].isin(["60", "00"])]
        d = d[d.n1_ok == True]
        if len(d):
            parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    cal2 = np.sort(d.trade_date.unique())
    cal2 = cal2[cal2 >= tr.signal.min()]
    busy = set()
    for _, r in tr.iterrows():
        busy.add(r.buy)
        busy.add(r.sell)
    sig = set(tr.signal)
    free = set(x for x in cal2 if x not in busy and x not in sig)
    nxt = {cal2[i]: cal2[i + 1] for i in range(len(cal2) - 1)}
    nxt[cal2[-1]] = cal2[-1]
    s = d[(d.idx_dist <= -2.0) & (d.dist_high60 <= -24.46)]
    days = []
    for dt in sorted(s.trade_date.unique()):
        if dt not in free:
            continue
        bd = nxt.get(dt)
        sd = nxt.get(bd, bd)
        if bd in busy or sd in busy:
            continue
        days.append(dt)
    rec = []
    for dt in days:
        g = d[(d.trade_date == dt) & d.dist_high60.notna()]
        if not len(g):
            continue
        g = g.sort_values(["dist_high60", "amt_yi"], ascending=[True, False])
        rec.append({"signal": dt, "code": g.iloc[0].ts_code})
    Q = pd.DataFrame(rec)
    print(f"  补位日 {len(Q)} 次 (idx_dist<=-2.0)")
    rows = []
    for r in Q.itertuples():
        L = labels(px, r.code, int(r.signal))
        if L:
            L["signal"] = int(r.signal)
            L["code"] = r.code
            rows.append(L)
    B = pd.DataFrame(rows)
    rr = []
    for k in ["H1_开1开2", "H2_开1收2", "H3_开1开3", "H4_收1收2"]:
        s2 = B.dropna(subset=[k])
        v = s2[k] - COST
        rr.append({"时点": k, "次数": len(v), "单笔": round(v.mean(), 3),
                   "胜率%": round((v > 0).mean() * 100, 1),
                   "最差": round(v.min(), 1), "最好": round(v.max(), 1),
                   "标准差": round(v.std(), 2)})
    print(pd.DataFrame(rr).to_string(index=False))
    B.to_csv(OUT / "w243c_picks_px.csv", index=False, encoding="utf-8-sig")
    T.to_csv(OUT / "w243c_main_px.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
