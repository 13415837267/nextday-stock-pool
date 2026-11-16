"""第243轮 A: 单吊补位的持有期与止损 —— 242 轮只测了"固定 T+2 卖"

242 轮把补位做成单吊满仓: 年化 152.7% / 回撤 -45.2% / 卡玛 3.38, 但卖出端一刀切"T+2 收盘"。
破位日买的是超跌反弹, 反弹寿命可能只有 1 天; 也可能需要 3 天。这个从没在单吊口径下测过。

本脚本在完全相同的 192 次触发上, 只换卖出规则:
  S1  T+1 收盘卖      y_1d
  S2  T+2 收盘卖      y        (= 242 轮基准)
  S3  T+3 收盘卖      y_3d
  S4  T+2 卖但 T+1 已亏则 T+1 走   if y_1d<0 then y_1d else y
  S5  T+1 收盘亏超 -6% 就走        if y_1d<-6 then y_1d else y
  S6  T+1 收盘赚超 +5% 就止盈      if y_1d>5 then y_1d else y
  S7  T+2 与 T+1 取差(只在 T+1 很强时提前走) 同 S6 的镜像

另外测: 触发日是否必须"次日低开"(n1_ok)。242 轮沿用了主策略的 n1_ok=True,
但破位日高开高走的概率可能更大 —— 放宽后笔数会变, 这是单吊下唯一能扩样本的旋钮。
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "output" / "feat234"
OUT = ROOT / "output"
COST = 0.10

COLS = ["ts_code", "trade_date", "y", "y_1d", "y_3d", "n1_ok", "n1_gap", "pct",
        "circ_mv", "idx_dist", "dist_high60", "amt_yi", "vol20", "amp",
        "close_pos", "rel_pct", "is_paused", "is_st", "listed"]


def load(require_low_open: bool = True) -> pd.DataFrame:
    parts = []
    for p in sorted(FEAT.glob("f*.parquet")):
        d = pd.read_parquet(p, columns=COLS)
        d = d[(d.is_paused == 0) & (d.is_st == 0) & (d.listed >= 250)]
        d = d[d.pct.between(-11.5, 11.5) & d.y.between(-40, 60)]
        d = d[d.ts_code.str[:2].isin(["60", "00"])]
        if require_low_open:
            d = d[d.n1_ok == True]
        if len(d):
            parts.append(d)
    return pd.concat(parts, ignore_index=True)


def M(r: pd.Series):
    nav = (1 + r / 100).cumprod()
    ann = nav.iloc[-1] ** (244 / len(r)) - 1
    dd = (nav / nav.cummax() - 1).min()
    return round(ann * 100, 2), round(dd * 100, 2), round(ann * 100 / abs(dd * 100), 2)


def build(d: pd.DataFrame):
    """返回 (日历, 基准日收益, 补位日字典)"""
    cal = np.sort(d.trade_date.unique())
    tr = pd.read_csv(OUT / "w232_trades.csv")
    cal = cal[cal >= tr.signal.min()]
    tr = tr[tr.signal.isin(cal)]
    busy = set()
    for _, r in tr.iterrows():
        busy.add(r.buy)
        busy.add(r.sell)
    sig = set(tr.signal)
    free = set(x for x in cal if x not in busy and x not in sig)
    base = pd.Series(0.0, index=cal)
    for dt, g in tr.groupby("signal"):
        base[dt] = g["ret"].mean()
    nxt = {cal[i]: cal[i + 1] for i in range(len(cal) - 1)}
    nxt[cal[-1]] = cal[-1]
    s = d[(d.idx_dist <= -3.51) & (d.dist_high60 <= -24.46)]
    days = []
    for dt in sorted(s.trade_date.unique()):
        if dt not in free:
            continue
        bd = nxt.get(dt)
        sd = nxt.get(bd, bd)
        if bd in busy or sd in busy:
            continue
        days.append(dt)
    sub = {dt: g for dt, g in d[d.trade_date.isin(days)].groupby("trade_date")}
    pick = {}
    for dt, g in sub.items():
        g2 = g.dropna(subset=["dist_high60"]).sort_values(
            ["dist_high60", "amt_yi"], ascending=[True, False])
        if len(g2):
            pick[dt] = g2.iloc[0]
    P = pd.DataFrame(pick.values())
    P["signal"] = list(pick.keys())
    return cal, base, P


def report(tag: str, v: pd.Series, base: pd.Series, b, w=1.0):
    sim = base.copy()
    sim[v.index] += v.values * w
    a, dd, cm = M(sim)
    run = mx = 0
    for x in v.values:
        run = run + 1 if x < 0 else 0
        mx = max(mx, run)
    return {"规则": tag, "次数": len(v), "单笔": round(v.mean(), 3),
            "胜率%": round((v > 0).mean() * 100, 1), "最差": round(v.min(), 1),
            "年化": a, "回撤": dd, "卡玛": cm, "Δ卡玛": round(cm - b[2], 2),
            "最长连亏": mx}


def main():
    for req in (True, False):
        d = load(req)
        cal, base, P = build(d)
        b = M(base)
        print("=" * 100)
        print(f"{'次日必须低开(242轮口径)' if req else '不限制次日开盘方向(放宽)'}"
              f"  补位 {len(P)} 次  基准 年化{b[0]}% 回撤{b[1]}% 卡玛{b[2]}")
        if not req:
            g = P.n1_gap
            print(f"  n1_gap 分布: 中位 {g.median():.2f}%  >0 占 {(g > 0).mean() * 100:.0f}%  "
                  f"分位 10/90 = {g.quantile(.1):.1f}/{g.quantile(.9):.1f}")
        rows = []
        y1, y2, y3 = P.y_1d, P.y, P.y_3d
        V = {
            "S1 T+1收盘卖": y1,
            "S2 T+2收盘卖 (242基准)": y2,
            "S3 T+3收盘卖": y3,
            "S4 T+1亏则早走": np.where(y1 < 0, y1, y2),
            "S5 T+1亏超-6%才走": np.where(y1 < -6, y1, y2),
            "S6 T+1赚超+5%止盈": np.where(y1 > 5, y1, y2),
            "S7 T+1赚超+3%止盈": np.where(y1 > 3, y1, y2),
            "S8 T+1亏超-3%才走": np.where(y1 < -3, y1, y2),
        }
        for k, arr in V.items():
            v = pd.Series(np.asarray(arr) - COST, index=P["signal"].values)
            for w in ((1.0, 0.3) if k.startswith("S2") else (1.0,)):
                r = report(k + ("" if w == 1 else f" @{w:.0%}仓"), v, base, b, w)
                rows.append(r)
        r = pd.DataFrame(rows)
        print(r.to_string(index=False))
        r.to_csv(OUT / f"w243a_hold_{'lowopen' if req else 'any'}.csv",
                 index=False, encoding="utf-8-sig")
        if req:
            # 分年一致性: 只对前 5 个主流法
            print("\n分年单笔 (%):")
            t = pd.DataFrame({"yr": P["signal"].values // 10000})
            for k, arr in list(V.items())[:5]:
                t[k] = np.asarray(arr) - COST
            print(t.groupby("yr").mean().round(2).to_string())
        print()


if __name__ == "__main__":
    main()
