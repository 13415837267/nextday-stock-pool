"""第246轮 F: 严谨版 —— 并列触发(破位 or 个股深低开) + 排除主策略日 + 冷却

246e 的净值算错了口径(没排除主策略持仓日, 也没冷却), 但信号本身很硬:
「低开 <= -5%」的 70 天在 12 年里有 11 年为正(2017 无样本), 合计 377pp,
其中「大盘没破位、只是这只票自己崩了」的 50 来笔, 单笔 4.30%,
2015-2026 十年来每年都出现, 不是股灾专属。

本轮把它做成能实盘的版本:
  - 只在主策略空仓日做(资金不重叠, 与 245 一致)
  - 冷却 10 日(与 245 定稿一致): 冷却后换同一天的第二只, 全被冷却才退回
  - 排序用 dh250(全量面板只有单窗口; 243 轮测过 9 窗口平均只多 0.1 卡玛)
  - 滑点 0.2

触发/执行分离(都能实盘):
  9:25 看「当天低开组里 dh250 最小的那只」是否触发 -> 今天做不做
  做的话, 买「冷却过滤后的第一只」

三个方案:
  A  245 口径: 只在指数 <= -3.5% 时买
  B  246 并列: 指数 <= -3.5%  或  当天最超跌那只低开 <= -5%
  C  只买深低开: 只看个股低开 <= -5%, 不管指数
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
COST = 0.10
SLIP = 0.20


def metrics(r: pd.Series) -> tuple[float, float, float]:
    nav = (1 + r / 100).cumprod()
    ann = nav.iloc[-1] ** (244 / len(r)) - 1
    dd = (nav / nav.cummax() - 1).min()
    return round(ann * 100, 2), round(dd * 100, 2), round(ann * 100 / abs(dd * 100), 2)


def main() -> None:
    tr = pd.read_csv(OUT / "w232_trades.csv")
    all_dates = sorted(int(p.stem.replace(".csv", "")) for p in
                       (ROOT / "data" / "panel").glob("*.csv.gz"))
    cal = np.array([x for x in all_dates if x >= int(tr.signal.min())])
    dayrank = {int(d): i for i, d in enumerate(cal)}
    base = pd.Series(0.0, index=cal)
    busy = set(int(x) for x in tr.signal)
    for _, r in tr.iterrows():                          # ★ 与 244g 一致: 买入/卖出日也算占用
        busy.add(int(r["buy"]))
        busy.add(int(r["sell"]))
    for dt, g in tr.groupby("signal"):
        base[int(dt)] = g["ret"].mean()
    print(f"主策略占用 {len(busy)} 天")

    d = pd.read_parquet(OUT / "w246b_all.parquet",
                        columns=["ts_code", "trade_date", "dh250", "dh60", "listed",
                                 "n1_gap", "y", "amt_yi", "circ_mv", "pct"])
    d = d.dropna(subset=["dh250", "dh60", "n1_gap", "y"])
    d = d[d.listed >= 250]                             # ★ 与 245 池子一致: 上市满一年
    d["trade_date"] = d["trade_date"].astype("int64")
    d = d[d.trade_date >= cal[0]]
    d = d[~d.trade_date.isin(busy)]                    # ★ 只在主策略空仓日
    mk = pd.read_csv(OUT / "feat234" / "market.csv",
                     usecols=["trade_date", "idx_ma20_dist", "breadth", "limit_dn",
                              "med_pct", "eq_ret", "amt_shrink"])
    d = d.merge(mk, on="trade_date", how="inner").dropna(subset=["idx_ma20_dist"])
    print(f"空仓日候选 {len(d):,} 行  {d.trade_date.nunique()} 天")

    lo = d[d.n1_gap <= 0.05].sort_values(
        ["trade_date", "dh250", "amt_yi"], ascending=[True, True, False])
    lo = lo[["ts_code", "trade_date", "dh250", "dh60", "n1_gap", "y", "amt_yi",
             "idx_ma20_dist"]].reset_index(drop=True)
    gb = {int(dt): g for dt, g in lo.groupby("trade_date")}
    print(f"低开组覆盖 {len(gb)} 天")

    def select(rule: str, cd: int) -> pd.DataFrame:
        last: dict[str, int] = {}
        out = []
        for dt in sorted(gb):
            g = gb[dt]
            top = g.iloc[0]
            broke = (top.idx_ma20_dist <= -3.5) and (top.dh60 <= -24.46)
            if rule == "A":                       # 245 口径: 指数破位 + 个股距60日高<=-24.46
                trig = broke
            elif rule == "D":                     # 只指数破位, 不要个股深度条件
                trig = top.idx_ma20_dist <= -3.5
            elif rule == "B":                     # 并列: 上面那条 or 个股自己崩出缺口
                trig = broke or (top.n1_gap <= -5)
            else:                                 # C: 只看个股低开 <= -5
                trig = top.n1_gap <= -5
            if not trig:
                continue
            i = dayrank[dt]
            pick = top
            if cd:
                for j in range(len(g)):
                    rr = g.iloc[j]
                    if rr.ts_code not in last or i - last[rr.ts_code] >= cd:
                        pick = rr
                        break
            last[pick.ts_code] = i
            out.append({"trade_date": dt, "ts_code": pick.ts_code, "y": pick.y,
                        "n1_gap": pick.n1_gap, "dh250": pick.dh250,
                        "idx_ma20_dist": top.idx_ma20_dist,
                        "dh60": top.dh60,
                        "deep": bool(top.n1_gap <= -5), "j": 0})
        return pd.DataFrame(out)

    def net(c: pd.DataFrame, P: float) -> tuple[float, float, float, pd.Series]:
        s = pd.Series((c["y"].values - COST - SLIP) * P, index=c.trade_date.values)
        s = s.sort_index()
        sim = base.copy()
        sim[s.index] = sim[s.index] + s.values
        return (*metrics(sim), s)

    a0, d0, c0 = metrics(base)
    print(f"\n仅主策略  年化{a0:.2f} 回撤{d0:.2f} 卡玛{c0:.2f}")
    print(f"\n{'方案':<30}{'笔数':>5}{'单笔':>8}{'年化':>9}{'回撤':>9}{'卡玛':>7}")
    store = {}
    for rule, name in (("A", "A 245口径(破位+深跌)"), ("D", "D 只破位(无深跌条件)"),
                       ("B", "B 并列(破位 or 低开<=-5)"), ("C", "C 仅深低开")):
        for cd in (0, 10, 20):
            c = select(rule, cd)
            if not len(c):
                continue
            for P in (0.50, 0.70):
                a, dd, cm, s = net(c, P)
                print(f"{name+' C='+str(cd):<30}{len(c):>5}"
                      f"{(s.mean() / P):>8.2f}{a:>9.2f}{dd:>9.2f}{cm:>7.2f}"
                      + ("" if P == 0.5 else "   70%仓"))
                store[(rule, cd, P)] = (c, a, dd, cm)
        print()

    cB = store[("B", 10, 0.50)][0].copy()
    cB["yr"] = cB.trade_date // 10000
    cB["y_net"] = cB.y - COST - SLIP
    print("【B 方案 分年 (冷却10)】")
    g = cB.groupby("yr").agg(n=("y_net", "size"), 单笔=("y_net", "mean"),
                             合计=("y_net", "sum"))
    g["胜率"] = cB.groupby("yr").y_net.apply(lambda x: (x > 0).mean() * 100).round(0)
    print(g.round(2).to_string())

    print("\n【B 里「指数没破位、纯个股崩」的那些天】")
    ex = cB[cB.idx_ma20_dist > -3.5]
    print(f"  {len(ex)} 笔  单笔 {ex.y_net.mean():.2f}%  胜率 {(ex.y_net>0).mean()*100:.0f}%"
          f"  合计 {ex.y_net.sum():.0f}pp")
    print(ex.groupby("yr").agg(n=("y_net", "size"), 单笔=("y_net", "mean")).round(2).T.to_string())

    print("\n【仓位扫描 (B, 冷却10)】")
    for P in (0.35, 0.50, 0.60, 0.70, 0.85, 1.00):
        a, dd, cm, s = net(cB, P)
        print(f"  {P:>5.0%}  年化{a:>7.2f}  回撤{dd:>7.2f}  卡玛{cm:>6.2f}")

    s = (cB.y - COST - SLIP)
    print(f"\n【极端依赖】合计 {s.sum():.0f}pp  最大5笔 {s.nlargest(5).sum():.0f}pp "
          f"({s.nlargest(5).sum()/s.sum()*100:.0f}%)  剔除最大5笔后单笔 "
          f"{s.drop(s.nlargest(5).index).mean():.2f}%")
    print(f"  票数 {cB.ts_code.nunique()}  最长连买 "
          f"{cB.groupby('ts_code').size().max()} 次")
    cB.to_csv(OUT / "w246f_picks.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
