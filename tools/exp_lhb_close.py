"""第166轮: 龙虎榜在"T+2 收盘卖"(主策略真实口径)下的重算 —— 第 165 轮用的是 T+2 开盘卖。

第 165 轮结论"30 个维度 0 组为正"是卖点口径造成的: config 里 sell_open_frac=0.0,
`u_o = units × 0`, 100% 按 T+2 收盘价成交。开盘卖切掉了 T+2 日内那一段, 而龙虎榜股的
日内漂移 (+1.0pp) 是市场平均 (+0.13pp) 的 8 倍 —— 于是"最该吃的肉"正好被切掉。

本脚本输出: 全样本/分组的开盘卖 vs 收盘卖, 以及扣掉同期全市场等权基准后的超额。
"""
import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
COST = 0.13      # 双边费(%) —— 与第 165 轮同口径; 主策略全成本(含滑点)是 0.30%


def main():
    lhb = pd.read_pickle(os.path.join(ROOT, "data", "cache", "lhb_probe.pkl"))
    p = pd.read_pickle(os.path.join(ROOT, "data", "cache", "panel_all.pkl"))[
        ["ts_code", "d", "open", "close", "is_paused"]]
    p = p[(p["is_paused"] == 0) & (p["open"] > 0) & (p["close"] > 0)]
    p["intra"] = p["close"] / p["open"] - 1
    days = sorted(p["d"].unique())
    nxt = {d: days[i + 1] for i, d in enumerate(days[:-1])}

    lhb["d2"] = lhb["d"].map(nxt)
    intra = p.set_index(["ts_code", "d"])["intra"]
    idx = pd.MultiIndex.from_arrays([lhb["code"], lhb["d2"]])
    lhb["intra2"] = intra.reindex(idx).values
    lhb = lhb.dropna(subset=["intra2", "r2"])
    lhb["ro"] = lhb["r2"] * 100
    lhb["rc"] = ((1 + lhb["r2"]) * (1 + lhb["intra2"]) - 1) * 100

    # 同期全市场等权基准(同口径)
    ps = p.sort_values(["ts_code", "d"])
    ps["c_next"] = ps.groupby("ts_code")["close"].shift(-1)
    ps["r2c"] = ps["c_next"] / ps["open"] - 1
    ps["r2o"] = ps.groupby("ts_code")["open"].shift(-1) / ps["open"] - 1
    mkt = ps.groupby("d").agg(mc=("r2c", "mean"), mo=("r2o", "mean")).dropna()
    mkt = mkt.loc[mkt.index.isin(set(lhb["d"].unique()))]
    mc, mo = mkt["mc"].mean() * 100, mkt["mo"].mean() * 100

    def grp(mask, name):
        s = lhb[mask]
        return {"group": name, "n": int(len(s)), "open": round(s["ro"].mean(), 3),
                "close": round(s["rc"].mean(), 3),
                "excess": round(s["rc"].mean() - mc, 3),
                "net_excess": round(s["rc"].mean() - mc - COST, 3)}

    rows = [grp(lhb["reason"].str.contains("换手率"), "换手榜"),
            grp((lhb["north"] > 0) & (lhb["chg"] <= -5), "北向净买&跌5%+"),
            grp(lhb["inst"] > 0, "机构净买"),
            grp(lhb["north"] > 0, "北向净买"),
            grp(lhb["reason"].str.contains("跌幅"), "跌幅榜"),
            grp(lhb["youzi"] > 0, "游资净买")]

    out = {"n": int(len(lhb)), "open_all": round(lhb["ro"].mean(), 3),
           "close_all": round(lhb["rc"].mean(), 3),
           "win_open": round((lhb["ro"] > 0).mean() * 100, 1),
           "win_close": round((lhb["rc"] > 0).mean() * 100, 1),
           "mkt_close": round(mc, 3), "mkt_open": round(mo, 3),
           "excess_open": round(lhb["ro"].mean() - mo, 3),
           "excess_close": round(lhb["rc"].mean() - mc, 3),
           "groups": rows}
    print(f"样本 {out['n']}: 开盘 {out['open_all']}% (胜率 {out['win_open']}%) / "
          f"收盘 {out['close_all']}% (胜率 {out['win_close']}%)")
    print(f"同期全市场等权: 开盘 {out['mkt_open']}% / 收盘 {out['mkt_close']}%")
    print(f"超额: 开盘 {out['excess_open']}pp / 收盘 {out['excess_close']}pp")
    for r in rows:
        print(f"  {r['group']:<14} n={r['n']:>5} 毛 {r['close']:>7}% 超额 {r['excess']:>7}pp "
              f"扣费后 {r['net_excess']:>7}pp")
    json.dump(out, open(os.path.join(ROOT, "output", "exp_lhb_close.json"), "w"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
