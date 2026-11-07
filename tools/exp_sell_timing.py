"""第166轮: T+2 卖出时点的条件化 —— 现行"全部收盘卖", 但若 T+2 高开,
开盘价可能就是当日高点。用成交明细直接比"开盘卖"与"收盘卖"在**不同开盘缺口**下的差异。

口径: 对每笔实际成交, d1=买入日, d2=d1 的下一交易日(应卖日)。
  g   = d2 开盘 / d1 收盘 - 1        (T+2 竞价缺口)
  ro  = d2 开盘 / d1 开盘 - 1        (若开盘卖)
  rc  = d2 收盘 / d1 开盘 - 1        (现行: 收盘卖)
两者都未扣费, 差额即"时点选择"的毛贡献。
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ["BT_FULL_TRADES"] = "1"

from src.backtest import prepare_context, run_backtest  # noqa: E402
from src.config import load_config  # noqa: E402


def main():
    cfg = load_config()
    ctx = prepare_context(cfg, cfg["backtest"]["trading_days"])
    r = run_backtest(cfg, None, ctx)
    tr = [t for t in r["trades_all"] if t.get("ret_pct") is not None]
    days = sorted(ctx["by_date"].keys())
    idx = {d: i for i, d in enumerate(days)}
    per = ctx["per_code"]

    rows = []
    for t in tr:
        d1 = t["buy_date"]
        i = idx.get(d1)
        if i is None or i + 1 >= len(days):
            continue
        d2 = days[i + 1]
        sub = per.get(t["code"])
        if sub is None or d1 not in sub.index or d2 not in sub.index:
            continue
        a, b = sub.loc[d1], sub.loc[d2]
        if a["open_adj"] <= 0 or b["open_adj"] <= 0:
            continue
        rows.append({"code": t["code"], "d1": d1, "d2": d2,
                     "g": (b["open_adj"] / a["close_adj"] - 1) * 100,
                     "ro": (b["open_adj"] / a["open_adj"] - 1) * 100,
                     "rc": (b["close_adj"] / a["open_adj"] - 1) * 100,
                     "ret": t["ret_pct"], "score": t.get("score"), "gap0": t.get("gap_pct")})
    df = pd.DataFrame(rows)
    print(f"样本 {len(df)} 笔")
    print(f"开盘卖 均值 {df['ro'].mean():.3f}% / 收盘卖 均值 {df['rc'].mean():.3f}% "
          f"→ 收盘腿价值 {(df['rc'] - df['ro']).mean():.3f}pp/笔")

    print("\n=== T+2 竞价缺口 g 分桶: 开盘卖 vs 收盘卖 ===")
    bins = [-99, -3, -2, -1, -0.3, 0, 0.3, 1, 2, 3, 99]
    df["gb"] = pd.cut(df["g"], bins)
    t = df.groupby("gb", observed=True).agg(n=("ro", "size"), ro=("ro", "mean"), rc=("rc", "mean"),
                                            win_o=("ro", lambda x: (x > 0).mean() * 100),
                                            win_c=("rc", lambda x: (x > 0).mean() * 100))
    t["rc-ro"] = (t["rc"] - t["ro"]).round(3)
    print(t.round(3).to_string())

    print("\n=== 若 g>=阈值 则开盘卖, 否则收盘卖: 组合均值 ===")
    for th in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 99):
        pick = np.where(df["g"] >= th, df["ro"], df["rc"])
        n_open = int((df["g"] >= th).sum())
        print(f"  阈值 {th:>5}: 组合均值 {pick.mean():.3f}%  开盘卖 {n_open:>3} 笔 "
              f"(占比 {n_open/len(df)*100:.1f}%)  Δ vs 全收盘 {pick.mean() - df['rc'].mean():+.3f}pp")

    print("\n=== 反向: g<=阈值 则收盘卖(其余开盘卖) —— 只为验证是不是'越弱越该等' ===")
    for th in (-2.0, -1.0, -0.5):
        pick = np.where(df["g"] <= th, df["rc"], df["ro"])
        print(f"  阈值 {th:>5}: 组合均值 {pick.mean():.3f}%  Δ {pick.mean() - df['rc'].mean():+.3f}pp")

    df.to_pickle(os.path.join(ROOT, "data", "cache", "sell_timing.pkl"))
    json.dump({"n": len(df), "mean_open": round(float(df['ro'].mean()), 3),
               "mean_close": round(float(df['rc'].mean()), 3)},
              open(os.path.join(ROOT, "output", "exp_sell_timing.json"), "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
