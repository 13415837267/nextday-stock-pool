"""940 次跳过，如果真买了会怎样？（虚拟成交归因）

这是"交易次数少是不是缺陷"的决定性证据：
  - 若被跳过票的期望为负 -> 门槛拦得对，379 笔已是合理产量
  - 若为正 -> 空仓是真实损失，值得优化

口径：按现行规则虚拟成交（T+1 开盘买、T+2 收盘卖），用复权价，扣滑点/佣金/印花税，
      但**不计资金约束**（纯信号期望，与"钱够不够"无关）。

注意：这是"孤立单笔期望"，不含复利路径效应，也不代表"全买了就一定赚"——
      它只回答"这一类信号本身有没有正期望"。
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

os.environ.setdefault("BT_FULL_TRADES", "1")

from src.backtest import prepare_context
from src.config import load_config


def main():
    cfg = load_config()
    ex = cfg["execution"]
    slip = float(ex.get("slippage", 0))
    comm = float(ex.get("commission_rate", 0))
    stamp = float(ex.get("stamp_tax", 0))
    print(f"成本口径: 滑点{slip*1000:g}‰ + 佣金{comm*10000:g}万 + 印花税{stamp*10000:g}万\n")

    ctx = prepare_context(cfg, int(cfg["backtest"]["trading_days"]))
    by_date = ctx["by_date"]
    dates = ctx["main_dates"]
    didx = {d: i for i, d in enumerate(dates)}

    tr = json.load(open("output/backtest.json", encoding="utf-8")).get("trades_all") or []
    skip = [t for t in tr if t.get("ret_pct") is None and t.get("buy_date")]
    print(f"跳过记录: {len(skip)} 条（有 buy_date 的）\n")

    def px(d, code, col):
        df = by_date.get(d)
        if df is None:
            return None
        h = df[df["ts_code"] == code]
        if h.empty:
            return None
        v = h.iloc[0].get(col)
        return float(v) if pd.notna(v) else None

    rows = []
    for t in skip:
        bd = str(t["buy_date"])
        i = didx.get(bd)
        if i is None or i + 1 >= len(dates):
            continue
        sd = dates[i + 1]
        bo = px(bd, t["code"], "open_adj")
        sc = px(sd, t["code"], "close_adj")
        if not bo or not sc or bo <= 0:
            continue
        # 买入价含滑点, 卖出价含滑点+双边费用
        buy = bo * (1 + slip)
        sell = sc * (1 - slip)
        gross = sell / buy - 1
        fee = comm * 2 + stamp
        ret = (gross - fee) * 100
        rows.append({"note": t.get("note") or "", "ret": ret, "code": t["code"],
                     "signal_date": t.get("signal_date"), "buy_date": bd,
                     "pool": t.get("pool")})

    df = pd.DataFrame(rows)
    print(f"可虚拟成交: {len(df)} 条\n")

    def kind(n):
        if "停牌" in n:
            return "停牌"
        if "一字涨停" in n:
            return "一字涨停"
        if "高开" in n:
            return "高开"
        if "低开" in n and "承接线" not in n:
            return "低开"
        if "未达承接线" in n:
            return "缺口未达承接线"
        if "崩塌线" in n:
            return "缺口深于崩塌线"
        if "低于门槛" in n:
            return "评分低于门槛"
        return "其他"

    df["kind"] = df["note"].map(kind)
    print("=" * 80)
    print("一、按跳过原因：假如买了的期望")
    print("=" * 80)
    print(f"{'原因':18s} {'笔数':>5s} {'均值%':>8s} {'中位%':>8s} {'胜率%':>7s} {'总贡献pp':>9s}")
    for k, g in df.groupby("kind"):
        print(f"{k:18s} {len(g):5d} {g['ret'].mean():8.3f} {g['ret'].median():8.3f} "
              f"{(g['ret']>0).mean()*100:7.1f} {g['ret'].sum():9.1f}")

    # 缺口档位细分
    g1 = df[df["kind"] == "缺口未达承接线"].copy()
    if len(g1):
        g1["gap"] = g1["note"].str.extract(r"缺口([-\d.]+)%").astype(float)
        print("\n" + "=" * 80)
        print("二、'缺口未达承接线'按缺口深度细分（承接线=-2.5%）")
        print("=" * 80)
        bins = [-99, -2.0, -1.5, -1.0, -0.5, 0.5]
        labels = ["≤-2.0(最接近门槛)", "-2.0~-1.5", "-1.5~-1.0", "-1.0~-0.5", ">-0.5"]
        g1["bucket"] = pd.cut(g1["gap"], bins=bins, labels=labels)
        print(f"{'档位':22s} {'笔数':>5s} {'均值%':>8s} {'中位%':>8s} {'胜率%':>7s}")
        for lb, sub in g1.groupby("bucket", observed=True):
            print(f"{str(lb):22s} {len(sub):5d} {sub['ret'].mean():8.3f} "
                  f"{sub['ret'].median():8.3f} {(sub['ret']>0).mean()*100:7.1f}")

    # 评分细分
    g2 = df[df["kind"] == "评分低于门槛"].copy()
    if len(g2):
        g2["sc"] = g2["note"].str.extract(r"评分([\d.]+)低于").astype(float)
        print("\n" + "=" * 80)
        print("三、'评分低于门槛'按评分细分（门槛=1.0）")
        print("=" * 80)
        bins2 = [-99, 0.2, 0.4, 0.6, 0.8, 1.01]
        labels2 = ["<0.2", "0.2~0.4", "0.4~0.6", "0.6~0.8", "0.8~1.0"]
        g2["bucket"] = pd.cut(g2["sc"], bins=bins2, labels=labels2)
        print(f"{'档位':12s} {'笔数':>5s} {'均值%':>8s} {'中位%':>8s} {'胜率%':>7s}")
        for lb, sub in g2.groupby("bucket", observed=True):
            print(f"{str(lb):12s} {len(sub):5d} {sub['ret'].mean():8.3f} "
                  f"{sub['ret'].median():8.3f} {(sub['ret']>0).mean()*100:7.1f}")

    # 汇总对比
    real = [t for t in tr if t.get("ret_pct") is not None]
    rp = np.array([t["ret_pct"] for t in real])
    print("\n" + "=" * 80)
    print("四、与已成交对比")
    print("=" * 80)
    print(f"  已成交 {len(rp):4d} 笔: 均值 {rp.mean():+7.3f}% / 中位 {np.median(rp):+7.3f}% "
          f"/ 胜率 {(rp>0).mean()*100:5.1f}%")
    print(f"  被跳过 {len(df):4d} 笔: 均值 {df['ret'].mean():+7.3f}% / 中位 {df['ret'].median():+7.3f}% "
          f"/ 胜率 {(df['ret']>0).mean()*100:5.1f}%")
    print(f"  差值: {df['ret'].mean()-rp.mean():+.3f}pp/笔")


if __name__ == "__main__":
    main()
