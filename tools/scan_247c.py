"""第247轮 C: 撤回上一轮的猜测 —— 主策略并没有按年 rolling 的失真

246 轮报告末尾写「主策略也在用 bias60/ret20 打分, 每年 1 月分数同样失真」。
这是猜的, 没有验证。本脚本验证它, 结论是**错的**:

  · src/backtest.py:prepare_context 一次性 fetch_panel(主窗口 2500 日 + warmup 160 日),
    然后 build_indicators 在整块面板上算 rolling —— 没有按年分片。
  · src/selector.py 里最长的滚动窗口是 20 日, warmup 160 日远超它。
  · tools/build_features_234.py 才是按年跑的, 那是 243 轮起为了做特征挖掘自建的旁路,
    主策略从来不读 output/feat234。

顺带回答一个真问题: 主策略 1 月确实弱(43 笔单笔 0.38%, 非 1 月 1.32%)。
这是"1 月全市场本来就弱"还是"策略在 1 月失灵"? 用全市场等权日收益按月对照。
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"


def main() -> None:
    print("=" * 78)
    print("第247轮 C: 主策略是否失真 / 1 月季节性的归因")
    print("=" * 78)

    tr = pd.read_csv(OUT / "w232_trades.csv")
    tr["signal"] = tr["signal"].astype(int)
    tr["m"] = tr.signal // 100 % 100
    tr["yr"] = tr.signal // 10000

    mk = pd.read_csv(OUT / "feat234" / "market.csv",
                     usecols=["trade_date", "eq_ret", "med_pct"])
    mk["trade_date"] = mk["trade_date"].astype(int)
    mk["m"] = mk.trade_date // 100 % 100

    # 全市场: 按月聚合的"平均个股日涨幅"(eq_ret 是等权组合日收益, 单位 pp)
    mkt_m = mk.groupby("m")["eq_ret"].agg(["mean", "count"])
    mkt_m.columns = ["全市场日均", "天数"]

    g = tr.groupby("m").agg(n=("ret", "size"), 单笔=("ret", "mean"),
                            胜率=("ret", lambda x: (x > 0).mean() * 100))
    tab = g.join(mkt_m)
    tab["超额"] = tab["单笔"] - tab["全市场日均"]
    print("\n【1】按信号月: 主策略 vs 全市场")
    print(tab.round(2).to_string())
    print(f"\n  1 月超额 {tab.loc[1, '超额']:+.2f}pp   非 1 月超额 "
          f"{((tab['单笔'] * tab['n']).sum() - tab.loc[1, '单笔'] * tab.loc[1, 'n']) / (tab['n'].sum() - tab.loc[1, 'n']) - tab.loc[tab.index != 1, '全市场日均'].mean():+.2f}pp")

    print("\n【2】1 月 vs 其他月, 分年")
    rows = []
    for yr, gg in tr.groupby("yr"):
        j = gg[gg.m == 1]
        o = gg[gg.m != 1]
        mj = mk[(mk.trade_date // 10000 == yr) & (mk.m == 1)]["eq_ret"].mean()
        mo = mk[(mk.trade_date // 10000 == yr) & (mk.m != 1)]["eq_ret"].mean()
        rows.append({"yr": yr, "1月笔数": len(j),
                     "1月单笔": round(j.ret.mean(), 2) if len(j) else np.nan,
                     "1月全市场": round(mj, 2) if pd.notna(mj) else np.nan,
                     "其他月笔数": len(o), "其他月单笔": round(o.ret.mean(), 2) if len(o) else np.nan,
                     "其他月全市场": round(mo, 2) if pd.notna(mo) else np.nan})
    t = pd.DataFrame(rows)
    t["1月超额"] = t["1月单笔"] - t["1月全市场"]
    t["其他月超额"] = t["其他月单笔"] - t["其他月全市场"]
    print(t.to_string(index=False))
    print(f"\n  1 月超额均值 {t['1月超额'].mean():+.2f}pp (n={t['1月超额'].notna().sum()} 年)  "
          f"其他月 {t['其他月超额'].mean():+.2f}pp")
    print(f"  1 月超额为正的年数 {int((t['1月超额'] > 0).sum())}/{int(t['1月超额'].notna().sum())}")

    print("\n【3】如果主策略 1 月不做, 会怎样(仅诊断, 不改策略)")
    cal = sorted(int(f.split("/")[-1].replace(".csv.gz", ""))
                 for f in glob.glob(str(ROOT / "data" / "panel" / "*.csv.gz")))
    cal = np.array([d for d in cal if d >= tr.signal.min()])

    def M(sub: pd.DataFrame) -> tuple[float, float, float]:
        base = pd.Series(0.0, index=cal)
        for dt, gg in sub.groupby("signal"):
            base[int(dt)] = gg["ret"].mean()
        nav = (1 + base / 100).cumprod()
        ann = nav.iloc[-1] ** (244 / len(base)) - 1
        dd = (nav / nav.cummax() - 1).min()
        return round(ann * 100, 2), round(dd * 100, 2), round(ann * 100 / abs(dd * 100), 2)

    print(f"  全部 559 笔        {M(tr)}")
    print(f"  去掉 1 月 43 笔    {M(tr[tr.m != 1])}  ← 空仓那天收益记 0, 不是重分配")

    print("\n【4】代码事实核对")
    import re
    sel = (ROOT / "src" / "selector.py").read_text(encoding="utf-8")
    wins = sorted(set(int(x) for x in re.findall(r"rolling\((\d+)", sel)))
    print(f"  selector.py 里的滚动窗口: {wins}  → 最大 {max(wins)} 日")
    bt = (ROOT / "src" / "backtest.py").read_text(encoding="utf-8")
    print(f"  backtest.py 一次性取面板: {'fetch_panel(client, panel_dates' in bt}")
    print(f"  backtest.py 在整块面板上算指标: {'build_indicators(panel)' in bt}")
    print("  → 主策略没有任何按年分片, 246 轮末尾的猜测不成立")


if __name__ == "__main__":
    main()
