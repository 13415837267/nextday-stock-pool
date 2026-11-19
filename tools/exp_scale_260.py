"""第260轮 · 规模-年化曲线: 这套策略究竟能装多少钱

第129轮的容量结论(1250 万)用的是 **全天成交额 x 5%**, 隐含"一整天慢慢买"。
真实的执行是 T+1 开盘一笔进。用 730 笔买入日的分钟线实测:
首分钟(含集合竞价)只占全天成交额的 2.78%(中位), 开盘一笔的真实容量是 33 万, 不是 1125 万。

本轮把"钱越多、年化越差"这条曲线画出来:
  对每个规模 S, 每笔找到"能以 5% 参与率装下 S 的最小窗口" -> 用该窗口 VWAP 溢价成交;
  若 240 分钟都装不下, 按容量比例部分成交(剩余资金空转)。
只改执行, 不改选股 —— 这是规模对策略的定价, 不是对策略的优化。
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WINS = [1, 3, 5, 10, 15, 30, 60, 120, 240]
PART = 0.05          # 参与率上限(129轮口径)
TRADING_DAYS = 3200


def main():
    d = pickle.load(open(ROOT / "output" / "w258_skip_3200.pkl", "rb"))
    g = [x for x in d if x["name"].startswith("M ")][0]
    t = g["trades"]
    t = t[t.ret_pct.notna()].copy().sort_values(["signal_date", "code"])
    ex = pd.read_csv(ROOT / "output" / "w260_exec.csv", dtype={"buy_date": str})
    ex = pd.read_csv(ROOT / "output" / "w260_exec.csv", dtype={"buy_date": str})
    ex["key"] = ex.code + "_" + ex.buy_date
    t["key"] = t.code + "_" + t.buy_date
    t = t.merge(ex[["key"] + [f"amt{w}" for w in WINS] + [f"px{w}" for w in WINS]],
                on="key", how="left", validate="1:1")
    miss = t[f"amt{WINS[0]}"].isna().sum()
    print(f"成交 {len(t)} 笔, 分钟线缺失 {miss} 笔(按首分钟口径填充)")

    base = (1 + t.pos_frac * t.ret_pct / 100).prod()
    print(f"基准(不计容量): 终值 {base:,.0f}x  年化 {(base ** (252 / TRADING_DAYS) - 1) * 100:.1f}%\n")

    rows = []
    for S in [30e4, 50e4, 100e4, 200e4, 300e4, 500e4, 1000e4, 2000e4, 5000e4, 1e8, 2e8]:
        wsel, prem, fillfrac = [], [], []
        for r in t.itertuples():
            got = False
            for w in WINS:
                a = getattr(r, f"amt{w}")
                if pd.notna(a) and a * PART >= S:
                    wsel.append(w); prem.append(getattr(r, f"px{w}")); fillfrac.append(1.0)
                    got = True
                    break
            if not got:
                a240 = getattr(r, "amt240")
                f = 1.0 if pd.isna(a240) else min(1.0, a240 * PART / S)
                # 装不下: 剩余按全天 VWAP 成交(不是按开盘价), 否则会低估大资金的成本
                p240 = getattr(r, "px240", np.nan)
                wsel.append(np.nan); prem.append(0.0 if pd.isna(p240) else float(p240))
                fillfrac.append(max(f, 0.0))
        t["w"] = wsel
        t["px"] = prem
        t["ff"] = fillfrac
        px = t["px"].fillna(0.0)
        ret_adj = ((1 + t.ret_pct / 100) / (1 + px) - 1) * 100
        # 装不下的部分: 资金空转(按 0 收益, 保守)
        r_eff = t.pos_frac * t.ff * ret_adj / 100
        mult = (1 + r_eff).prod()
        ann = (mult ** (252 / TRADING_DAYS) - 1) * 100
        rows.append({
            "规模(万)": S / 1e4,
            "年化%": round(ann, 1),
            "终值倍数": round(mult, 1),
            "需等待分钟(中位)": np.nanmedian(t.w),
            "溢价%(均值)": round(np.nanmean(px) * 100, 3),
            "部分成交笔占比%": round((t.ff < 1).mean() * 100, 1),
            "平均仓位%": round(t.ff.mean() * 100, 1),
        })
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    out.to_csv(ROOT / "output" / "w260_scale.csv", index=False)

    # 单笔层面的分档(不依赖复利): 规模对单笔的伤害
    print("\n=== 规模对单笔的影响 (同一批笔, 只换成交价与成交量) ===")
    for S in [50e4, 100e4, 300e4, 1000e4]:
        wsel, prem = [], []
        for r in t.itertuples():
            got = False
            for w in WINS:
                a = getattr(r, f"amt{w}")
                if pd.notna(a) and a * PART >= S:
                    wsel.append(w); prem.append(getattr(r, f"px{w}")); got = True; break
            if not got:
                wsel.append(240); prem.append(getattr(r, "px240", 0.0))
        px = pd.Series(prem).fillna(0).values
        ra = ((1 + t.ret_pct.values / 100) / (1 + px) - 1) * 100
        print(f"  {S/1e4:>6.0f}万: 单笔 {ra.mean():.3f}% (原始 {t.ret_pct.mean():.3f}%), "
              f"中位等待 {np.nanmedian(wsel):.0f} 分钟")


if __name__ == "__main__":
    main()
