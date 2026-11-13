#!/usr/bin/env python3
"""第224轮: 大盘风控参数扫描。

为什么扫这个: 卖出规则在第98/102/104/115轮被反复调过, 且注释里已经钉死了一个结构事实
——T+2 开盘浮盈>=5% 的少数笔贡献了全部复利, 其余笔合计为负。也就是说"何时卖"已到边际。
真正还没系统扫过的是"什么时候根本不该在场": regime 的 breadth_min / market_ma / vol_max
直接决定一年有多少天能交易(现在是 43.7%), 空仓期既躲开了回撤, 也丢掉了机会。

口径: 复用 prepare_context 的面板, 只重建风控序列, 因此 64 个组合只需一次取数。
输出: 每个组合的年化/回撤/卡玛/笔数, 以及前后半段各自的年化 —— 只看"两段都赢"的组合,
      避免扫出一组只在前半段有效的过拟合参数。
"""
from __future__ import annotations

import copy
import itertools
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context, run_backtest  # noqa: E402
from src.config import load_config  # noqa: E402


def metrics_of(res: dict, lo: float = 0.0, hi: float = 1.0) -> dict:
    """从净值序列切段算年化/回撤。lo/hi 是相对位置。"""
    eq = pd.DataFrame(res["equity"])
    n = len(eq)
    a, b = int(n * lo), int(n * hi)
    seg = eq.iloc[a:b]
    if len(seg) < 30:
        return {}
    nav = seg["nav"].values
    nav = nav / nav[0]
    peak, mdd = 1.0, 0.0
    for v in nav:
        peak = max(peak, v)
        mdd = min(mdd, v / peak - 1)
    d0 = pd.Timestamp(seg["d"].iloc[0])
    d1 = pd.Timestamp(seg["d"].iloc[-1])
    yrs = max(0.5, (d1 - d0).days / 365.25)
    ann = nav[-1] ** (1 / yrs) - 1
    return {"ann": ann * 100, "mdd": mdd * 100, "calmar": ann / abs(mdd) if mdd else float("nan")}


def main() -> None:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    cfg = load_config()
    t0 = time.time()
    ctx = prepare_context(cfg, days)
    print(f"上下文就绪 {len(ctx['main_dates'])} 日, {time.time()-t0:.0f}s", flush=True)

    base = run_backtest(cfg, days, ctx)
    m = base["metrics"]
    print("\n=== 基准(现行参数) ===")
    print(f"年化 {m['年化收益率%']}% / 回撤 {m['最大回撤%']}% / 卡玛 {m['卡玛比率']} / 笔数 {m['交易次数']} / 胜率 {m['胜率%']}%")

    grid = {
        "breadth_min": [0.20, 0.25, 0.30, 0.35],
        "market_ma": [10, 20, 30, 60],
        "vol_max": [1.8, 2.2, 2.6, None],
    }
    keys = list(grid)
    rows = []
    combos = list(itertools.product(*[grid[k] for k in keys]))
    print(f"\n扫描 {len(combos)} 个组合...", flush=True)
    for i, vals in enumerate(combos, 1):
        c = copy.deepcopy(cfg)
        for k, v in zip(keys, vals):
            c["regime"][k] = v
        try:
            r = run_backtest(c, days, ctx)
        except Exception as e:  # noqa: BLE001
            print(f"  [FAIL] {dict(zip(keys, vals))}: {e}")
            continue
        mm = r["metrics"]
        h1 = metrics_of(r, 0.0, 0.5)
        h2 = metrics_of(r, 0.5, 1.0)
        rows.append({
            "breadth": vals[0], "ma": vals[1], "vol": vals[2],
            "年化": mm["年化收益率%"], "回撤": mm["最大回撤%"], "卡玛": mm["卡玛比率"],
            "笔数": mm["交易次数"], "胜率": mm["胜率%"], "单笔": mm["平均单笔收益%"],
            "前半": round(h1.get("ann", float("nan")), 1),
            "后半": round(h2.get("ann", float("nan")), 1),
        })
        if i % 8 == 0:
            print(f"  {i}/{len(combos)}  {time.time()-t0:.0f}s", flush=True)

    df = pd.DataFrame(rows).sort_values("卡玛", ascending=False)
    print("\n=== 按卡玛排序(前 20) ===")
    print(df.head(20).to_string(index=False))
    print("\n=== 两段都为正且卡玛前 10(稳健性筛选) ===")
    stable = df[(df["前半"] > 0) & (df["后半"] > 0)]
    print(stable.head(10).to_string(index=False))
    print("\n=== 各维度边际 ===")
    for k in ["breadth", "ma", "vol"]:
        print(f"\n-- {k} --")
        print(df.groupby(k, dropna=False).agg(年化中位=("年化", "median"), 卡玛中位=("卡玛", "median"),
                                              笔数中位=("笔数", "median")).to_string())

    out = ROOT / "output" / "regime_scan_224.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n已存 {out}")


if __name__ == "__main__":
    main()
