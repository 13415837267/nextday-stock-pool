"""第252轮: 补位池最多三个 —— 候选宽度收缩的判决

用户: "补位池最多三个, 继续优化"（251 定稿后补位仍是 71 笔/2500 日, 候选宽度 10/8）。

概念澄清:
  · "补位池" = T 日收盘给 T+1 的候选名单(cands), 不是"同时在手的只数"
    (T+1 开盘买、T+2 收盘卖, 天然最多 2 层, 一天只买一只 -> 上限根本用不到)。
  · 池宽的真正作用: choose() 从上往下找第一只"开盘缺口达标"的票。
    池越宽 = 排名靠后的票也有机会被挑中; 池窄 = 要么排名最前的达标, 要么今天不做。

分组:
  A 主策略单独(不补位)           —— 基准
  B 251 定稿 list_n=10 / deep=8  —— 现状
  C 5 / 5
  D 3 / 3                        —— 用户口径
  E 2 / 2                        —— 更极端
  F 10 / 3                       —— 只收深低开
  G 3 / 8                        —— 只收破位
  另: 每组统计买入名次分布 / 补位同时持仓峰值

用法: python3 tools/bt_fillin_252.py [交易日数, 默认2500]
"""

from __future__ import annotations

import copy
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context, run_backtest  # noqa: E402
from src.config import load_config  # noqa: E402

os.environ["BT_FULL_TRADES"] = "1"

BASE = dict(enabled=True, cooldown_days=10, idx_dist_max=-5.0,
            dist_high60_max=-42.0, deep_lo=-8.0)

GROUPS = [
    ("A 主策略单独", dict(enabled=False)),
    ("B 251定稿 10/8", dict(BASE, list_n=10, deep_rank_max=8)),
    ("C 5/5", dict(BASE, list_n=5, deep_rank_max=5)),
    ("D 3/3", dict(BASE, list_n=3, deep_rank_max=3)),
    ("E 2/2", dict(BASE, list_n=2, deep_rank_max=2)),
    ("F 10/3", dict(BASE, list_n=10, deep_rank_max=3)),
    ("G 3/8", dict(BASE, list_n=3, deep_rank_max=8)),
]

KEYS = ("年化收益率%", "最大回撤%", "卡玛比率", "夏普比率", "交易次数", "胜率%",
        "平均单笔收益%", "空仓天数", "主策略笔数", "补位笔数", "补位·破位",
        "补位·深低开", "补位触发", "补位放弃")


def peak_hold(fl: pd.DataFrame) -> int:
    """补位同时持仓峰值: T+1 买、T+2 卖, 按 buy/sell 日期扫描。"""
    if not len(fl):
        return 0
    ev = []
    for r in fl.itertuples():
        ev.append((str(r.buy_date), 1))
        ev.append((str(getattr(r, "sell_date", r.buy_date)), -1))
    ev.sort()
    cur = best = 0
    for _, k in ev:
        cur += k
        best = max(best, cur)
    return best


def main() -> None:
    cfg0 = load_config()
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2500
    c1 = copy.deepcopy(cfg0)
    c1["fillin"]["enabled"] = True
    ctx = prepare_context(c1, n)

    out = []
    for nm, ov in GROUPS:
        c = copy.deepcopy(cfg0)
        c["fillin"].update(ov)
        r = run_backtest(c, n, ctx)
        m = r["metrics"]
        tr = pd.DataFrame(r.get("trades_all") or [])
        fl = pd.DataFrame()
        if len(tr) and "_fillin" in tr.columns:
            _f = tr["_fillin"].fillna(False).astype(bool)
            fl = tr[_f & tr.ret_pct.notna()].copy()
        out.append({"name": nm, "m": m, "trades": tr, "fill": fl,
                    "equity": r.get("equity")})
        print(f"\n【{nm}】")
        for k in KEYS:
            if k in m:
                print(f"  {k:<14}{m[k]}")
        if len(fl):
            print(f"  补位明细: {len(fl)} 笔 单笔 {fl.ret_pct.mean():.2f}% "
                  f"胜率 {(fl.ret_pct > 0).mean() * 100:.0f}% "
                  f"名次中位 {fl['_rk'].median() if '_rk' in fl else 'NA'} "
                  f"持仓峰值 {peak_hold(fl)}")

    print("\n" + "=" * 96)
    print(f"{'方案':<18}{'年化':>8}{'回撤':>9}{'卡玛':>7}{'笔数':>7}{'补位':>6}"
          f"{'破位':>5}{'深低':>5}{'单笔':>7}{'补位单笔':>9}{'放弃':>6}")
    for o in out:
        m = o["m"]
        fl = o["fill"]
        avg = fl.ret_pct.mean() if len(fl) else float("nan")
        print(f"{o['name']:<18}{m['年化收益率%']:>8.2f}{m['最大回撤%']:>9.2f}"
              f"{m['卡玛比率']:>7.2f}{m['交易次数']:>7}{m.get('补位笔数', 0):>6}"
              f"{m.get('补位·破位', 0):>5}{m.get('补位·深低开', 0):>5}"
              f"{m['平均单笔收益%']:>7.2f}{avg:>9.2f}{m.get('补位放弃', 0):>6}")

    # 补位逐年
    print("\n【补位逐年笔数】")
    rows = []
    for o in out:
        fl = o["fill"]
        if len(fl):
            g = fl.assign(yr=fl.buy_date.str[:4]).groupby("yr").agg(
                n=("ret_pct", "size"), avg=("ret_pct", "mean"))
            rows.append(g.assign(方案=o["name"]))
    if rows:
        t = pd.concat(rows).reset_index()
        print(t.pivot(index="yr", columns="方案", values="n").fillna(0).astype(int).to_string())
        print("\n【补位逐年单笔%】")
        print(t.pivot(index="yr", columns="方案", values="avg").round(2).to_string())

    with open(ROOT / "output" / f"w252_pool_{n}.pkl", "wb") as f:
        pickle.dump(out, f)
    print(f"\n明细已存 output/w252_pool_{n}.pkl")


if __name__ == "__main__":
    main()
