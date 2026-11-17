"""第251轮 C: 补位收敛 —— 单吊定位下的"少而精"

用户: 这是单吊系统, 补位 2500 日 306 笔(年均 28 笔)太多。
A 笔级体检发现:
  · 破位 332 笔落在 54 波里, 最长一波连续 23 个交易日 —— 那是网格摊平, 不是单吊
  · 指数 -4.2~-3.51 档 77 笔单笔仅 0.64%; dh60 -38~-35 档单笔 -0.63%
  · 深低开 -9.5~-8 档单笔 0.99%(跌停附近开盘往往继续跌)

本脚本用真实引擎跑 6 组, 只为回答: 砍到多少笔, 卡玛最大。
ctx 只建一次, 各组共用。
"""

from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context, run_backtest  # noqa: E402
from src.config import load_config  # noqa: E402

os.environ["BT_FULL_TRADES"] = "1"

GROUPS = [
    ("A 主策略单独", dict(enabled=False)),
    ("B8 251定稿(分腿冷却)", dict(enabled=True, cooldown_days=10, idx_dist_max=-5.0,
                            dist_high60_max=-42.0, deep_lo=-8.0, deep_rank_max=8)),
]


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
            fl = tr[_f & tr.ret_pct.notna()]
        out.append({"name": nm, "m": m, "trades": tr, "fill": fl,
                    "equity": r.get("equity")})
        print(f"\n【{nm}】")
        for k in ("年化收益率%", "最大回撤%", "卡玛比率", "夏普比率", "交易次数", "胜率%",
                  "平均单笔收益%", "空仓天数", "主策略笔数", "补位笔数", "补位·破位",
                  "补位·深低开", "补位触发", "补位冷却跳过"):
            if k in m:
                print(f"  {k:<14}{m[k]}")
        if len(fl):
            print(f"  补位明细: {len(fl)} 笔 单笔 {fl.ret_pct.mean():.2f}% "
                  f"胜率 {(fl.ret_pct>0).mean()*100:.0f}%")

    print("\n" + "=" * 92)
    print(f"{'方案':<26}{'年化':>8}{'回撤':>9}{'卡玛':>7}{'笔数':>7}{'补位':>6}"
          f"{'破位':>5}{'深低':>5}{'单笔':>7}{'空仓':>6}")
    for o in out:
        m = o["m"]
        print(f"{o['name']:<26}{m['年化收益率%']:>8.2f}{m['最大回撤%']:>9.2f}"
              f"{m['卡玛比率']:>7.2f}{m['交易次数']:>7}{m.get('补位笔数',0):>6}"
              f"{m.get('补位·破位',0):>5}{m.get('补位·深低开',0):>5}"
              f"{m['平均单笔收益%']:>7.2f}{m['空仓天数']:>6}")

    # 补位逐年
    print("\n【补位逐年笔数 / 单笔】")
    rows = []
    for o in out:
        fl = o["fill"]
        if not len(fl):
            continue
        g = fl.assign(yr=fl.buy_date.str[:4]).groupby("yr").agg(
            n=("ret_pct", "size"), avg=("ret_pct", "mean"))
        rows.append(g.assign(方案=o["name"]))
    if rows:
        t = pd.concat(rows).reset_index()
        piv = t.pivot(index="yr", columns="方案", values="n")
        print("笔数:")
        print(piv.fillna(0).astype(int).to_string())
        piv2 = t.pivot(index="yr", columns="方案", values="avg")
        print("\n单笔%:")
        print(piv2.round(2).to_string())

    import pickle
    with open(ROOT / "output" / f"w251_chk_{n}.pkl", "wb") as f:
        pickle.dump(out, f)
    print(f"\n明细已存 output/w251_chk_{n}.pkl")


if __name__ == "__main__":
    main()
