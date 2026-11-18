"""第257轮: 把全历史事件表挖出来的两个条件落回引擎验证

全历史事件表（2009-2026, 4 万行, tools/mine_fillin_257.py）挖出两处:
  1. 破位腿加"第1名当日跌幅 <= -5%": 154 笔 2.15% -> 63 笔 4.19%
  2. 深低开带 -8~-5 改 -6~-4: 64 笔 2.03% -> 78 笔 2.97%

本脚本在回测里验证（含补位资金冲突/冷却/滑点，与事件表的裸收益不同口径）:
  A 现状(256)      -8/-5, 无条件
  B 只改带          -6/-4
  C 只加破位条件     -8/-5 + break_pct -5
  D 两条都上        -6/-4 + break_pct -5
  E D + 深低开跌停   -6/-4 + break_pct -5 + deep_pct -9
"""

from __future__ import annotations

import copy
import os
import pickle
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context, run_backtest  # noqa: E402
from src.config import load_config  # noqa: E402

os.environ["BT_FULL_TRADES"] = "1"

BASE = dict(enabled=True, cooldown_days=10, idx_dist_max=-5.0,
            dist_high60_max=-42.0, deep_skip=3, list_n=3, deep_rank_max=3,
            pick_break="best", pick_deep="best")

GROUPS = [
    ("A 现状256", dict(BASE, deep_lo=-8.0, deep_hi=-5.0)),
    ("B 带-6~-4", dict(BASE, deep_lo=-6.0, deep_hi=-4.0)),
    ("C 破位跌5", dict(BASE, deep_lo=-8.0, deep_hi=-5.0, break_pct_max=-5.0)),
    ("D 两条都上", dict(BASE, deep_lo=-6.0, deep_hi=-4.0, break_pct_max=-5.0)),
    ("E D+深低开跌9", dict(BASE, deep_lo=-6.0, deep_hi=-4.0, break_pct_max=-5.0,
                          deep_pct_max=-9.0)),
    ("F 原带+深低开跌9", dict(BASE, deep_lo=-8.0, deep_hi=-5.0, break_pct_max=-5.0,
                            deep_pct_max=-9.0)),
    ("G 带-10~-4+跌9", dict(BASE, deep_lo=-10.0, deep_hi=-4.0, break_pct_max=-5.0,
                           deep_pct_max=-9.0)),
    ("H 破位跌7", dict(BASE, deep_lo=-8.0, deep_hi=-5.0, break_pct_max=-7.0)),
]


def main() -> None:
    cfg0 = load_config()
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2500
    only = sys.argv[2].split(",") if len(sys.argv) > 2 else None
    c1 = copy.deepcopy(cfg0)
    c1["fillin"]["enabled"] = True
    ctx = prepare_context(c1, n)

    out = []
    for nm, ov in GROUPS:
        if only and nm.split()[0] not in only:
            continue
        c = copy.deepcopy(cfg0)
        c["fillin"].update(ov)
        r = run_backtest(c, n, ctx)
        m = r["metrics"]
        tr = pd.DataFrame(r.get("trades_all") or [])
        fl = pd.DataFrame()
        if len(tr) and "_fillin" in tr.columns:
            f = tr["_fillin"].fillna(False).astype(bool)
            fl = tr[f & tr.ret_pct.notna()].copy()
        out.append({"name": nm, "m": m, "trades": tr, "fill": fl})
        avg = fl.ret_pct.mean() if len(fl) else float("nan")
        bp = (fl.fill_path == "破位").sum() if len(fl) and "fill_path" in fl else 0
        dp = (fl.fill_path == "深低开").sum() if len(fl) and "fill_path" in fl else 0
        print(f"\n【{nm}】 年化{m['年化收益率%']:.2f} 回撤{m['最大回撤%']:.2f} "
              f"卡玛{m['卡玛比率']:.2f} 笔数{m['交易次数']} 补位{m.get('补位笔数',0)}"
              f"(破{bp}/深{dp}) 补位单笔{avg:.2f}% 放弃{m.get('补位放弃',0)}", flush=True)

    print("\n" + "=" * 96)
    print(f"{'方案':<14}{'年化':>9}{'回撤':>9}{'卡玛':>7}{'笔数':>6}{'补位':>6}"
          f"{'破位':>5}{'深低':>5}{'补位单笔':>9}")
    for o in out:
        m, fl = o["m"], o["fill"]
        avg = fl.ret_pct.mean() if len(fl) else float("nan")
        bp = (fl.fill_path == "破位").sum() if len(fl) and "fill_path" in fl else 0
        dp = (fl.fill_path == "深低开").sum() if len(fl) and "fill_path" in fl else 0
        print(f"{o['name']:<14}{m['年化收益率%']:>9.2f}{m['最大回撤%']:>9.2f}"
              f"{m['卡玛比率']:>7.2f}{m['交易次数']:>6}{m.get('补位笔数',0):>6}"
              f"{bp:>5}{dp:>5}{avg:>9.2f}")
    with open(ROOT / f"output/w257_mine_{n}.pkl", "wb") as fh:
        pickle.dump(out, fh)


if __name__ == "__main__":
    main()
