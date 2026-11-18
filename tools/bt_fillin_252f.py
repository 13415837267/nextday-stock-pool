"""第252轮 F: 深低开腿的"名次效应"能不能救回 3 只池

252B/B 组 43 笔深低开按名次分档（K 组独立复现）:
  1-3 名  16 笔  0.43%（K: 0.38%）
  4-5 名  10 笔  0.99%
  6-8 名  17 笔  3.06%（K: 2.37% 对 >3 名）

距前高最远 = 长期崩塌趋势, 深低开后继续跌; 中段超跌才有反弹弹性。
所以"池压到 3 只"恰好把最赚的 6-8 名砍掉, 只留下最差的前三名 ——
这不是"更精", 是"只留毒药"。

本组: 深低开腿从第 4 名开始取（池仍 3~5 只, 页面只列这 3~5 只）。
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
            dist_high60_max=-42.0, deep_lo=-8.0, pick_break="lowest")

GROUPS = [
    ("H 3/3 前3名", dict(BASE, list_n=3, deep_rank_max=3)),
    ("AA 3/3 第4-6名", dict(BASE, list_n=3, deep_rank_max=3, deep_skip=3)),
    ("AB 3/5 第4-8名", dict(BASE, list_n=3, deep_rank_max=5, deep_skip=3)),
    ("AC 3/4 第5-8名", dict(BASE, list_n=3, deep_rank_max=4, deep_skip=4)),
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
            fl = tr[_f & tr.ret_pct.notna()].copy()
        out.append({"name": nm, "m": m, "trades": tr, "fill": fl,
                    "equity": r.get("equity")})
        g = fl.groupby("fill_path").agg(n=("ret_pct", "size"), avg=("ret_pct", "mean")) if len(fl) else None
        print(f"\n【{nm}】年化{m['年化收益率%']} 回撤{m['最大回撤%']} 卡玛{m['卡玛比率']} "
              f"笔数{m['交易次数']} 补位{m.get('补位笔数', 0)}"
              + ("  " + "; ".join(f"{i} {int(r2.n)}笔 {r2.avg:.2f}%" for i, r2 in g.iterrows()) if g is not None else ""))

    print("\n" + "=" * 92)
    print(f"{'方案':<20}{'年化':>8}{'回撤':>9}{'卡玛':>7}{'笔数':>7}{'补位':>6}"
          f"{'破位':>5}{'深低':>5}{'补位单笔':>9}")
    for o in out:
        m, fl = o["m"], o["fill"]
        avg = fl.ret_pct.mean() if len(fl) else float("nan")
        print(f"{o['name']:<20}{m['年化收益率%']:>8.2f}{m['最大回撤%']:>9.2f}"
              f"{m['卡玛比率']:>7.2f}{m['交易次数']:>7}{m.get('补位笔数', 0):>6}"
              f"{m.get('补位·破位', 0):>5}{m.get('补位·深低开', 0):>5}{avg:>9.2f}")

    with open(ROOT / "output" / f"w252f_skip_{n}.pkl", "wb") as f:
        pickle.dump(out, f)
    print(f"\n明细已存 output/w252f_skip_{n}.pkl")


if __name__ == "__main__":
    main()
