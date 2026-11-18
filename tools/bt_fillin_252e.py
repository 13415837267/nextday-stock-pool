"""第252轮 E: 3 只池下, 深低开腿还值不值得留

252D 证伪了"跌幅榜当候选源"(深低开 135 笔单笔 -0.80%, 卡玛 2.55) ——
"昨天跌最狠的"不是"明天崩到位的", 而是"接着崩的"。

3 只池下深低开只剩 19 笔、单笔 0.72%（8 只池时 43 笔 1.60%）。既然几乎不赚钱,
就问一句: 干脆只留破位腿?
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
    ("H 3/3", dict(BASE, list_n=3, deep_rank_max=3)),
    ("Z1 3只池 只留破位", dict(BASE, list_n=3, deep_rank_max=3, deep_hi=-99.0)),
    ("Z2 3/5", dict(BASE, list_n=3, deep_rank_max=5)),
    ("Z3 10/3", dict(BASE, list_n=10, deep_rank_max=3)),
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
        print(f"\n【{nm}】 年化{m['年化收益率%']} 回撤{m['最大回撤%']} 卡玛{m['卡玛比率']}"
              f" 笔数{m['交易次数']} 补位{m.get('补位笔数', 0)}")

    print("\n" + "=" * 96)
    print(f"{'方案':<20}{'年化':>8}{'回撤':>9}{'卡玛':>7}{'笔数':>7}{'补位':>6}"
          f"{'破位':>5}{'深低':>5}{'单笔':>7}{'补位单笔':>9}")
    for o in out:
        m, fl = o["m"], o["fill"]
        avg = fl.ret_pct.mean() if len(fl) else float("nan")
        print(f"{o['name']:<20}{m['年化收益率%']:>8.2f}{m['最大回撤%']:>9.2f}"
              f"{m['卡玛比率']:>7.2f}{m['交易次数']:>7}{m.get('补位笔数', 0):>6}"
              f"{m.get('补位·破位', 0):>5}{m.get('补位·深低开', 0):>5}"
              f"{m['平均单笔收益%']:>7.2f}{avg:>9.2f}")

    with open(ROOT / "output" / f"w252e_only_{n}.pkl", "wb") as f:
        pickle.dump(out, f)
    print(f"\n明细已存 output/w252e_only_{n}.pkl")


if __name__ == "__main__":
    main()
