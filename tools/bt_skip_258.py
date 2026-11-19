"""第258轮: 深低开腿的 deep_skip=3 在全历史 196 笔上不成立, 回测复核

第252轮把 deep_skip 定成 3(跳过距高最远的 3 只, 取第 4-6 名), 依据是当时
2500 日的 23 笔成交(1-3 名单笔 0.43% vs 4-8 名 2.29%)。

第257轮建的全历史事件表(4041 天 x 前 8 层)把这个判定放大到 196 笔:
  第1层  37 笔 2.53% / 胜 65% / p10 -8.95%
  第2层  32 笔 1.91%
  第3层  21 笔 1.45%
  第4层  24 笔 1.73%
  第5层  23 笔 1.07% / p10 -10.65%
  第6层  19 笔 3.04%
  第7层  21 笔 2.55%
  第8层  19 笔 -0.24% / 胜 42%
  第1-3 层  90 笔 2.06% / 胜 63% / p10 -8.14%
  第4-6 层  66 笔 1.88% / 胜 62% / p10 -8.60%   <- 当前口径
第1层不但不差, 均值和尾部都优于第4-6层。252 的"跳过最崩的3只"是 23 笔上的
噪声。本脚本在引擎里复核(含补位资金冲突/冷却/滑点)。

  A 现状 skip=3 (第4-6层)
  B skip=0 (第1-3层)
  C skip=1 (第2-4层)
  D skip=0 rank_max=1 (只第1层)
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
            dist_high60_max=-42.0, break_pct_max=-5.0,
            deep_lo=-8.0, deep_hi=-5.0,
            list_n=3, deep_rank_max=3,
            pick_break="best", pick_deep="best")

GROUPS = [
    ("A skip3现状", dict(BASE, deep_skip=3)),
    ("B skip0", dict(BASE, deep_skip=0)),
    ("C skip1", dict(BASE, deep_skip=1)),
    ("D skip0只1层", dict(BASE, deep_skip=0, deep_rank_max=1)),
    ("F 带-9~-6skip0", dict(BASE, deep_lo=-9.0, deep_hi=-6.0, deep_skip=0)),
    ("G 带-9~-6skip3", dict(BASE, deep_lo=-9.0, deep_hi=-6.0, deep_skip=3)),
    ("H 带-9~-5skip0", dict(BASE, deep_lo=-9.0, deep_hi=-5.0, deep_skip=0)),
    ("I 带-8~-6skip3", dict(BASE, deep_lo=-8.0, deep_hi=-6.0, deep_skip=3)),
    ("J 现状+d60<=2", dict(BASE, deep_skip=3, deep_dist60_max=2.0)),
    ("M 11-15半仓", dict(BASE, deep_skip=3,
                        _pool_tiers=[{"min": 21, "mult": 1.0}, {"min": 16, "mult": 1.0},
                                     {"min": 11, "mult": 0.5}, {"min": 1, "mult": 0.0}])),
    ("K 现状+d60<=0", dict(BASE, deep_skip=3, deep_dist60_max=0.0)),
    ("L 现状+d60<=5", dict(BASE, deep_skip=3, deep_dist60_max=5.0)),
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
        c["fillin"].update({k: v for k, v in ov.items() if k != "_pool_tiers"})
        if "_pool_tiers" in ov:
            c["execution"]["pool_n_tiers"] = ov["_pool_tiers"]
        r = run_backtest(c, n, ctx)
        m = r["metrics"]
        tr = pd.DataFrame(r.get("trades_all") or [])
        fl = pd.DataFrame()
        if len(tr) and "_fillin" in tr.columns:
            fl = tr[tr["_fillin"].fillna(False).astype(bool)]
        print(f"{nm:<16} 年化 {m['年化收益率%']:>7.2f}%  回撤 {m['最大回撤%']:>7.2f}%  "
              f"卡玛 {m['卡玛比率']:>5.2f}  笔数 {m['交易次数']:>4}"
              f"  补位 {len(fl)}"
              + (f"(破位{len(fl[fl['fill_path']=='破位'])}/深低{len(fl[fl['fill_path']=='深低开'])})"
                 if len(fl) else "")
              + (f"  补位单笔 {fl['ret_pct'].mean()*100:.2f}%" if len(fl) else ""), flush=True)
        if len(fl):
            g = fl.groupby("fill_rk").ret_pct.agg(["size", "mean"])
            print("     层:", " ".join(f"{int(k)}:{int(v['size'])}笔{v['mean']*100:.1f}%"
                                      for k, v in g.iterrows()), flush=True)
        out.append(dict(name=nm, metrics=m, trades=tr, fill=fl,
                        equity=r.get('equity') or []))

    with open(ROOT / f"output/w258_skip_{n}.pkl", "wb") as f:
        pickle.dump(out, f)


if __name__ == "__main__":
    main()
