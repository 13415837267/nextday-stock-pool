"""第259轮: 单槽的机会成本 —— 31-40 只这一档该不该降仓

第258轮把 11-15 只改成半仓, 机制不是"这档票更差"而是**资金周转**: 半仓时一笔没结算完,
下一笔仍买得起, 等于在单槽里挤出第二条腿; 省下的仓位损失由多出来的成交补回。
代价是"这档的敞口减半", 收益是"多做了 12~15 笔"。

同一机制应该适用于所有"单笔期望低于单槽机会成本"的档位。3200 日逐笔按池深摊开:

    pool_n   笔数   单笔
    11-20    141  +0.97%
    21-30    106  +1.72%
    31-40     67  +0.97%   ← 坑(第175 轮也见过: 30~40 +0.54%)
    41-60     71  +1.81%
    61-80     46  +2.75%
    81-120    74  +2.82%

31-40 与 11-20 一样是 0.97%, 但它现在是满仓。若"单槽机会成本 ≈ 1.5%/2 天"成立,
这一档降半仓应该和 258 的 11-15 一样: 少赚一点敞口, 多换几笔成交。

  A 现状(41+/31-40/21-30/16-20/11-15 = 1/1/1/1/0.5)
  B 31-40 半仓
  C 31-40 七成
  D 31-40 三成
  E 31-40 半仓 + 去掉 pullback 备用族(3200 日 10 笔 -1.9%)
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


def tiers(m31: float) -> list[dict]:
    return [{"min": 41, "mult": 1.0}, {"min": 31, "mult": m31}, {"min": 21, "mult": 1.0},
            {"min": 16, "mult": 1.0}, {"min": 11, "mult": 0.5}, {"min": 1, "mult": 0.0}]


GROUPS = [
    ("A 现状", dict(m31=1.0)),
    ("B 31-40半仓", dict(m31=0.5)),
    ("C 31-40七成", dict(m31=0.7)),
    ("D 31-40三成", dict(m31=0.3)),
    ("E 半仓+去pb", dict(m31=0.5, no_pb=True)),
]


def main() -> None:
    cfg0 = load_config()
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    only = sys.argv[2].split(",") if len(sys.argv) > 2 else None
    c1 = copy.deepcopy(cfg0)
    c1["fillin"]["enabled"] = True
    ctx = prepare_context(c1, n)

    out = []
    for nm, ov in GROUPS:
        if only and nm.split()[0] not in only:
            continue
        c = copy.deepcopy(cfg0)
        c["execution"]["pool_n_tiers"] = tiers(ov["m31"])
        if ov.get("no_pb"):
            c["selection"]["fallbacks"] = [f for f in c["selection"]["fallbacks"]
                                           if f.get("mode") != "pullback"]
        r = run_backtest(c, n, ctx)
        m = r["metrics"]
        tr = pd.DataFrame(r.get("trades_all") or [])
        real = tr[tr["ret_pct"].notna()] if len(tr) else tr
        fb = real[real["pool"].astype(str).str.startswith("fb")] if len(real) else real
        print(f"{nm:<14} 年化 {m['年化收益率%']:>7.2f}%  回撤 {m['最大回撤%']:>7.2f}%  "
              f"卡玛 {m['卡玛比率']:>5.2f}  笔数 {m['交易次数']:>4}"
              f"  备用 {len(fb)}  单笔 {real['ret_pct'].mean():.2f}%", flush=True)
        out.append(dict(name=nm, metrics=m, trades=tr, equity=r.get("equity") or []))

    with open(ROOT / f"output/w259_pool_{n}.pkl", "wb") as f:
        pickle.dump(out, f)


if __name__ == "__main__":
    main()
