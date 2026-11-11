"""第 201 轮：把"加仓"量化成一条前沿曲线。

日线维度已走完（参数 195 / 买价 194 / 杠杆 196 / 分散 190 / 组合 197 /
副策略 198 / 环境 199 / 分钟 200）。本轮不再找新因子，只回答一个被
"1.0x 不准融资"挡在门外的问题：

**在已经测出边际最高的那些日子（池深 ≥41），多下 20%~50% 的注，
能换来多少年化、付出多少回撤与利息？**

口径说明（重要）：引擎里 `alloc = pick_frac * equity * leverage`，
所以 leverage 与 pos_frac_cap 同时 >1 会**叠乘**（1.3 × 1.3 = 1.69x）。
本轮固定 leverage=1.0，只用 pos_frac_cap + 加仓档表达真实敞口，
这样 1.3 就是 1.3，融资利息按 5.5%/年逐日计提（计入总费用）。
产能口径不变：config 现行 1.0/1.0 与旧结果逐位一致。
"""
import json
import os
import sys
from pathlib import Path

import yaml

os.environ["BT_FULL_TRADES"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context, run_backtest  # noqa: E402

VARIANTS = {
    "base": (1.0, None),
    "pool31_x1.3": (1.3, [{"min": 31, "mult": 1.3}]),
    "pool41_x1.2": (1.2, [{"min": 41, "mult": 1.2}]),
    "pool41_x1.3": (1.3, [{"min": 41, "mult": 1.3}]),
    "pool41_x1.5": (1.5, [{"min": 41, "mult": 1.5}]),
    "pool61_x1.3": (1.3, [{"min": 61, "mult": 1.3}]),
    "all_x1.3": (1.3, [{"min": 0, "mult": 1.3}]),
}


def run(days, name, cap, tiers):
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    cfg["execution"]["pos_frac_cap"] = cap
    cfg["execution"]["leverage"] = 1.0
    if tiers:
        cfg["execution"]["pool_n_boost_tiers"] = tiers
    ctx = prepare_context(cfg, days)
    r = run_backtest(cfg, days=days, ctx=ctx)
    m = r["metrics"]
    pf = [t.get("pos_frac") or 0 for t in r["trades_all"] if t.get("ret_pct") is not None]
    row = dict(days=days, name=name, cap=cap, avg_pos=sum(pf) / len(pf), max_pos=max(pf), **m)
    print(f"{days:5d} {name:13s} 年化 {m['年化收益率%']:7.2f}%  回撤 {m['最大回撤%']:7.2f}%  "
          f"卡玛 {m['卡玛比率']:5.2f}  均仓 {row['avg_pos']:.2f} 最大 {row['max_pos']:.2f}  "
          f"费用 {m['总费用(元)']:>14,.0f}", flush=True)
    return row


if __name__ == "__main__":
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    names = sys.argv[2:] or list(VARIANTS)
    f = ROOT / "output" / f"exp_exposure_201_{days}.json"
    rows = json.load(open(f)) if f.exists() else []
    have = {r["name"] for r in rows}
    for name in names:
        if name in have:
            continue
        cap, tiers = VARIANTS[name]
        try:
            rows.append(run(days, name, cap, tiers))
            json.dump(rows, open(f, "w"), ensure_ascii=False, indent=1)
        except Exception as e:
            print(days, name, "ERR", type(e).__name__, str(e)[:150], flush=True)
    print("saved", f)
