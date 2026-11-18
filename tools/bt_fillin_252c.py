"""第252轮 C: 池宽 3 落地前的最后三问

252B 得到硬结论:
  · 破位腿"挑开盘最低"是独立优化: 10 只池里挑最低开, 单笔 1.90% -> 3.57%, 年化 +3pp
  · 深低开腿收窄 8->3 是纯损失: 43 笔 1.60% -> 19 笔 0.72%（不是更精, 是更差）
  · 深低开池内"挑最深"与"挑最前"完全一致（3 只里同时达标的极少, 没有选择空间）

于是只剩两件事要试:
  A. 破位 3 / 深低开 8 + 挑最低开 —— 严格"最多三个"做不到, 但只差一条腿
  B. 深低开也压到 3, 用放宽缺口带补回命中率（上界 -5 -> -4.5/-4, 下界 -8 -> -9）
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
    ("S 10/8 只破位最低开", dict(BASE, list_n=10, deep_rank_max=8)),
    ("R 3/8 破位最低开", dict(BASE, list_n=3, deep_rank_max=8)),
    ("P 3/3 上界-4.5", dict(BASE, list_n=3, deep_rank_max=3, deep_hi=-4.5)),
    ("Q 3/3 上界-4.0", dict(BASE, list_n=3, deep_rank_max=3, deep_hi=-4.0)),
    ("T 3/3 下界-9.0", dict(BASE, list_n=3, deep_rank_max=3, deep_lo=-9.0)),
    ("U 3/3 上下都放", dict(BASE, list_n=3, deep_rank_max=3, deep_lo=-9.0, deep_hi=-4.0)),
]

KEYS = ("年化收益率%", "最大回撤%", "卡玛比率", "夏普比率", "交易次数", "胜率%",
        "平均单笔收益%", "空仓天数", "主策略笔数", "补位笔数", "补位·破位",
        "补位·深低开", "补位触发", "补位放弃")


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
            g = fl.groupby("fill_path").agg(n=("ret_pct", "size"), avg=("ret_pct", "mean"))
            print("  路径: " + "; ".join(f"{i} {int(r.n)}笔 {r.avg:.2f}%"
                                         for i, r in g.iterrows()))

    print("\n" + "=" * 96)
    print(f"{'方案':<22}{'年化':>8}{'回撤':>9}{'卡玛':>7}{'笔数':>7}{'补位':>6}"
          f"{'破位':>5}{'深低':>5}{'单笔':>7}{'补位单笔':>9}")
    for o in out:
        m, fl = o["m"], o["fill"]
        avg = fl.ret_pct.mean() if len(fl) else float("nan")
        print(f"{o['name']:<22}{m['年化收益率%']:>8.2f}{m['最大回撤%']:>9.2f}"
              f"{m['卡玛比率']:>7.2f}{m['交易次数']:>7}{m.get('补位笔数', 0):>6}"
              f"{m.get('补位·破位', 0):>5}{m.get('补位·深低开', 0):>5}"
              f"{m['平均单笔收益%']:>7.2f}{avg:>9.2f}")

    with open(ROOT / "output" / f"w252c_mix_{n}.pkl", "wb") as f:
        pickle.dump(out, f)
    print(f"\n明细已存 output/w252c_mix_{n}.pkl")


if __name__ == "__main__":
    main()
