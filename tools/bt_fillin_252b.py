"""第252轮 B: 池宽 3 的两种补救 —— 换"怎么挑"而不是"挑几个"

252A 的结论: 池宽 10/8 -> 3/3, 补位 71 -> 47 笔, 卡玛 5.37 -> 4.62。
掉的地方全在深低开(43 -> 19 笔): 原来前 8 名里总有一只跌进 -8~-5 的带子,
收窄到 3 名后就常常"今天没有一只跌到位" -> 不做了。

于是问题从"挑得到吗"变成"挑得对吗": 3 只里如果有多只合格,
该挑排名最前的(best, 现状)还是开盘最便宜的(lowest)?
  · 破位腿: 合格线是"开盘 <= 昨收"(低开即买), 挑最便宜 = 买得最低
  · 深低开腿: 合格带是 -8~-5, 挑最便宜 = 更容易滑到 -8 以下那一档(251 实测 0.99%)

分组: 6 组, 共用 ctx。
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
            dist_high60_max=-42.0, deep_lo=-8.0)

GROUPS = [
    ("B 251定稿 10/8", dict(BASE, list_n=10, deep_rank_max=8)),
    ("D 3/3 排名优先", dict(BASE, list_n=3, deep_rank_max=3)),
    ("H 3/3 破位最低开", dict(BASE, list_n=3, deep_rank_max=3, pick_break="lowest")),
    ("I 3/3 深低开最深", dict(BASE, list_n=3, deep_rank_max=3, pick_deep="lowest")),
    ("J 3/3 都最深", dict(BASE, list_n=3, deep_rank_max=3,
                        pick_break="lowest", pick_deep="lowest")),
    ("K 10/8 都最深", dict(BASE, list_n=10, deep_rank_max=8,
                        pick_break="lowest", pick_deep="lowest")),
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
            print(f"  补位: {len(fl)} 笔 单笔 {fl.ret_pct.mean():.2f}% "
                  f"胜率 {(fl.ret_pct > 0).mean() * 100:.0f}%")
            if "fill_rk" in fl.columns:
                print("  名次分布: " + " ".join(
                    f"{int(k)}:{v}" for k, v in fl.fill_rk.value_counts().sort_index().items()))
                g = fl.groupby("fill_path").agg(n=("ret_pct", "size"), avg=("ret_pct", "mean"))
                print("  路径: " + "; ".join(f"{i} {int(r.n)}笔 {r.avg:.2f}%"
                                             for i, r in g.iterrows()))

    print("\n" + "=" * 96)
    print(f"{'方案':<20}{'年化':>8}{'回撤':>9}{'卡玛':>7}{'笔数':>7}{'补位':>6}"
          f"{'破位':>5}{'深低':>5}{'单笔':>7}{'补位单笔':>9}{'放弃':>6}")
    for o in out:
        m, fl = o["m"], o["fill"]
        avg = fl.ret_pct.mean() if len(fl) else float("nan")
        print(f"{o['name']:<20}{m['年化收益率%']:>8.2f}{m['最大回撤%']:>9.2f}"
              f"{m['卡玛比率']:>7.2f}{m['交易次数']:>7}{m.get('补位笔数', 0):>6}"
              f"{m.get('补位·破位', 0):>5}{m.get('补位·深低开', 0):>5}"
              f"{m['平均单笔收益%']:>7.2f}{avg:>9.2f}{m.get('补位放弃', 0):>6}")

    # 破位腿: 名次 1 和 2~3 的差别(回答"往后挑是不是更差")
    print("\n【破位腿 按名次】")
    for o in out:
        fl = o["fill"]
        if not len(fl) or "fill_path" not in fl.columns:
            continue
        b = fl[fl.fill_path == "破位"]
        if len(b):
            r1 = b[b.fill_rk == 1]
            r2 = b[b.fill_rk > 1]
            print(f"  {o['name']:<20} 第1名 {len(r1):>3}笔 {r1.ret_pct.mean() if len(r1) else float('nan'):>6.2f}%"
                  f"   第2+ {len(r2):>3}笔 {r2.ret_pct.mean() if len(r2) else float('nan'):>6.2f}%")

    with open(ROOT / "output" / f"w252b_pick_{n}.pkl", "wb") as f:
        pickle.dump(out, f)
    print(f"\n明细已存 output/w252b_pick_{n}.pkl")


if __name__ == "__main__":
    main()
