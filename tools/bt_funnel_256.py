"""第256轮: 补位从"3 只里挑最便宜的"改成"漏斗式一层一只"

用户: "我不要这么多候选计划补位, 我只要计划一个补位一个, 或者漏斗式一个一个地
       补位最多三层"。

现状(252 定稿): 名单 3 只, 破位腿在 3 只里挑"开盘缺口最深"的那只(pick_break=lowest)。
  代价: 早上要同时盯 3 只, 决策是"三选一"; 而 252 测出 lowest 比 best 单笔高 0.87pp
  (1.90% -> 2.77%) —— 这是"挑最便宜"的收益, 不是"排名"的收益。

改法(漏斗): 名单仍是 3 只, 但按名次一层一只, 第 1 层不合格(开盘不够低/不在 -8~-5 带)
  才看第 2 层, 依次到第 3 层。这就是 choose(mode="best") 的原生行为。

对照:
  A 现状      3/3 破位 lowest   (三选一, 挑最便宜)
  B 漏斗3层   3/3 破位 best     (一层一只, 最多三层)
  C 只补1只   1/1 破位 best     (没有第 2 层可退)
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
            dist_high60_max=-42.0, deep_lo=-8.0, deep_hi=-5.0, deep_skip=3)

GROUPS = [
    ("A 现状 3选1", dict(BASE, list_n=3, deep_rank_max=3,
                        pick_break="lowest", pick_deep="best")),
    ("B 漏斗三层", dict(BASE, list_n=3, deep_rank_max=3,
                       pick_break="best", pick_deep="best")),
    ("C 只补1只", dict(BASE, list_n=1, deep_rank_max=1,
                      pick_break="best", pick_deep="best")),
]

KEYS = ("年化收益率%", "最大回撤%", "卡玛比率", "交易次数", "胜率%",
        "平均单笔收益%", "空仓天数", "补位笔数", "补位·破位", "补位·深低开",
        "补位触发", "补位放弃")


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
        avg = fl.ret_pct.mean() if len(fl) else float("nan")
        print(f"\n【{nm}】 年化{m['年化收益率%']:.2f} 回撤{m['最大回撤%']:.2f}"
              f" 卡玛{m['卡玛比率']:.2f} 笔数{m['交易次数']} 补位{m.get('补位笔数', 0)}"
              f" 补位单笔{avg:.2f}% 放弃{m.get('补位放弃', 0)}", flush=True)
        if len(fl) and "fill_path" in fl.columns:
            g = fl.groupby("fill_path").agg(n=("ret_pct", "size"), avg=("ret_pct", "mean"))
            print("  路径: " + "; ".join(f"{i} {int(r.n)}笔 {r.avg:.2f}%" for i, r in g.iterrows()))
            if "fill_rk" in fl.columns:
                print("  名次: " + " ".join(
                    f"{int(k)}:{v}" for k, v in fl.fill_rk.value_counts().sort_index().items()))

    print("\n" + "=" * 92)
    print(f"{'方案':<14}{'年化':>8}{'回撤':>9}{'卡玛':>7}{'笔数':>6}{'补位':>6}"
          f"{'破位':>5}{'深低':>5}{'补位单笔':>9}{'放弃':>6}")
    for o in out:
        m, fl = o["m"], o["fill"]
        avg = fl.ret_pct.mean() if len(fl) else float("nan")
        print(f"{o['name']:<14}{m['年化收益率%']:>8.2f}{m['最大回撤%']:>9.2f}"
              f"{m['卡玛比率']:>7.2f}{m['交易次数']:>6}{m.get('补位笔数', 0):>6}"
              f"{m.get('补位·破位', 0):>5}{m.get('补位·深低开', 0):>5}"
              f"{avg:>9.2f}{m.get('补位放弃', 0):>6}")

    with open(ROOT / "output" / f"w256_funnel_{n}.pkl", "wb") as f:
        pickle.dump(out, f)
    print(f"\n明细已存 output/w256_funnel_{n}.pkl")


if __name__ == "__main__":
    main()
