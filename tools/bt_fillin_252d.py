"""第252轮 D: 池宽 3 的命中率补救 —— 深低开腿改用当日跌幅榜

问题: 深低开腿的候选来自"9 窗口距前高排名"(跌得透), 但真正的触发条件是 T+1 开盘缺口
      -8~-5% —— 隔夜信息, T 日不知道。池宽 8 时靠"总有一只崩到位"兜住命中率(19/43),
      压到 3 后只剩 19 笔, 且剩下的单笔从 1.60% 掉到 0.72%。

假设: "昨天跌得最狠的"比"距前高最远的"更接近"明天接着崩"。
      换成当日跌幅榜前 3, 命中率应显著高于 9 窗口排名前 3。

分组(破位腿一律 3 只 + 挑最低开, 因为 252B 已定):
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
    ("R 3/8 排名源", dict(BASE, list_n=3, deep_rank_max=8)),
    ("H 3/3 排名源", dict(BASE, list_n=3, deep_rank_max=3)),
    ("V 3/3 跌幅榜", dict(BASE, list_n=3, deep_rank_max=3, deep_pool="pct")),
    ("W 3/5 跌幅榜", dict(BASE, list_n=3, deep_rank_max=5, deep_pool="pct")),
    ("X 3/8 跌幅榜", dict(BASE, list_n=3, deep_rank_max=8, deep_pool="pct")),
    ("Y 10/3 跌幅榜", dict(BASE, list_n=10, deep_rank_max=3, deep_pool="pct")),
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
            g = fl.groupby("fill_path").agg(n=("ret_pct", "size"), avg=("ret_pct", "mean"),
                                            win=("ret_pct", lambda s: (s > 0).mean() * 100))
            print("  路径: " + "; ".join(f"{i} {int(r.n)}笔 {r.avg:.2f}% 胜{r.win:.0f}%"
                                         for i, r in g.iterrows()))

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

    with open(ROOT / "output" / f"w252d_pct_{n}.pkl", "wb") as f:
        pickle.dump(out, f)
    print(f"\n明细已存 output/w252d_pct_{n}.pkl")


if __name__ == "__main__":
    main()
