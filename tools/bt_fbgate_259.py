"""第259轮: 评分不达标的日子, 备用族能不能接

现状(第91轮单槽漏斗): 主力池当日**有候选**就只做主力, 备用族(momentum/oversold/pullback)
只在"主力完全没票"的日子补位。

但 3200 日的跳过记录显示, 空仓有两个来源, 性质完全不同:
  · 缺口未达承接线  1585 天 (64%) —— 主力出票, 但开盘没跌到位, 逆向选择最强的一天(91 轮已证)
  · 评分低于门槛     886 天 (36%) —— 主力出票, 但票本身就差(T 日收盘即知)

后者在**选择阶段**就没算备用池, 所以账面空仓的理由不是"今天没机会", 而是"唯一的机会质量差"。
而 momentum/oversold/pullback 三族是独立形态, 与"今天的 dip 票评分低"并不矛盾。

本脚本把"评分门触发时并入备用池"做成开关(fb_on_score_gate), 在引擎里量它值多少。

  A 现状(单槽漏斗)
  B 评分不达标 -> 三族全接
  C 评分不达标 -> 只接 momentum(样本最大)
  D 评分不达标 -> 接 momentum + oversold
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

ALL3 = [
    {"mode": "momentum", "top_n": 1, "gap_below": 0, "segment": "all"},
    {"mode": "oversold", "top_n": 1, "gap_below": 0, "segment": "all"},
    {"mode": "pullback", "top_n": 1, "gap_below": 0, "segment": "all"},
]

GROUPS = [
    ("A 现状", dict()),
    ("B 三族接", dict(on=True, fbs=ALL3)),
    ("C 只momentum", dict(on=True, fbs=ALL3[:1])),
    ("D mom+oversold", dict(on=True, fbs=ALL3[:2])),
    ("F 全顺延", dict(slot=False)),
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
        c["selection"]["fb_on_score_gate"] = bool(ov.get("on"))
        if ov.get("slot") is False:
            c["selection"]["fallback_slot"] = False
        if ov.get("fbs") is not None:
            c["selection"]["fallbacks"] = copy.deepcopy(ov["fbs"])
        r = run_backtest(c, n, ctx)
        m = r["metrics"]
        tr = pd.DataFrame(r.get("trades_all") or [])
        real = tr[tr["ret_pct"].notna()] if len(tr) else tr
        fb = real[real["pool"].astype(str).str.startswith("fb")] if len(real) else real
        fl = real[real["_fillin"].fillna(False).astype(bool)] if len(real) and "_fillin" in real else real
        print(f"{nm:<14} 年化 {m['年化收益率%']:>7.2f}%  回撤 {m['最大回撤%']:>7.2f}%  "
              f"卡玛 {m['卡玛比率']:>5.2f}  笔数 {m['交易次数']:>4}"
              f"  备用 {len(fb)}"
              + (f" 备用单笔 {fb['ret_pct'].mean():.2f}%" if len(fb) else "")
              + (f"  补位 {len(fl)}" if len(fl) else ""), flush=True)
        if len(fb):
            g = fb.groupby("pool").ret_pct.agg(["size", "mean"])
            print("     族:", " ".join(f"{k}:{int(v['size'])}笔{v['mean']:.2f}%" for k, v in g.iterrows()),
                  flush=True)
        out.append(dict(name=nm, metrics=m, trades=tr, equity=r.get("equity") or []))

    with open(ROOT / f"output/w259_fbgate_{n}.pkl", "wb") as f:
        pickle.dump(out, f)


if __name__ == "__main__":
    main()
