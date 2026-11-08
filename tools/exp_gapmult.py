"""第一百七十七轮(二): 高开降仓 mult 重扫 —— 从"砍掉"反过来看"加压"。

上一支脚本证明砍高开是负的:
    skip_open_gain 10→3→1.5→1→0.5→0
    年化 180.7/178.3/155.5/163.5/153.4/100.7%, 笔数 352→223, 单笔 1.44%→1.38%
单笔几乎不动而笔数掉 37% —— 被砍掉的成交是**零期望但正右偏**, 在复利下仍是正贡献。

既然如此, 第 97 轮那条 `gap_above_tiers: {above:0.5, score_below:1.6, mult:0.30}`
(高开且评分未达满仓线 → 仓位 ×0.3) 可能已经**压得过头**。那是 545 笔旧口径下调的,
之后 min_pool / gap_floor / ma20 三个口径参数都动过。本轮在现行口径下重扫 mult。
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config          # noqa: E402
from src.backtest import prepare_context, run_backtest   # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=1500)
    ap.add_argument("--grid", default="0.3,0.4,0.5,0.7,1.0")
    ap.add_argument("--out", default="output/exp_gapmult.json")
    a = ap.parse_args()

    cfg = load_config()
    ctx = prepare_context(cfg, a.days)
    rows = []
    for mv in [float(x) for x in a.grid.split(",")]:
        c2 = copy.deepcopy(cfg)
        tiers = list((c2["execution"].get("gap_above_tiers") or []))
        if tiers:
            tiers[0] = {**tiers[0], "mult": mv}
        c2["execution"]["gap_above_tiers"] = tiers
        r = run_backtest(c2, ctx=ctx)
        m = r["metrics"]
        rows.append({"mult": mv, "年化%": m["年化收益率%"], "回撤%": m["最大回撤%"],
                     "卡玛": m["卡玛比率"], "夏普": m["夏普比率"],
                     "笔数": m["交易次数"], "单笔%": m["平均单笔收益%"],
                     "胜率%": m["胜率%"]})
        print(f"  mult={mv:>4}: 年化{m['年化收益率%']:>8.2f}% 回撤{m['最大回撤%']:>7.2f}% "
              f"卡玛{m['卡玛比率']:>5.2f} 夏普{m['夏普比率']:>4.2f} "
              f"{m['交易次数']:>4d}笔 单笔{m['平均单笔收益%']:>5.3f}% "
              f"胜率{m['胜率%']:>5.2f}%", flush=True)

    Path(ROOT / a.out).write_text(json.dumps(rows, ensure_ascii=False, indent=1),
                                  encoding="utf-8")
    print("写出", a.out)


if __name__ == "__main__":
    main()
