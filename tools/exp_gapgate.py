"""第一百七十七轮: 高开过滤门槛扫描(单参数, 复用同一份面板)。

动机: 成交票按 T+1 竞价缺口分档(1328 条可复算样本) —
    gap<=-2.5  31 笔 +2.17%/胜率71%   ← 深低开才是 edge
    -2.5~-2     5 笔 +3.16%
    -2~-1      60 笔 +2.20%/65%
    -1~-0.5    54 笔 +0.55%
    -0.5~0     48 笔 +2.10%
    >=0       约 550 笔 合计≈0 且 0~0.5 一档 -0.41%(t=-2.08)
现行 skip_open_gain=10(只拦一字涨停), 等于把整段高开都买下来。
本脚本把门槛从 10 逐档下压, 看"少买高开"是否净改善。
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
    ap.add_argument("--grid", default="10,3,1.5,1.0,0.5,0")
    ap.add_argument("--out", default="output/exp_gapgate.json")
    a = ap.parse_args()

    cfg = load_config()
    ctx = prepare_context(cfg, a.days)
    rows = []
    for sg in [float(x) for x in a.grid.split(",")]:
        c2 = copy.deepcopy(cfg)
        c2["execution"]["skip_open_gain"] = sg
        r = run_backtest(c2, ctx=ctx)
        m = r["metrics"]
        rows.append({"skip_open_gain": sg,
                     "年化%": m["年化收益率%"], "回撤%": m["最大回撤%"],
                     "卡玛": m["卡玛比率"], "夏普": m["夏普比率"],
                     "笔数": m["交易次数"], "单笔%": m["平均单笔收益%"],
                     "胜率%": m["胜率%"]})
        print(f"  skip_open_gain={sg:>5}: 年化{m['年化收益率%']:>8.2f}% "
              f"回撤{m['最大回撤%']:>7.2f}% 卡玛{m['卡玛比率']:>5.2f} "
              f"夏普{m['夏普比率']:>4.2f} {m['交易次数']:>4d}笔 "
              f"单笔{m['平均单笔收益%']:>5.3f}% 胜率{m['胜率%']:>5.2f}%", flush=True)

    Path(ROOT / a.out).write_text(json.dumps(rows, ensure_ascii=False, indent=1),
                                  encoding="utf-8")
    print("写出", a.out)


if __name__ == "__main__":
    main()
