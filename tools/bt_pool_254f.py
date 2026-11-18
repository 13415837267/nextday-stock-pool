"""第254轮 E: 253 的"11-20 满仓"要退回半步 —— 11-15 档半仓

3200 日(2013-2026)复核把 253 的问题暴露出来了:

    方案                     年化      最大回撤    卡玛   笔数
    A 253之前 21/11/0.3     99.80%   -21.67%   4.61    752
    B 253定稿 21/11/0      103.87%   -25.81%   4.02    659
    E 21/16/11-半/0        103.88%   -21.67%   4.79    674   ← 最好

B 与 A 的最大回撤都发生在 2015-08-17 → 09-10(股灾 2.0), 但 B 深了 4.1pp。
原因正是 253 把 11-20 档从 0.5 提到 1.0 —— 股灾里普跌、合格池天然只剩十几只,
"池窄"在那几天不是"没有共振", 而是"全市场都在跌", 满仓买进去就是 -25%。

253 的论证是"2500 日上最大回撤逐位不变(-17.21%)", 但 2500 日窗口从 2016-11 开始,
**根本不含 2015 股灾**。这不是数据说 1.0 无害, 是样本里没有能证伪它的那段。

E 的切法是折中: 16 只以上满仓(2500 日 16-20 档单笔 1.50%), 11-15 只半仓
(同口径单笔 0.73%, 不到前者一半), 11 只以下不做(单笔 -0.53%)。

本脚本在 2500 / 1500 两个窗口上复核 A / B / E 三组, 看 E 是否两边都站得住。
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

CASES = {
    "A 253前 21/11/0.3": [{"min": 21, "mult": 1.0}, {"min": 11, "mult": 0.5},
                           {"min": 1, "mult": 0.3}],
    "B 253定稿 21/11/0": [{"min": 21, "mult": 1.0}, {"min": 11, "mult": 1.0},
                           {"min": 1, "mult": 0.0}],
    "E 21/16/11半/0": [{"min": 21, "mult": 1.0}, {"min": 16, "mult": 1.0},
                        {"min": 11, "mult": 0.5}, {"min": 1, "mult": 0.0}],
    "F 21满/16半/11半/0": [{"min": 21, "mult": 1.0}, {"min": 16, "mult": 0.5},
                            {"min": 11, "mult": 0.5}, {"min": 1, "mult": 0.0}],
}


def main() -> None:
    cfg0 = load_config()
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2500
    off = os.environ.get("W254_NOFILL", "1") == "1"
    if off:
        cfg0 = copy.deepcopy(cfg0)
        cfg0["fillin"]["enabled"] = False
    ctx = prepare_context(cfg0, n)

    out = []
    for nm, tiers in CASES.items():
        c = copy.deepcopy(cfg0)
        c["execution"]["pool_n_tiers"] = copy.deepcopy(tiers)
        c["execution"]["pool_rate_tiers"] = []
        r = run_backtest(c, n, ctx)
        m = r["metrics"]
        tr = pd.DataFrame(r.get("trades_all") or [])
        out.append({"name": nm, "m": m, "trades": tr, "equity": r.get("equity")})
        print(f"\n【{nm}】 年化{m['年化收益率%']} 回撤{m['最大回撤%']} 卡玛{m['卡玛比率']}"
              f" 笔数{m['交易次数']} 胜率{m['胜率%']} 单笔{m['平均单笔收益%']}", flush=True)

    print("\n" + "=" * 92)
    print(f"{'方案':<20}{'年化':>8}{'回撤':>9}{'卡玛':>7}{'笔数':>6}{'胜率':>7}{'单笔':>7}{'空仓':>7}")
    for o in out:
        m = o["m"]
        print(f"{o['name']:<20}{m['年化收益率%']:>8.2f}{m['最大回撤%']:>9.2f}"
              f"{m['卡玛比率']:>7.2f}{m['交易次数']:>6}{m['胜率%']:>7.2f}"
              f"{m['平均单笔收益%']:>7.2f}{m['空仓天数']:>7}")

    tag = f"w254_tier{'nofill' if off else ''}_{n}.pkl"
    with open(ROOT / "output" / tag, "wb") as f:
        pickle.dump(out, f)
    print(f"\n明细已存 output/{tag}")


if __name__ == "__main__":
    main()
