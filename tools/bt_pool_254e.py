"""第254轮 D: 全史 4201 日复核 —— 253 的池深改动与相对口径, 在 17 年上还成立吗

2500 日窗口只覆盖 2016-11 ~ 2026-11。而"绝对只数当门槛"这件事的病根恰恰在更早:
2016 年全市场只有 2,948 只, 2009 年只有 1,700 只左右 —— 那时候"11 只"是今天的 2 倍严格。

所以 253 的定稿(11 只以下不做)在 2500 日上赢, 不代表在 17 年上赢:
  · 2016-2015 的 universe 小, <11 只的日子占比天然更高, 砍掉的可能是"正常日";
  · 反过来, 如果那个年代"池子小"确实等于"没有共振", 砍掉就是对的。

本组跑四件事(4201 日, 2009-07 ~ 2026-11):
  A 253 之前的口径   21→1.0 / 11→0.5 / 1→0.3
  B 253 定稿         21→1.0 / 11→1.0 / 1→0.0
  C 相对口径 25‱
  D 相对口径 35‱
  E 253 定稿 + 11-15 半仓(2500 日上 11-15 档单笔 0.73%, 16-20 档 1.50%)
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
    "A 253前 21/11/0.3": ([{"min": 21, "mult": 1.0}, {"min": 11, "mult": 0.5},
                            {"min": 1, "mult": 0.3}], None),
    "B 253定稿 21/11/0": ([{"min": 21, "mult": 1.0}, {"min": 11, "mult": 1.0},
                            {"min": 1, "mult": 0.0}], None),
    "C 相对 25‱": ([{"min": 21, "mult": 1.0}, {"min": 11, "mult": 1.0},
                     {"min": 1, "mult": 0.0}], [{"min": 25, "mult": 1.0}, {"min": 1, "mult": 0.0}]),
    "D 相对 35‱": ([{"min": 21, "mult": 1.0}, {"min": 11, "mult": 1.0},
                     {"min": 1, "mult": 0.0}], [{"min": 35, "mult": 1.0}, {"min": 1, "mult": 0.0}]),
    "E 253+11-15半仓": ([{"min": 21, "mult": 1.0}, {"min": 16, "mult": 1.0},
                          {"min": 11, "mult": 0.5}, {"min": 1, "mult": 0.0}], None),
}


def main() -> None:
    cfg0 = load_config()
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4201
    # 沙箱 cgroup 只有 8GB: 4201 日 + 补位的 9 个距高窗口会在指标阶段 OOM(EXIT=137)。
    # 池深分档只作用于主策略, 与补位腿无关 —— 本组关掉补位, 只比"主策略在不同池深口径下"。
    # 代价: 净值/笔数与 253 的报告不可比(少了 50 笔补位与它们的资金占用), 但 A~E 之间可比。
    off = os.environ.get("W254_NOFILL", "1") == "1"
    if off:
        cfg0 = copy.deepcopy(cfg0)
        cfg0["fillin"]["enabled"] = False
        print("[注] 补位腿关闭(8GB 内存上限), 只比主策略池深口径")
    ctx = prepare_context(cfg0, n)

    out = []
    for nm, (abs_t, rate_t) in CASES.items():
        c = copy.deepcopy(cfg0)
        c["execution"]["pool_n_tiers"] = copy.deepcopy(abs_t)
        c["execution"]["pool_rate_tiers"] = copy.deepcopy(rate_t) if rate_t else []
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

    tag = f"w254_full{'nofill' if off else ''}_{n}.pkl"
    with open(ROOT / "output" / tag, "wb") as f:
        pickle.dump(out, f)
    print(f"\n明细已存 output/{tag}")


if __name__ == "__main__":
    main()
