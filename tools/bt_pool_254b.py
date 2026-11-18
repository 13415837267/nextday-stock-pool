"""第254轮 A: 池深改相对口径 —— 达标票数 / 当日可交易只数

253 把池深档改成"11 只以下不做、11-20 满仓"之后, 那张表仍然是**绝对只数**。
问题: universe 从 2016 年的 2,948 只涨到 2026 年的 5,512 只(近翻倍), 同一个"11 只"
的严格程度在十年里漂了一倍:

    年份   universe   11只=  21只=   当年达标率中位
    2016     2,948   37.3‱  71.2‱      24.1‱
    2020     3,869   28.4‱  54.3‱      31.7‱
    2023     5,398   20.4‱  38.9‱      17.6‱
    2026     5,512   20.0‱  38.1‱      32.7‱

也就是说"池子窄=孤立回调"这件事本身没变, 但"多窄算窄"的尺子在悄悄缩短:
2026 年的 11 只相当于 2016 年的 5.9 只, 门槛严了 87%。

本组把阈值换成 达标率(万分之), 找一条不随 universe 漂移的线:
  A 现状   绝对 21/11/不做        (253 定稿)
  B 相对   25‱ 满仓, 以下不做
  C 相对   20‱
  D 相对   30‱
  E 相对   35‱
  F 相对   25‱ 满仓, 12‱ 以下不做, 中间 0.5 仓(三段式)
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

ABS = [{"min": 21, "mult": 1.0}, {"min": 11, "mult": 1.0}, {"min": 1, "mult": 0.0}]

CASES = {
    "A 绝对 21/11/0": (ABS, None),
    "B 相对 25‱": (ABS, [{"min": 25, "mult": 1.0}, {"min": 1, "mult": 0.0}]),
    "C 相对 20‱": (ABS, [{"min": 20, "mult": 1.0}, {"min": 1, "mult": 0.0}]),
    "D 相对 30‱": (ABS, [{"min": 30, "mult": 1.0}, {"min": 1, "mult": 0.0}]),
    "E 相对 35‱": (ABS, [{"min": 35, "mult": 1.0}, {"min": 1, "mult": 0.0}]),
    "F 相对 25/12": (ABS, [{"min": 25, "mult": 1.0}, {"min": 12, "mult": 0.5},
                            {"min": 1, "mult": 0.0}]),
}


def main() -> None:
    cfg0 = load_config()
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2500
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
    print(f"{'方案':<18}{'年化':>8}{'回撤':>9}{'卡玛':>7}{'笔数':>6}{'胜率':>7}{'单笔':>7}{'空仓':>7}")
    for o in out:
        m = o["m"]
        print(f"{o['name']:<18}{m['年化收益率%']:>8.2f}{m['最大回撤%']:>9.2f}"
              f"{m['卡玛比率']:>7.2f}{m['交易次数']:>6}{m['胜率%']:>7.2f}"
              f"{m['平均单笔收益%']:>7.2f}{m['空仓天数']:>7}")

    with open(ROOT / "output" / f"w254_rate_{n}.pkl", "wb") as f:
        pickle.dump(out, f)
    print(f"\n明细已存 output/w254_rate_{n}.pkl")


if __name__ == "__main__":
    main()
