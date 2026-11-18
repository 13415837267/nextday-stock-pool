"""第254轮 C: 弹性刻度进评分 —— 20 日均振幅当 dip 的第 6 个因子

495 笔按信号日的 20 日均振幅(amp20)四分档:

    Q1 低   94 笔  单笔 +0.69%  胜率 55%
    Q2      93 笔  单笔 +1.00%  胜率 60%
    Q3      93 笔  单笔 +1.82%  胜率 58%
    Q4 高   94 笔  单笔 +2.09%  胜率 64%     (去掉 2020-2021 后仍单调)

分年 10 年里 6 年高波更好。与现有评分相关仅 0.37(与池深 0.29), 是独立维度。

机制: dip 买的是"被砸之后的两天", 赔率由这只票平时的波动幅度决定 ——
同样是缩量阴线, 平时每天晃 4% 的票和晃 8% 的票, 反弹空间不是一回事。
而现有评分全是"形态位置"(乖离/缩量/下影线), 没有"弹性刻度"。

本组把 amp20 加进 dip_weights 的 z 分合成, 权重 0 / 0.1 / 0.2 / 0.3 / 0.5,
看能否在笔数不变(选股仍每天 1 只)的前提下提高单笔。
权重 0 = 逐位等价于现状(评分完全不变), 作为基线。
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

WS = [0.0, 0.1, 0.2, 0.3, 0.5]


def main() -> None:
    cfg0 = load_config()
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2500
    ctx = prepare_context(cfg0, n)

    out = []
    for w in WS:
        c = copy.deepcopy(cfg0)
        dw = dict(c["selection"]["dip_weights"])
        dw["amp20"] = w
        c["selection"]["dip_weights"] = dw
        r = run_backtest(c, n, ctx)
        m = r["metrics"]
        tr = pd.DataFrame(r.get("trades_all") or [])
        nm = f"amp20={w:.1f}"
        out.append({"name": nm, "m": m, "trades": tr, "equity": r.get("equity")})
        print(f"\n【{nm}】 年化{m['年化收益率%']} 回撤{m['最大回撤%']} 卡玛{m['卡玛比率']}"
              f" 笔数{m['交易次数']} 胜率{m['胜率%']} 单笔{m['平均单笔收益%']}", flush=True)

    print("\n" + "=" * 92)
    print(f"{'方案':<14}{'年化':>8}{'回撤':>9}{'卡玛':>7}{'笔数':>6}{'胜率':>7}{'单笔':>7}{'空仓':>7}")
    for o in out:
        m = o["m"]
        print(f"{o['name']:<14}{m['年化收益率%']:>8.2f}{m['最大回撤%']:>9.2f}"
              f"{m['卡玛比率']:>7.2f}{m['交易次数']:>6}{m['胜率%']:>7.2f}"
              f"{m['平均单笔收益%']:>7.2f}{m['空仓天数']:>7}")

    with open(ROOT / "output" / f"w254_amp_{n}.pkl", "wb") as f:
        pickle.dump(out, f)
    print(f"\n明细已存 output/w254_amp_{n}.pkl")


if __name__ == "__main__":
    main()
