# -*- coding: utf-8 -*-
"""多重比较校正后的夏普（Deflated Sharpe Ratio, Bailey & López de Prado 2014）。

回答一个问题：**在 N 次试验里挑出来的最高夏普，有多少可能是运气？**

    SR0 = sqrt(V) · [(1-γ)·Z⁻¹(1-1/N) + γ·Z⁻¹(1-1/(N·e))]      # 纯噪声下的期望最大值
    V   = (1 - γ3·SR + (γ4-1)/4·SR²) / (T-1)                     # 日频（缺省正态近似）
    DSR = Φ[(SR - SR0) / sqrt(V)]

判读：
    DSR >= 0.95  即便按 N 次独立试验计，该夏普仍非纯噪声
    DSR <  0.95  不能排除"从 N 次噪声试验中挑出最好"这一解释

**边界说明（重要）**：DSR 只排除"纯噪声"，不保证样本外能复制该夏普；
它假设收益近似正态，对右偏严重的策略（单吊、少数大赢家）偏乐观；
N 应取**有效独立试验数**，高度相关的变体不能简单相加（本仓库按 500~2500 给区间）。

用法：
    python tools/deflated_sharpe.py --sharpe 2.55 --days 1500 --trials 2500
    python tools/deflated_sharpe.py --sharpe 2.55,2.12 --days 1500,3377 --trials 500,1000,2500
"""
from __future__ import annotations

import argparse
import math
from statistics import NormalDist

Z = NormalDist().inv_cdf
PHI = NormalDist().cdf
GAMMA = 0.5772156649015329  # Euler–Mascheroni


def deflated_sharpe(sr_annual: float, days: int, trials: int) -> tuple[float, float, float]:
    """返回 (DSR, 噪声上限 SR0(年化), 夏普标准误(年化))。"""
    sr = sr_annual / math.sqrt(252)            # 日频夏普
    var = (1 + sr * sr / 2) / (days - 1)       # 正态近似（γ3=0, γ4=3）
    sr0 = math.sqrt(var) * ((1 - GAMMA) * Z(1 - 1 / trials)
                            + GAMMA * Z(1 - 1 / (trials * math.e)))
    return PHI((sr - sr0) / math.sqrt(var)), sr0 * math.sqrt(252), math.sqrt(var) * math.sqrt(252)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sharpe", default="2.55", help="年化夏普，逗号分隔多个")
    ap.add_argument("--days", default="1500", help="样本交易日数，与 sharpe 一一对应或单值广播")
    ap.add_argument("--trials", default="500,1000,2000,2500", help="试验次数，逗号分隔")
    a = ap.parse_args()

    sharpes = [float(x) for x in a.sharpe.split(",")]
    days = [int(x) for x in a.days.split(",")]
    trials = [int(x) for x in a.trials.split(",")]
    if len(days) == 1:
        days *= len(sharpes)

    print(f"{'夏普(年化)':>11} {'样本日数':>9} {'试验数N':>8} {'噪声上限SR0':>12} "
          f"{'SR标准误':>9} {'DSR':>8}  判定")
    print("-" * 84)
    any_fail = False
    for sr, T in zip(sharpes, days):
        for N in trials:
            d, sr0, se = deflated_sharpe(sr, T, N)
            ok = d >= 0.95
            any_fail |= not ok
            print(f"{sr:>11.2f} {T:>9d} {N:>8d} {sr0:>12.2f} {se:>9.2f} "
                  f"{d * 100:>7.2f}%  {'通过' if ok else '不显著'}")

    print("-" * 84)
    print("DSR 通过 ≠ 样本外可复制。它只是把\"纯噪声\"这一解释排除掉，")
    print("剩下的成分有多少来自结构、多少来自选择偏差，仍需 rank_stability + random_baseline 交叉判断。")
    return 1 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
