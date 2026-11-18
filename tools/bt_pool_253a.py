"""第253轮 A: 主策略的仓位档 —— 池子大小是择时信号, 不是"仓位旋钮"

第252轮把补位收到一年 4.5 笔后, 边际几乎为零; 真正的大头在主策略本身:
559 笔 / 84% / -18.6%。单吊满仓, 每一笔的仓位档位直接决定净值。

诊断(2500 日, 559 笔主策略)发现一条单调关系 —— 当日合格候选池越小, 次日越差:

    pool_n  1-5    11 笔  单笔 -1.73%  胜率 27%
            6-10   47 笔  单笔 +0.02%  胜率 43%
            11-20 122 笔  单笔 +1.03%  胜率 56%
            21-40 155 笔  单笔 +1.13%  胜率 57%
            41+   143 笔  单笔 +2.26%  胜率 67%

而现有 pool_n_tiers 的方向恰好是反的: <11 给 0.3 仓(还在做), 11-20 只给 0.5 仓
(该满仓却砍半), 21+ 才 1.0。也就是说"池子小"被当成"降低下注"而不是"别下注"。

本实验就是改这张表, 看四件事:
  B: 11-20 满仓 + <11 不做        —— "该满的满, 该停的停"
  C: 11-20 满仓, <11 保持 0.3     —— 只修一半
  D: 11-20 也不做                 —— 更保守(验证 1.03% 是不是噪声)
  E: 分五档 41+/21-40/11-20/6-10/<6
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

TIERS = {
    "A 现状 1.0/0.5/0.3": [
        {"min": 21, "mult": 1.0}, {"min": 11, "mult": 0.5}, {"min": 1, "mult": 0.3}],
    "B 1.0/1.0/0": [
        {"min": 21, "mult": 1.0}, {"min": 11, "mult": 1.0}, {"min": 1, "mult": 0.0}],
    "C 1.0/1.0/0.3": [
        {"min": 21, "mult": 1.0}, {"min": 11, "mult": 1.0}, {"min": 1, "mult": 0.3}],
    "D 1.0/0/0": [
        {"min": 21, "mult": 1.0}, {"min": 11, "mult": 0.0}, {"min": 1, "mult": 0.0}],
    "E 五档 41/21/11/6": [
        {"min": 41, "mult": 1.0}, {"min": 21, "mult": 1.0},
        {"min": 11, "mult": 0.7}, {"min": 6, "mult": 0.0}, {"min": 1, "mult": 0.0}],
}


def main() -> None:
    cfg0 = load_config()
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2500
    ctx = prepare_context(cfg0, n)

    out = []
    for nm, tiers in TIERS.items():
        c = copy.deepcopy(cfg0)
        c["execution"]["pool_n_tiers"] = copy.deepcopy(tiers)
        r = run_backtest(c, n, ctx)
        m = r["metrics"]
        tr = pd.DataFrame(r.get("trades_all") or [])
        out.append({"name": nm, "m": m, "trades": tr, "equity": r.get("equity")})
        print(f"\n【{nm}】 年化{m['年化收益率%']} 回撤{m['最大回撤%']} 卡玛{m['卡玛比率']}"
              f" 笔数{m['交易次数']} 胜率{m['胜率%']} 单笔{m['平均单笔收益%']}")

    print("\n" + "=" * 92)
    print(f"{'方案':<22}{'年化':>8}{'回撤':>9}{'卡玛':>7}{'笔数':>6}{'胜率':>7}{'单笔':>7}{'空仓':>7}")
    for o in out:
        m = o["m"]
        print(f"{o['name']:<22}{m['年化收益率%']:>8.2f}{m['最大回撤%']:>9.2f}"
              f"{m['卡玛比率']:>7.2f}{m['交易次数']:>6}{m['胜率%']:>7.2f}"
              f"{m['平均单笔收益%']:>7.2f}{m['空仓天数']:>7}")

    with open(ROOT / "output" / f"w253a_pooltier_{n}.pkl", "wb") as f:
        pickle.dump(out, f)
    print(f"\n明细已存 output/w253a_pooltier_{n}.pkl")


if __name__ == "__main__":
    main()
