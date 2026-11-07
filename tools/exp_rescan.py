# -*- coding: utf-8 -*-
"""第一百六十五轮: 印花税口径变更后的主策略参数复扫。

为什么复扫
----------
第一百一十八轮确立的原则: "口径变了, 峰就未必还在原来的位置"(当时 PIT 修正后
bias20_max 的最优值从 8 漂到 11)。第一百六十二轮把印花税从"全程万 5"改成
"2023-08-28 前千 1 / 之后万 5"——这是又一次口径变更, 之后主策略参数从未复扫。

结论
----
六组核心边界(22 档)在 1500 日窗口全部**没有漂移**: 现值全部落在峰或平台上。
唯一"更好"的是 blocked_frac=0.5(+0.78pp), 但它是已知的 1500 日窗口陷阱
(全史回撤 -31.59%、卡玛 2.46 < 2.69, 见第一百一十六/一百一十八轮), 不采纳。

用法: python tools/exp_rescan.py            # 全部 6 组
      python tools/exp_rescan.py --fast     # 只跑 3 组核心
"""
import copy
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.backtest import prepare_context, run_backtest
from src.config import load_config

FAST = "--fast" in sys.argv


def main():
    cfg0 = load_config()
    ctx = prepare_context(cfg0, 1500)

    def run(patch, tag):
        cfg = copy.deepcopy(cfg0)
        for path, val in patch.items():
            a, b = path.split(".")
            cfg[a][b] = val
        m = run_backtest(cfg, 1500, ctx)["metrics"]
        row = dict(tag=tag, **{k: m[k] for k in
                   ["总收益率%", "年化收益率%", "最大回撤%", "卡玛比率",
                    "交易次数", "胜率%", "平均单笔收益%"]})
        print(f"{tag:<32} 年化 {row['年化收益率%']:7.2f}%  笔 {row['交易次数']:>4}  "
              f"回撤 {row['最大回撤%']:7.2f}%  卡玛 {row['卡玛比率']:.2f}  "
              f"单笔 {row['平均单笔收益%']:.3f}%", flush=True)
        return row

    rows = [run({}, "基线(现值)")]
    grids = {
        "bias20_max": ("selection.bias20_max", [9.0, 11.4, 13.0, 15.0]),
        "dip_vol_ratio_max": ("selection.dip_vol_ratio_max", [3.5, 4.0, 4.5, 5.0]),
        "min_top1_score": ("execution.min_top1_score", [0.85, 1.0, 1.15, 1.3]),
        "blocked_frac": ("execution.blocked_frac", [0.3, 0.4, 0.5]),
        "market_ma": ("regime.market_ma", [18, 20, 22, 25]),
        "dip_close_pos_max": ("selection.dip_close_pos_max", [0.28, 0.32, 0.36]),
    }
    if FAST:
        grids = {k: grids[k] for k in ["bias20_max", "min_top1_score", "market_ma"]}
    for name, (path, vals) in grids.items():
        print(f"\n--- {name} ---", flush=True)
        for v in vals:
            rows.append(run({path: v}, f"{name}={v}"))

    json.dump(rows, open(os.path.join(ROOT, "output", "exp_rescan_r165.json"), "w"),
              ensure_ascii=False, indent=1)
    print("\nsaved output/exp_rescan_r165.json")


if __name__ == "__main__":
    main()
