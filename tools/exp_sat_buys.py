# -*- coding: utf-8 -*-
"""第 163 轮留档: 副策略"一次买满"的对照实验(buys_per_day = 1/2/3/4)。

结论: buys 从 1 提到 4(等于 max_pos), 年化 11.34% -> 20.28%(引擎口径)、
单笔 1.00% -> 1.44%, 且四个取值单调 —— 提升来自"买到当日信号本身, 而不是三天前的信号",
不是"多买摊薄"。

用法: python tools/exp_sat_buys.py [--days 1500]

为什么不用候选表缓存: 本脚本刻意只走官方引擎(tools/satellite_backtest.py 的 run),
保证与实盘账本同口径。配置改的是内存里的 dict, 不写回 config.yaml。
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context                     # noqa: E402
from src.config import load_config                           # noqa: E402
from tools.satellite_backtest import run                     # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=1500)
    ap.add_argument("--max-pos", type=int, default=4)
    a = ap.parse_args()

    cfg = load_config()
    ctx = prepare_context(cfg, a.days)          # 面板只装一次, 四组复用
    print(f"\n窗口 {a.days} 日 | max_pos={a.max_pos} hold={cfg['satellite']['hold_days']}\n")
    print(f"{'buys':>4s} {'笔数':>5s} {'总收益':>10s} {'年化':>8s} {'回撤':>8s} "
          f"{'卡玛':>6s} {'胜率':>7s} {'单笔':>7s}")
    base = None
    for b in (1, 2, 3, 4):
        c = copy.deepcopy(cfg)
        c["satellite"]["max_pos"] = a.max_pos
        c["satellite"]["buys_per_day"] = b
        r = run(ctx, c)
        m = r["metrics"]
        if base is None:
            base = m["年化收益率%"]
        print(f"{b:4d} {m['交易次数']:5d} {m['总收益率%']:9.2f}% {m['年化收益率%']:7.2f}% "
              f"{m['最大回撤%']:7.2f}% {m['卡玛比率']:6.2f} {m['胜率%']:6.2f}% "
              f"{m['平均单笔收益%']:6.3f}%", flush=True)
    print(f"\n提示: buys=1 的 {base:.2f}% 是本次运行的旧口径基准, 与 config 现值不同属正常 ——"
          f" config 现值已经是 buys=4。")
    print("注: 本脚本只做对照, 不写 output/。要更新线上数字请跑 tools/satellite_backtest.py。")


if __name__ == "__main__":
    main()
