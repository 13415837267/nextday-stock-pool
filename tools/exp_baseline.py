# -*- coding: utf-8 -*-
"""基线快照 —— 指定窗口跑一次现行配置, 只输出指标并写 output/exp_baseline_<days>.json。

为什么需要它
------------
`src/backtest.py` 的默认窗口是 config.backtest.trading_days(1500), 全史(3378 日)只能靠
命令行 `--days 3378` 跑, 而沙盒没有 data/panel(500MB+), 跑不了。research.yml 只接受
`tools/<script>.py`, 所以把"全史基线"做成一个只读脚本 —— 它不写 output/backtest.json,
不会覆盖页面数据。

用途: 每次引擎口径变更后, 用它给出"全史"面的一行对照(1500 日由 daily 工作流自动刷新)。

用法
----
  python3 tools/exp_baseline.py                # 3378 日(全史)
  python3 tools/exp_baseline.py --days 1500
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context, run_backtest  # noqa: E402
from src.config import load_config  # noqa: E402

FIELDS = ["总收益率%", "年化收益率%", "最大回撤%", "卡玛比率", "夏普比率",
          "交易次数", "胜率%", "平均单笔收益%", "风控空仓天数"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=3378)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config()
    print("[1/2] 构建上下文 %d 日 ..." % args.days, flush=True)
    t0 = time.time()
    ctx = prepare_context(cfg, args.days)
    print("      ctx ready %.0fs" % (time.time() - t0), flush=True)

    print("[2/2] 回测 ...", flush=True)
    res = run_backtest(cfg, args.days, ctx)
    m = res["metrics"]
    print("\n=== 基线(%d 日窗口, %.0fs) ===" % (args.days, time.time() - t0), flush=True)
    for k in FIELDS:
        print("  %-14s %s" % (k, m.get(k)), flush=True)
    print("  区间 %s ~ %s" % (res["meta"]["start"], res["meta"]["end"]), flush=True)

    op = Path(args.out) if args.out else ROOT / ("output/exp_baseline_%d.json" % args.days)
    if not op.is_absolute():
        op = ROOT / op
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps({"days": args.days, "metrics": m, "meta": res["meta"],
                              "config_fp": cfg.get("_fp")},
                             ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print("\n结果已写 %s" % op, flush=True)


if __name__ == "__main__":
    main()
