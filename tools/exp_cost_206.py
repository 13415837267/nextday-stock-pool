"""第206轮: 成本敏感性 —— 76% 年化里有多少是"假设出来的执行质量".

全史 839 笔、每笔几何 +0.98%。单笔显性成本(现行口径):
  佣金 0.025% x2 + 印花 0.05%(卖出, 2023-08 前 0.1%) + 滑点 0.10% x2 = 约 0.30%
占几何均值的 30%。也就是说"年化 76%"这个数字, 有近三分之一建立在
"每笔只滑 0.1%"这个假设上。本脚本把这个假设推到 0.5%、1.0%,
看策略在什么执行质量下仍然成立 —— 这是比"再调一个参数"更硬的结论。

用法: python3 tools/exp_cost_206.py [days] [变体...]
"""
from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context, run_backtest   # noqa: E402
from src.config import load_config                      # noqa: E402

OUT = ROOT / "output"
os.environ["BT_FULL_TRADES"] = "1"

VARIANTS: dict[str, dict] = {
    "s05": {"execution": {"slippage": 0.0005}},    # 0.05%/边
    "s20": {"execution": {"slippage": 0.002}},     # 0.20%/边
    "s30": {"execution": {"slippage": 0.003}},     # 0.30%/边
    "s50": {"execution": {"slippage": 0.005}},     # 0.50%/边(小票市价单的现实冲击)
    "c50": {"execution": {"commission_rate": 0.0005}},   # 佣金不免五/万5
}


def merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = v
    return out


def main() -> None:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 3645
    want = sys.argv[2:] or list(VARIANTS)
    cfg = load_config()
    print(f"== 第206轮 成本敏感性 {days} 日: {want} ==", flush=True)
    ctx = prepare_context(cfg, days)
    path = OUT / f"exp_cost_206_{days}.json"
    res = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    for name in want:
        if name in res:
            print(f"{name:4s} 复用 {res[name]['总收益率%']:.2f}%", flush=True)
            continue
        c = merge(cfg, VARIANTS[name])
        r = run_backtest(c, days, ctx)
        mm = r["metrics"]
        m = {k: mm.get(k, 0.0) for k in
             ("总收益率%", "年化收益率%", "最大回撤%", "卡玛比率", "交易次数", "胜率%")}
        res[name] = m
        path.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{name:4s} 总 {m['总收益率%']:>12.1f}%  年化 {m['年化收益率%']:>6.2f}%  "
              f"回撤 {m['最大回撤%']:>7.2f}%  卡玛 {m['卡玛比率']:>5.2f}  "
              f"笔 {m['交易次数']:>5.0f}  胜率 {m['胜率%']:.1f}%", flush=True)

    print("\n汇总(按总收益):")
    for k, v in sorted(res.items(), key=lambda x: -(x[1]["总收益率%"])):
        print(f"  {k:4s} {v['总收益率%']:>12.1f}%  年化 {v['年化收益率%']:>6.2f}%  "
              f"回撤 {v['最大回撤%']:>7.2f}%  卡玛 {v['卡玛比率']:>5.2f}  笔 {v['交易次数']}")


if __name__ == "__main__":
    main()
