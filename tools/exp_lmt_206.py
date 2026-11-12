"""第206轮: 把"追高开"换成"回落才接" —— 成本侧唯一没动过的杠杆.

引擎里已有 buy_limit_fill(默认 0=关闭): 当开盘缺口不够深(高于承接线 floor)时,
不在开盘价追, 而在 floor 价位挂全天限价单, 当日最低价触及才成交, 否则放弃。
实盘含义就是"挂单等回抽", 与副策略的回踩接货同源。

为什么值得单独跑: 全史 839 笔的复利几何均值只有 +0.98%/笔, 而单笔总成本约 0.30%
(佣金 0.025%×2 + 印花 0.05% + 滑点 0.10%×2)。成本占几何均值的 30% ——
入场价每压低 1%, 净收益就多 1%, 相当于把年化从 76% 推到 90% 量级。
代价是高开票可能等不到回落而放弃, 笔数下降。两者谁大, 只能让全史说话。

用法: python3 tools/exp_lmt_206.py [days] [变体...]
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
    # 只对"开盘不高于昨收 +open_max%"的票挂限价, 避免抓"高开后跳水"
    "lf00": {"execution": {"buy_limit_fill": 1, "buy_limit_open_max": 0.0}},
    "lf10": {"execution": {"buy_limit_fill": 1, "buy_limit_open_max": 1.0}},
    "lf20": {"execution": {"buy_limit_fill": 1, "buy_limit_open_max": 2.0}},
    # 放宽承接线: 从"缺口 0%"放到"-1.5%", 让更多日子按限价成交(但入场更深)
    "lfg15": {"execution": {"buy_limit_fill": 1, "buy_limit_open_max": 3.0,
                            "buy_gap_below": -1.5}},
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
    print(f"== 第206轮 限价承接实验 {days} 日: {want} ==", flush=True)
    ctx = prepare_context(cfg, days)
    path = OUT / f"exp_lmt_206_{days}.json"
    res = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    for name in want:
        if name in res:
            print(f"{name:6s} 复用 {res[name]['总收益率%']:.2f}%", flush=True)
            continue
        c = merge(cfg, VARIANTS[name])
        r = run_backtest(c, days, ctx)
        mm = r["metrics"]
        m = {k: mm.get(k, 0.0) for k in
             ("总收益率%", "年化收益率%", "最大回撤%", "卡玛比率", "交易次数", "胜率%")}
        res[name] = m
        path.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{name:6s} 总 {m['总收益率%']:>12.1f}%  年化 {m['年化收益率%']:>6.2f}%  "
              f"回撤 {m['最大回撤%']:>7.2f}%  卡玛 {m['卡玛比率']:>5.2f}  "
              f"笔 {m['交易次数']:>5.0f}  胜率 {m['胜率%']:.1f}%", flush=True)

    print("\n汇总(按卡玛):")
    for k, v in sorted(res.items(), key=lambda x: -(x[1]["卡玛比率"])):
        print(f"  {k:6s} {v['总收益率%']:>12.1f}%  年化 {v['年化收益率%']:>6.2f}%  "
              f"回撤 {v['最大回撤%']:>7.2f}%  卡玛 {v['卡玛比率']:>5.2f}  笔 {v['交易次数']}")


if __name__ == "__main__":
    main()
