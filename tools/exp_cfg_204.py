"""第 204 轮: 把"择时/入场条件"当参数训练 —— 全史面板已经指出真正的抓手在哪.

事实链:
  1) 22.9 万股票日上, dip 池的**日度等权**期望在 2018 年后为负(-0.09%/笔),
     现行 5 因子排序相对池均值只有 +0.04pp(t=0.46)。选股排序没有 alpha。
  2) 但引擎 717 笔真实成交平均 **+1.24%**(中位 0.48%, 胜率 55%)。差在"哪天买"。
  3) 按信号日市场收益分组(真实成交): 市场 <=-4% 时 +1.82%(13 笔), -2.5~-2%
     +3.20%(13 笔), -1.5~-1% +2.35%(47 笔), 而 -0.5~0% 只有 +0.75%(162 笔)。
     面板同一个方向: mkt_ret<=-2 的日子日度均值 +0.6~+2.3%, 微跌日 -0.2%。
  4) 也就是说, **跌得越狠的日子, 这套 dip 越准**; 而 regime 在这些日子把仓位
     压到 blocked_frac=0.40(还要额外要求 T+1 低开 <=-2.5% 才接)。

本脚本只改 regime/execution 这几个旋钮, 不动任何选股参数, 看"放开恐慌日"
到底值多少。全部在同一 ctx 上重算 regime, 指标面板不重建。

用法: python3 tools/exp_cfg_204.py [days]
输出: output/exp_cfg_204_{days}.json(增量写)
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
    "base": {},
    "bf100": {"execution": {"blocked_frac": 1.0}},
    "bf100_gap0": {"execution": {"blocked_frac": 1.0, "block_gap_below": 0}},
    "gap0": {"execution": {"block_gap_below": 0}},
    "bf70": {"execution": {"blocked_frac": 0.70}},
    "ma10": {"regime": {"market_ma": 10}},
    "ma30": {"regime": {"market_ma": 30}},
    "no_breadth": {"regime": {"breadth_min": 0.0}},
    "no_regime": {"regime": {"enabled": False}},
    # ---- 放宽选股条件以换取笔数(复利次数): 面板显示宽池日度均值(-0.051%)
    #      反而略好于旧硬条件池(-0.063%), 但真正的判据只能是引擎 ----
    "L1": {"selection": {"dip_pct_min": -7.0, "dip_pct_max": -2.5,
                         "dip_close_pos_max": 0.40}},
    "L2": {"selection": {"dip_pct_min": -7.0, "dip_pct_max": -2.5,
                         "dip_close_pos_max": 0.40, "dip_mom20_min": 0.0,
                         "dip_mom20_max": 30.0, "dip_dist_low20_min": 3.0,
                         "dip_vol_ratio_max": 5.0}},
    "L3": {"selection": {"dip_pct_min": -7.0, "dip_pct_max": -2.5}},
    "L4": {"selection": {"dip_close_pos_max": 0.40}},
    "L5": {"selection": {"dip_pct_min": -7.0, "dip_pct_max": -2.5,
                         "dip_close_pos_max": 0.40, "dip_dist_low20_min": 3.0}},
    # ---- 全史 ridge 训练出的权重(样本外 +0.04pp, 见 train_204.json) ----
    "W_train": {"selection": {"dip_weights": {
        "mom20": 0.0, "neg_bias20": 0.0, "neg_amount": 0.418,
        "neg_close_pos": 0.211, "neg_vol_ratio": 0.0, "low_tail": 0.013,
        "neg_turnover_cool": 0.150, "ivol": 0.207}}},
    "S08": {"execution": {"min_top1_score": 0.8}},
    "S12": {"execution": {"min_top1_score": 1.2}},
    "S15": {"execution": {"min_top1_score": 1.5}},
    "W_mix": {"selection": {"dip_weights": {
        "mom20": 0.10, "neg_bias20": 0.35, "neg_amount": 0.234,
        "neg_close_pos": 0.106, "neg_vol_ratio": 0.01, "low_tail": 0.022,
        "neg_turnover_cool": 0.075, "ivol": 0.104}}},
}


def merge(base: dict, ov: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in ov.items():
        if isinstance(v, dict):
            out[k] = merge(out.get(k) or {}, v)
        else:
            out[k] = v
    return out


def main() -> None:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    want = sys.argv[2:] or list(VARIANTS)
    cfg = load_config()
    print(f"== 第204轮 regime 实验 {days} 日: {want} ==", flush=True)
    ctx = prepare_context(cfg, days)
    path = OUT / f"exp_cfg_204_{days}.json"
    res = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    for name in want:
        if name in res:
            m = res[name]
            print(f"{name:12s} 复用 {m['总收益率%']:.2f}% / {m['交易次数']} 笔", flush=True)
            continue
        c = merge(cfg, VARIANTS[name])
        r = run_backtest(c, days, ctx)
        _mm = r["metrics"]
        m = {k: _mm.get(k, 0.0) for k in
             ("总收益率%", "年化收益率%", "最大回撤%", "卡玛比率", "交易次数", "胜率%")}
        res[name] = m
        path.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{name:12s} 总 {m['总收益率%']:>12.1f}%  年化 {m['年化收益率%']:>6.2f}%  "
              f"回撤 {m['最大回撤%']:>7.2f}%  卡玛 {m['卡玛比率']:>5.2f}  "
              f"笔 {m['交易次数']:>4.0f}  胜率 {m['胜率%']:.1f}%", flush=True)
    print("\n汇总(按总收益):")
    for k, v in sorted(res.items(), key=lambda x: -(x[1]["总收益率%"])):
        print(f"  {k:12s} {v['总收益率%']:>12.1f}%  年化 {v['年化收益率%']:>6.2f}%  "
              f"回撤 {v['最大回撤%']:>7.2f}%  卡玛 {v['卡玛比率']:>5.2f}  笔 {v['交易次数']}")


if __name__ == "__main__":
    main()
