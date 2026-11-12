"""第206轮: 把"主板独占"这条 16 年没动过的边界交给全史判.

诊断(diag_board_206)已证: 创业板在现行每一条 universe 门槛上的通过率都不低于主板
(价 97.8% vs 94.2%, 额 52.8% vs 52.6%, 换手 36.7% vs 25.2%), 每日可入池 426 只。
它不是被门槛挡住的, 是被 `universe.boards: [主板]` 一句话排除的 —— 纯增量池深 +57%。

信号侧的不确定性: dip 门槛 -7~-2.5% 在 10cm 票上是"显著超跌", 在 20cm 票上只相当于
-3.5~-1.25%, 可能意味着"跌得不够"。所以必须让引擎跑, 而不是看面板均值。

本脚本不重建指标面板(创业板指标已在 ctx 里), 只按新板块集重算风控基础序列。
用法: python3 tools/exp_board_206.py [days] [变体名...]
"""
from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context, run_backtest   # noqa: E402
from src.config import board_of, load_config             # noqa: E402

OUT = ROOT / "output"
os.environ["BT_FULL_TRADES"] = "1"

VARIANTS: dict[str, dict] = {
    "base":      {"universe": {"boards": ["主板"]}},
    "cyb":       {"universe": {"boards": ["主板", "创业板"]}},
    "cyb_only":  {"universe": {"boards": ["创业板"]}},
    "all3":      {"universe": {"boards": ["主板", "创业板", "科创板"]}},
}


def reg_base_for(ctx: dict, boards: list[str]) -> pd.DataFrame:
    """按给定板块集重算风控基础序列(全市场等权指数 + 宽度)。

    与 src.selector.regime_base 同口径(非停/非ST/bars>=21), 逐日聚合以免拼接长表。
    """
    bs = set(boards)
    rows = []
    for d in sorted(ctx["by_date"].keys()):
        day = ctx["by_date"][d]
        if day is None or not len(day):
            continue
        m = (day["is_paused"].values == 0) & (day["is_st"].values == 0) & (day["bars"].values >= 21)
        sub = day.loc[m, ["ts_code", "above_ma20", "pct"]]
        sub = sub[sub["ts_code"].map(lambda c: board_of(c) in bs)]
        if not len(sub):
            continue
        rows.append({"trade_date": d, "breadth": float(sub["above_ma20"].mean()),
                     "mkt_pct": float(sub["pct"].mean())})
    reg = pd.DataFrame(rows).set_index("trade_date").sort_index()
    reg["mkt_idx"] = (1 + reg["mkt_pct"] / 100).cumprod()
    return reg


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
    print(f"== 第206轮 板块边界实验 {days} 日: {want} ==", flush=True)
    ctx = prepare_context(cfg, days)
    path = OUT / f"exp_board_206_{days}.json"
    res = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    cache: dict[tuple, pd.DataFrame] = {}

    for name in want:
        if name in res:
            m = res[name]
            print(f"{name:10s} 复用 {m['总收益率%']:.2f}% / {m['交易次数']} 笔", flush=True)
            continue
        ov = VARIANTS[name]
        c = merge(cfg, ov)
        key = tuple(sorted(ov["universe"]["boards"]))
        if key not in cache:
            cache[key] = reg_base_for(ctx, list(key))
        cx = {**ctx, "reg_base": cache[key]}
        r = run_backtest(c, days, cx)
        mm = r["metrics"]
        m = {k: mm.get(k, 0.0) for k in
             ("总收益率%", "年化收益率%", "最大回撤%", "卡玛比率", "交易次数", "胜率%")}
        res[name] = m
        path.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{name:10s} 总 {m['总收益率%']:>12.1f}%  年化 {m['年化收益率%']:>6.2f}%  "
              f"回撤 {m['最大回撤%']:>7.2f}%  卡玛 {m['卡玛比率']:>5.2f}  "
              f"笔 {m['交易次数']:>4.0f}  胜率 {m['胜率%']:.1f}%", flush=True)

    print("\n汇总(按总收益):")
    for k, v in sorted(res.items(), key=lambda x: -(x[1]["总收益率%"])):
        print(f"  {k:10s} {v['总收益率%']:>12.1f}%  年化 {v['年化收益率%']:>6.2f}%  "
              f"回撤 {v['最大回撤%']:>7.2f}%  卡玛 {v['卡玛比率']:>5.2f}  笔 {v['交易次数']}")


if __name__ == "__main__":
    main()
