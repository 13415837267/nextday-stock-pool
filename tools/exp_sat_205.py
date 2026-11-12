"""第 205 轮: 在全史上检验副策略的"池边界"与"排序键"值多少.

面板结论(见 train_right_205.json):
  低波前 10% 池等权 H10 = -0.028%; 闸门允许日 +0.213% / 禁止日 -0.554%;
  池内 top1(mom20_ivol) 闸门允许日只有 **-0.565%** —— 比池均值差 0.78pp。
也就是说"挑哪只"是负贡献, 收益来自"哪天买"。本脚本用引擎复核: 只改
satellite.sort_by / ivol_pct_max, 其余(回踩限价/止损/trail/闸门)全部不动。

用法: python3 tools/exp_sat_205.py [days] [variant...]
"""
from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from src.backtest import prepare_context                      # noqa: E402
from src.config import load_config                            # noqa: E402
from src.satellite import entry_ok_series                     # noqa: E402
from tools.satellite_backtest import run                      # noqa: E402

OUT = ROOT / "output"
os.environ["BT_FULL_TRADES"] = "1"

VARIANTS: dict[str, dict] = {
    "base": {},
    # ---- 排序键: 满仓单吊,"挑哪只"直接决定路径 ----
    "s_mom20": {"satellite": {"sort_by": "mom20_desc"}},       # 动量最强(去掉 ivol 分母)
    "s_asc": {"satellite": {"sort_by": "mom20_asc"}},          # 动量最弱
    "s_dh20": {"satellite": {"sort_by": "dist_high20"}},       # 距20日前高最远(超跌)
    "s_cp": {"satellite": {"sort_by": "close_pos_desc"}},      # 收在日内最高(追高)
    "s_amt": {"satellite": {"sort_by": "amount"}},             # 成交额最小
    # ---- 池边界: 现行 ivol 前 10%, 完全放开看它值多少 ----
    "no_pool": {"satellite": {"ivol_pct_max": 0.0}},
    "pool20": {"satellite": {"ivol_pct_max": 0.20}},
    "pool05": {"satellite": {"ivol_pct_max": 0.05}},
    # ---- 逐笔归因: 54% 的笔是"大盘转弱清仓"(均 -0.25%), 74 笔持满 10 日均 +7.33%。
    #      闸门打断的是刚买就失守 MA20 的那批 -> 提高入场缓冲/放宽离场条件各试一遍 ----
    "buf10": {"satellite": {"gate_entry_buffer": 1.0}},
    "buf20": {"satellite": {"gate_entry_buffer": 2.0}},
    "buf30": {"satellite": {"gate_entry_buffer": 3.0}},
    "exit_loser": {"satellite": {"gate_exit_mode": "loser"}},
    "no_exit": {"satellite": {"gate_exit": False}},
    # ---- 最后三个方向: 个股趋势确认 / 分散两只 / 放宽池边界 ----
    "trend": {"satellite": {"mode": "trend_lowvol"}},
    "mp2": {"satellite": {"max_pos": 2, "buys_per_day": 2}},
    "pool20": {"satellite": {"ivol_pct_max": 0.20}},
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
    print(f"== 第205轮 副策略 池/排序键 {days} 日: {want} ==", flush=True)
    ctx = prepare_context(cfg, days)
    path = OUT / f"exp_sat_205_{days}.json"
    res = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    for name in want:
        if name in res:
            print(f"{name:10s} 复用 {res[name]['总收益率%']:.1f}%", flush=True)
            continue
        c = merge(cfg, VARIANTS[name])
        _e = entry_ok_series(ctx.get("reg"), c)
        r = run(ctx, c, entry_ok=_e)
        m = r["metrics"]
        row = {k: float(m.get(k, 0.0)) for k in
               ("总收益率%", "年化收益率%", "最大回撤%", "卡玛比率", "交易次数", "胜率%",
                "平均单笔收益%")}
        res[name] = row
        path.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{name:10s} 总 {row['总收益率%']:>12.1f}%  年化 {row['年化收益率%']:>6.2f}%  "
              f"回撤 {row['最大回撤%']:>7.2f}%  卡玛 {row['卡玛比率']:>5.2f}  "
              f"笔 {row['交易次数']:>5.0f}  单笔 {row['平均单笔收益%']:+.3f}%", flush=True)

    print("\n汇总:")
    for k, v in sorted(res.items(), key=lambda x: -x[1]["总收益率%"]):
        print(f"  {k:10s} {v['总收益率%']:>12.1f}%  年化 {v['年化收益率%']:>6.2f}%  "
              f"回撤 {v['最大回撤%']:>7.2f}%  卡玛 {v['卡玛比率']:>5.2f}  笔 {v['交易次数']:.0f}")


if __name__ == "__main__":
    main()
