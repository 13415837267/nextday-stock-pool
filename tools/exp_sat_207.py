"""第 207 轮: 副策略的执行面 —— 全史 3645 日, 只动"怎么成交"这三格.

起因(用户): 「叫你通过全历史行情数据研究副策略, 你在搞主策略」。第 205 轮确实做了副策略
(97 万上涨日 + 15 个引擎变体), 但第 206 轮转向了主策略的扩池/分散/限价。本轮把副策略
拉回全史(3645 日, 第 205 轮只到 3000 日), 并且只测三类此前**从没单独扫过**的东西:

  A. 挂单深度 pullback_pct —— 现行 1.0%(挂信号日收盘 ×0.99)。第 194 轮在主策略上测过
     "挂 −1% 等回踩"并否决(成交率 69%、单笔不升), 但副策略的单笔 +1.37%/持 10 日,
     成本占比完全不同, 不能照搬结论。
  B. 挂单窗口 pullback_days —— 现行 3 日。1 日=只在 T+1 试一次; 5 日=更贪但更滞后。
  C. 止损/移动止损 —— 现行 stop −12% / trail 8%。第 192 轮只扫过持有期, 这两格没扫过。

纪律: 不改 config.yaml。所有变体只覆盖 satellite.* 的单个键, 其余全部继承现行配置。
输出 output/exp_sat_207_{days}.json(增量写, 重复跑复用)。

用法: python3 tools/exp_sat_207.py [days] [variant...]
"""
from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context                      # noqa: E402
from src.config import load_config                            # noqa: E402
from src.satellite import entry_ok_series                     # noqa: E402
from tools.satellite_backtest import run                      # noqa: E402

OUT = ROOT / "output"
os.environ["BT_FULL_TRADES"] = "1"

VARIANTS: dict[str, dict] = {
    "base": {},
    # ---- A. 挂单深度: 挂多深才划算(成交率与买价的权衡) ----
    "pb00": {"satellite": {"pullback_pct": 0.0}},      # 不挂限价, 开盘直接买
    "pb05": {"satellite": {"pullback_pct": 0.5}},
    "pb15": {"satellite": {"pullback_pct": 1.5}},
    "pb20": {"satellite": {"pullback_pct": 2.0}},
    "pb30": {"satellite": {"pullback_pct": 3.0}},
    # ---- B. 挂单窗口: 等几天 ----
    "pbd1": {"satellite": {"pullback_days": 1}},
    "pbd2": {"satellite": {"pullback_days": 2}},
    "pbd5": {"satellite": {"pullback_days": 5}},
    # ---- C. 止损 / 移动止损 ----
    "stop08": {"satellite": {"stop_pct": -8.0}},
    "stop10": {"satellite": {"stop_pct": -10.0}},
    "stop15": {"satellite": {"stop_pct": -15.0}},
    "stop20": {"satellite": {"stop_pct": -20.0}},
    "trail00": {"satellite": {"trail_pct": 0.0}},
    "trail05": {"satellite": {"trail_pct": 5.0}},
    "trail10": {"satellite": {"trail_pct": 10.0}},
    "trail12": {"satellite": {"trail_pct": 12.0}},
    # ---- D. 百倍要多少杠杆: 执行面封死后, 只剩"借钱"这一条算术路 ----
    "lev15": {"satellite": {"leverage": 1.5, "margin_rate": 0.055}},
    "lev20": {"satellite": {"leverage": 2.0, "margin_rate": 0.055}},
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
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 3645
    want = sys.argv[2:] or list(VARIANTS)
    cfg = load_config()
    print(f"== 第207轮 副策略 执行面 {days} 日: {len(want)} 个变体 ==", flush=True)
    ctx = prepare_context(cfg, days)
    path = OUT / f"exp_sat_207_{days}.json"
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
        # 成交率: 有信号但没成交的比例(限价没触及) —— 判定 A/B 两组值不值的核心
        tr = r.get("trades") or []
        row["成交笔数"] = float(len(tr))
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
