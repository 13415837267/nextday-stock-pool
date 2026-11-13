"""第225轮C · 执行层仓位映射扫描（低开深度 × 仓位折减）

为什么扫这个: 选股权重刚刚在真实回测里全败, 说明"选谁"已到边际;
但"下多少注"这一层从未被网格化过 —— 现行规则是
  block_gap_below=-2.5 (T+1 低开超过 2.5%) -> blocked_frac=0.4 (只买 40%)
这条规则是人工拍的, 而第221轮可转债/第222轮ETF都显示"深低开才是收益最厚的一档",
若对股票同样成立, 现行规则恰好丢掉最肥的一段。

口径: 复用 prepare_context, 每组只跑模拟, 双窗口(1500日 + 全史)交叉验证。
"""
from __future__ import annotations

import copy
import itertools
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context, run_backtest  # noqa: E402
from src.config import load_config  # noqa: E402


def brief(m: dict) -> tuple:
    return (m["年化收益率%"], m["最大回撤%"], m["卡玛比率"], m["交易次数"], m["胜率%"])


def main() -> None:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    cfg = load_config()
    ctx = prepare_context(cfg, days)
    print(f"上下文 {len(ctx['main_dates'])} 日", flush=True)

    b = brief(run_backtest(cfg, days, ctx)["metrics"])
    print(f"基准: 年化 {b[0]:.2f}% 回撤 {b[1]:.2f}% 卡玛 {b[2]:.2f} 笔 {b[3]} 胜率 {b[4]:.1f}%", flush=True)

    grid = {
        "block_gap_below": [-1.5, -2.0, -2.5, -3.0, -4.0],
        "blocked_frac": [0.2, 0.4, 0.6, 0.8, 1.0],
    }
    keys = list(grid)
    rows = []
    combos = list(itertools.product(*[grid[k] for k in keys]))
    print(f"扫描 {len(combos)} 组 ...", flush=True)
    for i, vals in enumerate(combos, 1):
        c = copy.deepcopy(cfg)
        c["execution"].update(dict(zip(keys, vals)))
        try:
            r = run_backtest(c, days, ctx)
        except Exception as e:  # noqa: BLE001
            print(f"  [FAIL] {dict(zip(keys, vals))}: {e}", flush=True)
            continue
        ann, mdd, cal, n, win = brief(r["metrics"])
        rows.append({"block_gap": vals[0], "frac": vals[1], "年化": round(ann, 2),
                     "回撤": round(mdd, 2), "卡玛": round(cal, 2), "笔数": n,
                     "胜率": round(win, 1),
                     "Δ年化": round(ann - b[0], 2), "Δ卡玛": round(cal - b[2], 2)})
        if i % 5 == 0:
            print(f"  {i}/{len(combos)}", flush=True)

    df = pd.DataFrame(rows).sort_values("卡玛", ascending=False)
    df.to_csv(ROOT / "output" / "w225c_gap.csv", index=False, encoding="utf-8-sig")
    print("\n=== 按卡玛排序 top12 ===")
    print(df.head(12).to_string(index=False))
    print("\n=== 按年化排序 top8 ===")
    print(df.sort_values("年化", ascending=False).head(8).to_string(index=False))
    json.dump({"base": b, "rows": rows}, open(ROOT / "output" / "w225c_gap.json", "w"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
