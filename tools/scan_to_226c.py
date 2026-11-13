"""第226轮C · turnover_max 细扫（1500 日）

粗筛: 12->20 在 1500 日 +9.07pp, 750 日 +16.82pp, 两窗口同向, 是本轮唯一同向的改善。
风险: 换手上限放宽 = 纳入更活跃的票, 回测滑点固定千1 不反映真实冲击成本,
      所以必须同时看"笔数"与"回撤"是否恶化, 不能只看年化。
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context, run_backtest  # noqa: E402
from src.config import load_config  # noqa: E402


def main() -> None:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    cfg = load_config()
    ctx = prepare_context(cfg, days)
    print(f"窗口 {len(ctx['main_dates'])} 日", flush=True)

    rows = []
    for v in [12.0, 14.0, 16.0, 18.0, 20.0, 25.0, 30.0, 60.0]:
        c = copy.deepcopy(cfg)
        c["universe"]["turnover_max"] = v
        m = run_backtest(c, days, ctx)["metrics"]
        r = {"to_max": v, "年化": float(m["年化收益率%"]), "回撤": float(m["最大回撤%"]),
             "卡玛": float(m["卡玛比率"]), "笔数": int(m["交易次数"]),
             "胜率": float(m["胜率%"])}
        rows.append(r)
        tag = " ← 现行" if v == 12.0 else ""
        print(f"  to_max={v:<5} 年化 {r['年化']:7.2f}% 回撤 {r['回撤']:7.2f}% "
              f"卡玛 {r['卡玛']:5.2f} 笔 {r['笔数']:4d} 胜率 {r['胜率']:4.1f}%{tag}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "output" / "s226c_to.csv", index=False, encoding="utf-8-sig")
    print("\n" + df.to_string(index=False))


if __name__ == "__main__":
    main()
