"""第225轮H · 持仓只数扫描：每天买 1 只 vs 分散到 2~4 只

为什么这是最后一个没动过的维度:
  现行 exec_top_n=1 / max_buy_per_day=1 / alloc_frac_pct=100 —— 每个在場日把全部资金
  压在一只票上, 一年仅约 59 笔。前面所有优化(选股权重/仓位折减/风控)都只影响"选哪一只",
  而"压多重"这一层从未变过。分散到 K 只后单日收益=E[K只平均], 方差降为 1/K 量级:
    - 若年化基本不变而回撤明显下降 -> 卡玛提升, 对 3 万本金是实质性改善
    - 若年化也提升 -> 说明 top1 存在"单票踩雷"的系统性损耗, 那才是真收益
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
    base = load_config()
    ctx = prepare_context(base, days)
    print(f"上下文 {len(ctx['main_dates'])} 日", flush=True)

    rows = []
    for k in (1, 2, 3, 4):
        c = copy.deepcopy(base)
        c["selection"]["exec_top_n"] = k
        c["execution"]["max_buy_per_day"] = k
        c["execution"]["alloc_frac_pct"] = round(100.0 / k, 2)
        m = run_backtest(c, days, ctx)["metrics"]
        r = {"只数": k, "每笔仓位%": round(100.0 / k, 1),
             "年化": m["年化收益率%"], "回撤": m["最大回撤%"],
             "卡玛": m["卡玛比率"], "笔数": m["交易次数"], "胜率": m["胜率%"]}
        rows.append(r)
        print(f"  K={k} 每笔{r['每笔仓位%']:5.1f}%: 年化 {r['年化']:7.2f}% 回撤 {r['回撤']:7.2f}% "
              f"卡玛 {r['卡玛']:5.2f} 笔 {r['笔数']:4d} 胜率 {r['胜率']:4.1f}%", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "output" / "w225h_k.csv", index=False, encoding="utf-8-sig")
    print("\n" + df.to_string(index=False))


if __name__ == "__main__":
    main()
