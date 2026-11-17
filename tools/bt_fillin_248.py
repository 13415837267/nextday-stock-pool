"""第248轮 G: 合入主策略 —— 一个账户跑两条腿，与拼接口径对照

三组:
  A. fillin.enabled=false          -> 必须复现 86.42% / -24.24% / 3.57（证明改动零副作用）
  B. fillin.enabled=true, let_way=false -> 只有"主策略不做时补位"
  C. fillin.enabled=true, let_way=true  -> 248 定稿（重叠日让位）

ctx 只建一次（prepare_context 是耗时主体），三组共用同一份面板。
"""

from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context, run_backtest  # noqa: E402
from src.config import load_config  # noqa: E402

os.environ["BT_FULL_TRADES"] = "1"


def show(nm: str, r: dict) -> dict:
    m = r["metrics"]
    print(f"\n【{nm}】{r['meta']['start']}~{r['meta']['end']} ({r['meta']['days']} 日)")
    keys = ["年化收益率%", "最大回撤%", "卡玛比率", "夏普比率", "交易次数", "胜率%",
            "平均单笔收益%", "主策略笔数", "补位笔数", "补位·破位", "补位·深低开",
            "补位触发", "让位天数", "补位放弃"]
    for k in keys:
        if k in m:
            print(f"  {k:<14}{m[k]}")
    return m


def main() -> None:
    cfg0 = load_config()
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    c1 = copy.deepcopy(cfg0)
    c1["fillin"]["enabled"] = True
    ctx = prepare_context(c1, n)      # 面板带 dh 列, 关掉 fillin 也只是多几列不被读取

    runs = []
    for nm, en, lw in (("A 主策略单独", False, True),
                       ("B 补位只填空档", True, False),
                       ("C 248 定稿(让位)", True, True)):
        c = copy.deepcopy(cfg0)
        c["fillin"]["enabled"] = en
        c["fillin"]["let_way"] = lw
        r = run_backtest(c, n, ctx)
        m = show(nm, r)
        runs.append((nm, r, m))

    print("\n" + "=" * 78)
    print(f"{'方案':<18}{'年化':>9}{'回撤':>9}{'卡玛':>7}{'笔数':>7}{'补位':>6}{'胜率':>7}")
    for nm, r, m in runs:
        print(f"{nm:<18}{m['年化收益率%']:>9.2f}{m['最大回撤%']:>9.2f}{m['卡玛比率']:>7.2f}"
              f"{m['交易次数']:>7}{m.get('补位笔数', 0):>6}{m['胜率%']:>7.1f}")

    # 补位明细
    r = runs[-1][1]
    tr = pd.DataFrame(r.get("trades_all") or [])
    if len(tr) and "_fillin" in tr.columns:
        _fl = tr["_fillin"].fillna(False).astype(bool)
        f = tr[_fl & tr.ret_pct.notna()]
        print(f"\n【C 的补位成交 {len(f)} 笔】")
        if len(f):
            print(f"  单笔 {f.ret_pct.mean():.2f}%  胜率 {(f.ret_pct>0).mean()*100:.0f}%  "
                  f"仓位均值 {f.pos_frac.mean()*100:.0f}%")
            g = f.assign(yr=f.buy_date.str[:4]).groupby("yr").agg(
                n=("ret_pct", "size"), avg=("ret_pct", "mean"))
            print(g.round(2).to_string())
        # 主策略笔数变化
        mm = tr[(~_fl) & tr.ret_pct.notna()]
        print(f"  主策略成交 {len(mm)} 笔 单笔 {mm.ret_pct.mean():.2f}%")

    # 与拼接口径对比（248 报的 375 笔 / 135.87% / 5.76）
    print("\n【对照】外部拼接口径(scan_248c): 375 笔 / 135.87% / -23.59% / 5.76")
    out = {nm: {"metrics": m, "equity": r["equity"]} for nm, r, m in runs}
    (ROOT / "output" / f"w249_combo_{n}.json").write_text(
        json.dumps(out, ensure_ascii=False, default=float), encoding="utf-8")
    print(f"明细已写入 output/w249_combo_{n}.json")


if __name__ == "__main__":
    main()
