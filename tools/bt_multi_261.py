"""第261轮: 单吊 -> 同日多票等分, 到底能不能提高复利

第255轮实测: 507/730 笔主策略 score 与收益相关 **-0.018**, 五档单笔 1.67/1.31/1.20/1.62/1.49
完全非单调。也就是说"只买评分第 1 名"在统计上没有依据 —— 第 1/2/3 名的期望是同一个数。

而现状是 `max_buy_per_day: 1` + `alloc_frac_pct: 100`:
3200 个交易日里只有 730 天有成交(23%), 其余 2470 天资金全空着;
有成交的日子, 池深中位 20~40 只, 只用掉 1 只。

两种改法容易混淆, 这里只测第二种:
  · 放宽条件(降 gap 门槛/降评分门) —— 换的是"更差的日子也进场", 253~259 轮已全部证伪。
  · 同日多票(总敞口不变, 只把当天的池子用起来) —— 换的是"同一天多下几注",
    单笔期望不变, 下注次数上升。在 μ>0 时复利次数就是收益。

  A 现状         max=1, 全仓单吊
  B 两只等分     max=2, 每只 50%
  C 三只等分     max=3, 每只 33%
  D 两只+池深门  max=2, 仅当池深>=21 时才第二只(池浅时第二只质量存疑)
  E 三只+池深门  max=3, 仅当池深>=31 时第三只

口径: 每只的 alloc 由 `alloc_split_by_count` 按实际成交只数均分, 总敞口始终 ≈100%,
所以年化差异只来自"分散/复利次数", 不来自"加杠杆"。
"""

from __future__ import annotations

import copy
import os
import pickle
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context, run_backtest  # noqa: E402
from src.config import load_config  # noqa: E402

os.environ["BT_FULL_TRADES"] = "1"


GROUPS = [
    ("A 现状", dict(max_n=1)),
    ("B 2只等分", dict(max_n=2)),
    ("C 3只等分", dict(max_n=3)),
    ("D 2只池深21", dict(max_n=2, min_pool=21)),
    ("E 3只池深31", dict(max_n=3, min_pool=31)),
]


def main() -> None:
    cfg0 = load_config()
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    only = sys.argv[2].split(",") if len(sys.argv) > 2 else None
    c1 = copy.deepcopy(cfg0)
    c1["fillin"]["enabled"] = True
    ctx = prepare_context(c1, n)

    out = []
    for nm, ov in GROUPS:
        if only and nm.split()[0] not in only:
            continue
        c = copy.deepcopy(cfg0)
        ex = c["execution"]
        ex["max_buy_per_day"] = int(ov["max_n"])
        if ov["max_n"] > 1:
            ex["alloc_split_by_count"] = True
        if ov.get("min_pool"):
            ex["multi_min_pool"] = int(ov["min_pool"])
        r = run_backtest(c, n, ctx)
        m = r["metrics"]
        tr = pd.DataFrame(r.get("trades_all") or [])
        real = tr[tr["ret_pct"].notna()] if len(tr) else tr
        per_day = real.groupby("buy_date").size() if len(real) else pd.Series(dtype=int)
        multi = int((per_day > 1).sum()) if len(per_day) else 0
        print(f"{nm:<12} 年化 {m['年化收益率%']:>7.2f}%  回撤 {m['最大回撤%']:>7.2f}%  "
              f"卡玛 {m['卡玛比率']:>5.2f}  笔数 {m['交易次数']:>4}  多票日 {multi:>4}"
              f"  单笔 {real['ret_pct'].mean() if len(real) else 0:.2f}%", flush=True)
        out.append(dict(name=nm, metrics=m, trades=tr, equity=r.get("equity") or []))

    with open(ROOT / f"output/w261_multi_{n}.pkl", "wb") as f:
        pickle.dump(out, f)


if __name__ == "__main__":
    main()
