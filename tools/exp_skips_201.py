"""第 201 轮：把"没买的那 979 笔"逐笔算一遍反事实。

引擎每天都会留下 skip 记录（`trades_all` 里 ret_pct 为空、note 写明原因）。
此前所有轮次都只在"已成交的 355 笔"上做文章，从没问过：
**被过滤掉的那些信号，如果照买，会赚还是会亏？**

做法：对每条带代码的 skip，取 T+1 开盘买入、T+2 收盘卖出，用复权因子算收益，
再扣掉与回测同口径的成本（佣金双边 + 印花税卖出 + 滑点双边），按 note 归类。
"""
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

os.environ["BT_FULL_TRADES"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context, run_backtest  # noqa: E402

DAYS = sorted(p.stem.split(".")[0] for p in (ROOT / "data" / "panel").glob("*.csv.gz"))
POS = {d: i for i, d in enumerate(DAYS)}


def cat(note: str) -> str:
    n = re.sub(r"-?\d+(\.\d+)?", "#", note or "")
    n = n.split("(")[0].strip()
    return n or "无说明"


def load(d):
    return pd.read_csv(ROOT / "data" / "panel" / f"{d}.csv.gz")


def cf_ret(code, sd, cost=0.0030):
    i = POS.get(str(sd))
    if i is None or i + 2 >= len(DAYS):
        return None
    try:
        a = load(DAYS[i + 1])
        b = load(DAYS[i + 2])
    except Exception:
        return None
    ra = a[a.ts_code == code]
    rb = b[b.ts_code == code]
    if not len(ra) or not len(rb):
        return None
    ra, rb = ra.iloc[0], rb.iloc[0]
    if int(ra.is_paused) == 1 or int(rb.is_paused) == 1:
        return None
    o = float(ra.open) * float(ra.factor)
    c = float(rb.close) * float(rb.factor)
    if o <= 0 or c <= 0:
        return None
    return (c / o - 1 - cost) * 100


def main(days):
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    ctx = prepare_context(cfg, days)
    r = run_backtest(cfg, days=days, ctx=ctx)
    allt = r["trades_all"]
    adm = [t for t in allt if t.get("ret_pct") is not None]
    skp = [t for t in allt if t.get("ret_pct") is None and t.get("code")]
    print(f"{days} 日: 成交 {len(adm)} 笔, 带代码 skip {len(skp)} 笔")

    rows = []
    for t in skp:
        v = cf_ret(t["code"], t.get("signal_date") or t.get("buy_date"))
        rows.append({"note": t.get("note") or "", "cat": cat(t.get("note") or ""),
                     "cf": v, "sd": t.get("signal_date"), "code": t["code"]})
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "output" / f"skips_201_{days}.csv", index=False)
    ok = df[df.cf.notna()]
    print(f"  可算反事实 {len(ok)} 笔, 覆盖 {ok.cat.nunique()} 类")

    g = ok.groupby("cat").cf.agg(
        笔数="size", 均值="mean", 中位="median",
        胜率=lambda s: (s > 0).mean() * 100,
        几何=lambda s: (np.prod(1 + s / 100) ** (1 / len(s)) - 1) * 100,
        倍数=lambda s: np.prod(1 + s / 100),
    ).sort_values("笔数", ascending=False)
    print(g.round(3).to_string())
    print(f"\n  全部 skip 合计: {len(ok)} 笔, 几何 "
          f"{(np.prod(1 + ok.cf / 100) ** (1 / len(ok)) - 1) * 100:+.3f}%/笔, "
          f"总倍数 {np.prod(1 + ok.cf / 100):.1f}x")
    print(f"  已成交对照: {len(adm)} 笔, 均值 {np.mean([t['ret_pct'] for t in adm]):.3f}%, "
          f"总倍数 {np.prod([1 + t['ret_pct'] / 100 for t in adm]):.1f}x")
    g.to_csv(ROOT / "output" / f"skips_201_{days}_bycat.csv")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1500)
