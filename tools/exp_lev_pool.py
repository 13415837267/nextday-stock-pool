# -*- coding: utf-8 -*-
"""第一百七十三轮: 杠杆前沿 + 资金向高期望档集中 + 闲置现金计息。

本轮要回答的三个问题
--------------------
1. **1.25x 是不是欠配?** 第一百七十轮只跑了 1.0 / 1.25 两个点(1500 日)与 2.0x(全史),
   卡玛在两个点上都是单调上升(1.0 → 1.25: 5.87 → 6.47; 全史 1.0 → 2.0: 3.80 → 4.07)。
   凯利 f* = mu/sigma^2 ~ 6x。本组把 1500 日的 1.5 / 2.0 / 2.5 补齐, 看卡玛峰在哪。
2. **资金能不能向高期望档集中?** 第一百六十九轮的仓位审计(353 笔)给出:
   1.0x 档 +1.849%(239 笔) / 0.5x 档 +0.351%(62 笔) / 0.3x 档 +0.825%(52 笔)。
   低档不是"次优", 是"接近零期望"。历史第 119 轮否掉过"低分降仓", 但那**只降不加** ——
   在场次数减少, 绝对收益必崩。本组做的是"低档缩 + 高档用更高 lev 顶上去":
   总敞口不降, 只是把同样一笔风险预算从 +0.35% 的票挪到 +1.85% 的票。
3. **闲置现金值多少钱?** 年均只有 24% 的交易日有持仓。实盘上这部分可以买逆回购
   (年化 1.5~2%), 回测里是 0。本组量化它对复利的贡献(纯增益, 不改任何交易判定)。

用法: python tools/exp_lev_pool.py [days] [group1,group2,...]
输出: output/exp_lev_pool.json
"""
from __future__ import annotations

import copy
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ["BT_FULL_TRADES"] = "1"

from src.backtest import prepare_context, run_backtest       # noqa: E402
from src.config import load_config                          # noqa: E402

OUT = os.path.join(ROOT, "output")

BASE_TIERS = [{"min": 21, "mult": 1.0}, {"min": 11, "mult": 0.5}, {"min": 1, "mult": 0.3}]
TIGHT_TIERS = [{"min": 21, "mult": 1.0}, {"min": 11, "mult": 0.35}, {"min": 1, "mult": 0.20}]

GROUPS = [
    # name, leverage, pool_n_tiers, cash_yield
    ("base",    1.25, BASE_TIERS,  0.0),
    ("L2.0",    2.00, BASE_TIERS,  0.0),
    ("L2.0T",   2.00, TIGHT_TIERS, 0.0),
    ("L2.5T",   2.50, TIGHT_TIERS, 0.0),
    ("L2.0Tcy", 2.00, TIGHT_TIERS, 0.015),
]


def yearly(equity):
    by: dict[str, list] = {}
    for e in equity:
        by.setdefault(str(e["d"])[:4], []).append((str(e["d"]), e["nav"]))
    out, prev = {}, None
    for y in sorted(by):
        pts = by[y]
        start = prev if prev is not None else pts[0][1]
        out[y] = round((pts[-1][1] / start - 1) * 100, 2)
        prev = pts[-1][1]
    return out


def dd_segments(equity, topn=3):
    s = pd.Series({e["d"]: e["nav"] for e in equity})
    dd = s / s.cummax() - 1
    segs, cur = [], []
    prev = None
    for d, v in dd[dd < 0].items():
        if prev is None or (pd.Timestamp(d) - pd.Timestamp(prev)).days <= 15:
            cur.append((d, float(v)))
        else:
            if cur:
                segs.append(cur)
            cur = [(d, float(v))]
        prev = d
    if cur:
        segs.append(cur)
    segs = [x for x in segs if len(x) >= 3]
    segs.sort(key=lambda x: min(v for _, v in x))
    return [{"from": x[0][0], "to": x[-1][0], "min_pct": round(min(v for _, v in x) * 100, 2),
             "days": len(x)} for x in segs[:topn]]


def tier_audit(trades):
    """按实际下注比例分组(第一百六十九轮口径), 看高档加杠杆后各档单笔是否保持。"""
    real = [t for t in trades if t.get("ret_pct") is not None]
    if not real:
        return {}
    df = pd.DataFrame(real)
    df["pf"] = df["pos_frac"].round(2)
    g = df.groupby("pf")["ret_pct"].agg(["count", "mean"])
    return {str(k): {"n": int(v["count"]), "mean_pct": round(float(v["mean"]), 3)}
            for k, v in g.iterrows()}


def build_groups(spec: str):
    """解析 "L=1.5,1.75;CY=2.0" 形式的临时组定义。"""
    groups = []
    for part in spec.split(";"):
        part = part.strip()
        if part.startswith("L="):
            for x in part[2:].split(","):
                groups.append((f"L{x}", float(x), BASE_TIERS, 0.0))
        elif part.startswith("CY="):
            for x in part[3:].split(","):
                groups.append((f"L{x}cy", float(x), BASE_TIERS, 0.015))
    return groups


def main(days: int = 1500, only: list[str] | None = None, spec: str = "") -> None:
    cfg0 = load_config()
    print(f"\n[prepare] days={days}", flush=True)
    t0 = time.time()
    ctx = prepare_context(cfg0, days)
    print(f"[prepare] done in {time.time() - t0:.0f}s\n", flush=True)

    groups = build_groups(spec) if spec else GROUPS
    rows = []
    hdr = "%-8s %5s %12s %10s %9s %7s %7s %6s %8s %10s"
    print(hdr % ("组", "lev", "总收益%", "年化%", "回撤%", "卡玛", "夏普", "笔数", "单笔%", "均杠杆"))
    print("-" * 100, flush=True)

    for name, lev, tiers, cy in groups:
        if only and name not in only:
            continue
        cfg = copy.deepcopy(cfg0)
        cfg["execution"]["leverage"] = lev
        cfg["execution"]["pool_n_tiers"] = tiers
        cfg["execution"]["cash_yield"] = cy
        t1 = time.time()
        r = run_backtest(cfg, days, ctx)
        m = r["metrics"]
        trades = r.get("trades_all") or []
        real = [t for t in trades if t.get("ret_pct") is not None]
        avg_lev = float(np.mean([float(t.get("pos_frac") or 1.0) for t in real])) * lev if real else 0.0
        yr = yearly(r["equity"])
        seg = dd_segments(r["equity"])
        tau = tier_audit(trades)
        rows.append({"name": name, "lev": lev, "cy": cy,
                     "tiers": tiers, "metrics": m, "year": yr,
                     "dd_segments": seg, "tier_audit": tau,
                     "avg_eff_lev": round(avg_lev, 3), "secs": round(time.time() - t1)})
        print("%-8s %5.2f %12.1f %10.2f %9.2f %7.2f %7.2f %6d %8.3f %10.3f"
              % (name, lev, m["总收益率%"], m["年化收益率%"], m["最大回撤%"],
                 m["卡玛比率"] or 0, m["夏普比率"], m["交易次数"],
                 m["平均单笔收益%"], avg_lev), flush=True)
        print("        分档: " + "  ".join(f"{k}x n={v['n']} {v['mean_pct']:+.3f}%"
                                          for k, v in sorted(tau.items(), reverse=True)), flush=True)
        print("        耗时 %.0fs" % (time.time() - t1), flush=True)
        # 增量落盘: 单组 20~30 分钟, 5 组接近 job 上限, 中途超时也要保住已完成的部分。
        os.makedirs(OUT, exist_ok=True)
        with open(os.path.join(OUT, "exp_lev_pool.json"), "w", encoding="utf-8") as f:
            json.dump({"days": days, "rows": rows, "done": True, "partial": len(rows) < len(groups)},
                      f, ensure_ascii=False, indent=1)

    print("\n=== 分年收益 ===")
    years = sorted({y for x in rows for y in x["year"]})
    print("%-6s " % "年" + " ".join(f"{x['name']:>9s}" for x in rows))
    for y in years:
        print("%-6s " % y + " ".join(f"{x['year'].get(y, 0):>9.1f}" for x in rows))
    print("\n=== 回撤段 ===")
    for x in rows:
        print(x["name"], x["dd_segments"])

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "exp_lev_pool.json"), "w", encoding="utf-8") as f:
        json.dump({"days": days, "rows": rows}, f, ensure_ascii=False, indent=1)
    print("\nsaved output/exp_lev_pool.json")


if __name__ == "__main__":
    d = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    a2 = sys.argv[2] if len(sys.argv) > 2 else ""
    if "=" in a2:
        main(d, None, a2)
    else:
        main(d, a2.split(",") if a2 else None, "")
