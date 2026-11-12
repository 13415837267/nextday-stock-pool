# -*- coding: utf-8 -*-
"""第二百零三轮 · 自进化(B): 副策略的"配方"每季度由数据自己选。

为什么不是"重算权重"
--------------------
本轮先试了最直观的一种自进化 —— 主策略 dip 评分的四个因子按滚动 IC 重算权重
(tools/exp_evolve_203.py)。结果是**负的**: 动态权重年化 -29.2%, 静态 -13.7%,
动态在 16 年里只有 9 年更好。原因很直白: 日频 IC 的样本是"每天 20 来只票",
单笔噪声 5%, 750 日也只有 1066 个可算 IC 的交易日 —— 估出来的权重主要是噪声。

慢变量才估得准: 不猜"因子今天值多少", 而是每季度问一次"过去半年, 哪套配方最赚钱",
下季度照抄。这就是本脚本做的事。

配方池(全部是同一个"右侧"家族, 只变持有期与买价)
------------------------------------------------
  h5 / h10 / h20  ×  限价 0(开盘买) / 1%(挂 -1%) / 2%(挂 -2%, 3 日有效)

规则
----
  · 每季度末看过去 6 个月已实现的月收益, 选第 1 名, 下季度用它;
  · 换配方时清仓一次, 记 0.3% 往返成本(与单笔成本同口径);
  · 只用已发生的月份, 不做任何插值或回填。

对照: 同一批配方里"全期最好的那一个"(事后最优, 不可执行) 与"全期最差的"。
如果季度轮动打不过事后最优, 那是正常的; 关键是它能不能打过"事先不知道选谁"的均值。

用法: python tools/exp_evolve2_203.py 3000
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from src.backtest import prepare_context              # noqa: E402
from src.config import load_config                    # noqa: E402
from src.satellite import entry_ok_series             # noqa: E402
from satellite_backtest import run                    # noqa: E402

OUT = ROOT / "output"

POOL = {
    "h5_pb0":  {"hold_days": 5,  "pullback_pct": 0.0, "pullback_days": 1},
    "h5_pb1":  {"hold_days": 5,  "pullback_pct": 1.0, "pullback_days": 3},
    "h5_pb2":  {"hold_days": 5,  "pullback_pct": 2.0, "pullback_days": 3},
    "h10_pb0": {"hold_days": 10, "pullback_pct": 0.0, "pullback_days": 1},
    "h10_pb1": {"hold_days": 10, "pullback_pct": 1.0, "pullback_days": 3},
    "h10_pb2": {"hold_days": 10, "pullback_pct": 2.0, "pullback_days": 3},
    "h20_pb0": {"hold_days": 20, "pullback_pct": 0.0, "pullback_days": 1},
    "h20_pb1": {"hold_days": 20, "pullback_pct": 1.0, "pullback_days": 3},
}
LOOKBACK = 6      # 选配方看的月数
SWITCH_COST = 0.3  # 换仓一次的往返成本(%)


def _stats(nav: list[float]) -> dict:
    import math
    n = len(nav)
    yrs = n / 244.0
    tot = nav[-1]
    ann = (tot ** (1 / yrs) - 1) * 100 if yrs > 0 else 0.0
    peak, dd = nav[0], 0.0
    for v in nav:
        peak = max(peak, v)
        dd = min(dd, v / peak - 1)
    return {"总倍数": round(tot, 2), "年化%": round(ann, 2), "回撤%": round(dd * 100, 2),
            "卡玛": round(ann / abs(dd * 100), 2) if dd < 0 else None}


def main():
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    cfg0 = load_config()
    t0 = time.time()
    ctx = prepare_context(cfg0, days)
    print(f"装载 {len(ctx['main_dates'])} 日, {time.time()-t0:.0f}s", flush=True)

    # 底 = config.yaml 现行 satellite 全量(含 ivol_pct_max 0.10 / pullback 1.0x3日 /
    # stop -12 / trail 8), 只覆盖配方本身的三个旋钮。少抄一个字段就会把"低波前 10%"
    # 这个池边界弄丢 —— 第一版就是这么跑出 -90% 的, 记在这里防再犯。
    base = {**(cfg0.get("satellite") or {}), "mode": "lowvol", "max_pos": 1,
            "buys_per_day": 1, "sort_by": "mom20_ivol"}
    cache = OUT / f"exp_evolve2_curves_{days}.json"
    if cache.exists():
        curves = json.loads(cache.read_text(encoding="utf-8"))
        print(f"复用已算好的 {len(curves)} 条曲线: {cache.name}", flush=True)
    curves, meta = (curves if cache.exists() else {}), {}
    for name, ov in ({} if cache.exists() else POOL).items():
        cfg = copy.deepcopy(cfg0)
        sat = {**base, **ov}
        sat["mode"] = "lowvol"
        cfg["satellite"] = sat
        r = run(ctx, cfg, entry_ok=entry_ok_series(ctx.get("reg"), cfg))
        m = r["metrics"]
        monthly = {x["month"]: x["ret_pct"] for x in r["monthly"]}
        navs = [x["nav"] for x in r["equity"]]
        curves[name] = {"monthly": monthly, "nav": navs, "metrics": m}
        meta[name] = m
        print(f"{name:8s} 总 {m['总收益率%']:>10.1f}%  年化 {m['年化收益率%']:6.2f}%  "
              f"回撤 {m['最大回撤%']:7.2f}%  笔数 {m['交易次数']:3d}", flush=True)
    if not cache.exists():
        cache.write_text(json.dumps(curves, ensure_ascii=False), encoding="utf-8")

    months = sorted({k for v in curves.values() for k in v["monthly"]})
    # ── 季度轮动: 每 3 个月看过去 6 个月已实现收益, 选第 1 名
    nav, cur, picks, switches = 1.0, "h10_pb1", [], 0
    nav_series = []
    for i, m in enumerate(months):
        hist = months[max(0, i - LOOKBACK):i]
        if (i % 3 == 0 or cur is None) and hist:
            score = {}
            for name in POOL:
                rs = [curves[name]["monthly"].get(h) for h in hist]
                rs = [x for x in rs if x is not None]
                score[name] = (sum(rs) / len(rs)) if rs else -999
            nxt = max(score, key=score.get)
        else:
            nxt = cur
        if nxt != cur:
            if cur is not None:
                nav *= (1 - SWITCH_COST / 100)
                switches += 1
            cur = nxt
            picks.append({"month": m, "pick": cur})
        r = curves[cur]["monthly"].get(m)
        if r is not None:
            nav *= (1 + r / 100)
        nav_series.append(nav)

    roll = _stats(nav_series)
    stat = {k: _stats([x / v["nav"][0] for x in v["nav"]]) for k, v in curves.items()}
    best = max(stat, key=lambda k: stat[k]["总倍数"])
    worst = min(stat, key=lambda k: stat[k]["总倍数"])

    # 每次被选中之后, 该配方在下一季度实际赚了多少(检验"选得对不对")
    verify = []
    for j, p in enumerate(picks):
        m0 = p["month"]
        idx = months.index(m0)
        nxt = months[idx + 1: idx + 4]
        got = [curves[p["pick"]]["monthly"].get(x) for x in nxt]
        got = [x for x in got if x is not None]
        best_next = None
        for name in POOL:
            rr = [curves[name]["monthly"].get(x) for x in nxt]
            rr = [x for x in rr if x is not None]
            if rr and (best_next is None or sum(rr) > best_next[1]):
                best_next = (name, sum(rr))
        verify.append({"month": m0, "pick": p["pick"],
                       "next3_sum": round(sum(got), 2) if got else None,
                       "oracle_best": best_next[0] if best_next else None,
                       "oracle_sum": round(best_next[1], 2) if best_next else None})

    hit = sum(1 for v in verify if v["pick"] == v["oracle_best"]) / max(1, len(verify))
    out = {"days": days, "pool": POOL, "static": stat, "rolling": roll,
           "switches": switches, "best_static": best, "worst_static": worst,
           "oracle_hit": round(hit, 3), "verify": verify,
           "picks": picks,
           "nav_series": [round(x, 4) for x in nav_series]}
    (OUT / f"exp_evolve2_203_{days}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n季度轮动 总 {roll['总倍数']:.2f}x  年化 {roll['年化%']:.2f}%  "
          f"回撤 {roll['回撤%']:.2f}%  卡玛 {roll['卡玛']}  换手 {switches} 次")
    print(f"事后最优 {best} {stat[best]['总倍数']}x / 事后最差 {worst} {stat[worst]['总倍数']}x")
    print(f"选中即最优的比例 {hit:.0%}(随机 1/{len(POOL)}={1/len(POOL):.0%})")
    print("写出", OUT / f"exp_evolve2_203_{days}.json")


if __name__ == "__main__":
    main()
