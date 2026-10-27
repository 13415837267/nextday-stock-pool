# -*- coding: utf-8 -*-
"""IVOL因子权重走前训练(第六十九轮, 文献驱动接入流程, 事前注册)。

纪律(沿第四十四/六十二轮 wf_counterfactual 框架):
  - 训练段: buy_date < 20240901 (2016~2023), 排名标准=训练段连乘复利(事前注册);
  - 验证段: buy_date >= 20240901;
  - 接入判据: 训练段最优权重的验证段 avg >= BASE+0.5pp/笔 且双段同向 -> 讨论接入;
  - 变体: dip_weights.ivol ∈ {0.05, 0.10, 0.15, 0.20, 0.30}(BASE=0 未配置);
  - 其余权重不动(与既有因子线性叠加, _zs 横截面标准化)。

用法: python tools/ivol_wf.py
输出: 打印 + output/ivol_wf.json
"""
from __future__ import annotations

import copy
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("BT_FULL_TRADES", "1")   # 变体统计需逐笔成交(第六十四轮修复口径)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context, run_backtest  # noqa: E402
from src.config import load_config                      # noqa: E402

ERA = "20240901"
DAYS = 2500
OUT = ROOT / "output" / "ivol_wf.json"

VARIANTS = [("ivol_w005", 0.05), ("ivol_w010", 0.10), ("ivol_w015", 0.15),
            ("ivol_w020", 0.20), ("ivol_w030", 0.30)]


def seg_stats(trades: list, era_after: bool) -> dict:
    sel = [t for t in trades if t.get("ret_pct") is not None
           and ((t["buy_date"] >= ERA) == era_after)]
    if not sel:
        return {"n": 0, "avg": None, "wr": None, "comp": None}
    rets = [float(t["ret_pct"]) for t in sel]
    comp = 1.0
    for r in rets:
        comp *= (1 + r / 100)
    return {"n": len(sel), "avg": round(sum(rets) / len(rets), 3),
            "wr": round(100 * sum(r > 0 for r in rets) / len(rets), 1),
            "comp": round((comp - 1) * 100, 1)}


def main() -> None:
    cfg0 = load_config()
    bt = json.loads((ROOT / "output" / "backtest_long.json").read_text())
    base_tr = bt["trades_all"]
    rows = [{"key": "BASE(当前参数)",
             **{f"tr_{k}": v for k, v in seg_stats(base_tr, False).items()},
             **{f"va_{k}": v for k, v in seg_stats(base_tr, True).items()},
             "sim_s": 0}]
    r = rows[0]
    print(f"[BASE] 训练段 n={r['tr_n']} avg={r['tr_avg']} comp={r['tr_comp']}%  || "
          f"验证段 n={r['va_n']} avg={r['va_avg']} comp={r['va_comp']}%", flush=True)

    t0 = time.time()
    print("[1/2] 构建十年面板上下文(参数无关, 复用)...", flush=True)
    ctx = prepare_context(cfg0, DAYS)
    print(f"  PREPARE完成 {time.time()-t0:.0f}s", flush=True)

    print("[2/2] 逐权重模拟(训练段排名标准=训练段连乘复利)...", flush=True)
    for i, (key, w) in enumerate(VARIANTS, 1):
        cfg = copy.deepcopy(cfg0)
        cfg["selection"]["dip_weights"]["ivol"] = w
        t1 = time.time()
        try:
            res = run_backtest(cfg, DAYS, ctx)
            tr = res.get("trades_all") or []
            row = {"key": f"{key}(dip_weights.ivol=0->{w})",
                   **{f"tr_{k}": v for k, v in seg_stats(tr, False).items()},
                   **{f"va_{k}": v for k, v in seg_stats(tr, True).items()},
                   "sim_s": round(time.time() - t1)}
        except Exception as e:   # 单组失败不影响其余
            row = {"key": f"{key}(dip_weights.ivol=0->{w})",
                   "error": repr(e), "sim_s": round(time.time() - t1)}
        rows.append(row)
        if "error" in row:
            print(f"[{i:2d}/{len(VARIANTS)}] {key} 失败: {row['error']}", flush=True)
        else:
            print(f"[{i:2d}/{len(VARIANTS)}] {key:10s} 训练段 n={row['tr_n']:3d} "
                  f"avg={str(row['tr_avg']):>7} comp={str(row['tr_comp']):>8}%  || "
                  f"验证段 n={row['va_n']:3d} avg={str(row['va_avg']):>7} "
                  f"comp={str(row['va_comp']):>8}%  ({row['sim_s']}s)", flush=True)
        OUT.write_text(json.dumps({"meta": {"era": ERA, "days": DAYS,
                                            "criterion": "训练段连乘复利(事前注册)",
                                            "note": "IVOL因子权重走前训练(第六十九轮)"},
                                   "rows": rows}, ensure_ascii=False, indent=1),
                       encoding="utf-8")

    ok = [r for r in rows if "error" not in r and r.get("tr_comp") is not None]
    ok.sort(key=lambda r: r["tr_comp"], reverse=True)
    base = rows[0]
    print("\n== 训练段(2016-2023)排名 -> 验证段(2024-09后)表现 ==", flush=True)
    for r in ok:
        print(f"  {r['key']:40s} 训练段{r['tr_comp']:>8}% -> 验证段 avg={r['va_avg']} "
              f"comp={r['va_comp']}% (n={r['va_n']})", flush=True)
    print(f"\n  BASE验证段: avg={base['va_avg']}%/笔 comp={base['va_comp']}% (n={base['va_n']})")
    print("  接入判据: 训练段最优权重的验证段 avg >= BASE+0.5pp 且双段同向 -> 讨论接入", flush=True)
    print(f"== 完成, 结果落盘 {OUT} ==", flush=True)


if __name__ == "__main__":
    main()
