# -*- coding: utf-8 -*-
"""第四十四轮: 十年走前反证实验(事前注册, 零事后挑选)。

问题: 用十年数据(2016-2023训练段)重新选参, 能否得到2024-09后(验证段)更优的参数?
方法: prepare_context一次(2500日面板, 与选股参数无关, 可复用), 对事前注册的12组
参数变体各跑一次run_backtest, 按buy_date<20240901切训练段/验证段分别统计。
排名标准(事前注册): 训练段连乘复利——训练段八年全亏, "训练段最优"=亏最少。
判据(事前注册):
  - 训练段top参数在验证段均值 >= 当前参数(+3.142%/笔)+0.5pp -> 十年数据有增量(需再验)
  - ≈或更低 -> 实证"用十年数据调参无增量/负贡献"
  - 附加观察: 训练段top是否"交易骤减"型(空策略化, 即路B死路)
变体覆盖训练段可能偏好的"更保守"方向与反向, 各一档, 不做网格细化(防多重比较)。

用法: python3 tools/wf_counterfactual.py
输出: 控制台逐组进度 + output/wf_counterfactual.json
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context, run_backtest  # noqa: E402
from src.config import load_config  # noqa: E402

ERA = "20240901"
DAYS = 2500
OUT = ROOT / "output" / "wf_counterfactual.json"

# 事前注册变体(key, 配置块, 参数, 值): 各动一个维度一档, 覆盖紧/松两向
VARIANTS = [
    ("gap_loose",     "execution", "buy_gap_below",         -1.5),
    ("gap_tight",     "execution", "buy_gap_below",         -3.5),
    ("pct_shallow",   "selection", "dip_pct_min",           -4.5),
    ("pct_deep",      "selection", "dip_pct_max",           -4.0),
    ("volratio_tight","selection", "dip_vol_ratio_max",      2.0),
    ("closepos_tight","selection", "dip_close_pos_max",      0.25),
    ("dlow_tight",    "selection", "dip_dist_low20_min",     8.0),
    ("price5",        "universe",  "price_min",              5.0),
    ("turnover10",    "universe",  "turnover_max",          10.0),
    ("skipgain6",     "execution", "skip_open_gain",         6.0),
    ("mom20_tight",   "selection", "dip_mom20_max",         15.0),
    ("bias_tight",    "selection", "bias20_max",             6.0),
]


def seg_stats(trades: list, era_after: bool) -> dict:
    sel = [t for t in trades if t.get("ret_pct") is not None
           and ((t["buy_date"] >= ERA) == era_after)]
    if not sel:
        return {"n": 0, "avg": None, "wr": None, "comp": None}
    rets = [float(t["ret_pct"]) for t in sel]   # float32面板下ret_pct为np.float32
    comp = 1.0
    for r in rets:
        comp *= (1 + r / 100)
    return {"n": len(sel), "avg": round(sum(rets) / len(rets), 3),
            "wr": round(100 * sum(r > 0 for r in rets) / len(rets), 1),
            "comp": round((comp - 1) * 100, 1)}


def main() -> None:
    cfg0 = load_config()
    print(f"== 十年走前反证实验(第四十四轮): {DAYS}日, {len(VARIANTS)}组事前注册变体 ==", flush=True)

    # 基准: 当前参数(直接复用第四十二轮十年回测, 不重跑)
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

    print("[2/2] 逐变体模拟(训练段排名标准=训练段连乘复利)...", flush=True)
    for i, (key, block, param, val) in enumerate(VARIANTS, 1):
        cfg = copy.deepcopy(cfg0)
        old = cfg[block][param]
        cfg[block][param] = val
        t1 = time.time()
        try:
            res = run_backtest(cfg, DAYS, ctx)
            tr = res.get("trades_all") or []
            row = {"key": f"{key}({block}.{param}:{old}->{val})",
                   **{f"tr_{k}": v for k, v in seg_stats(tr, False).items()},
                   **{f"va_{k}": v for k, v in seg_stats(tr, True).items()},
                   "sim_s": round(time.time() - t1)}
        except Exception as e:   # 单组失败不影响其余
            row = {"key": f"{key}({block}.{param}:{old}->{val})",
                   "error": repr(e), "sim_s": round(time.time() - t1)}
        rows.append(row)
        if "error" in row:
            print(f"[{i:2d}/{len(VARIANTS)}] {key} 失败: {row['error']}", flush=True)
        else:
            print(f"[{i:2d}/{len(VARIANTS)}] {key:16s} 训练段 n={row['tr_n']:3d} "
                  f"avg={str(row['tr_avg']):>7} comp={str(row['tr_comp']):>8}%  || "
                  f"验证段 n={row['va_n']:3d} avg={str(row['va_avg']):>7} "
                  f"comp={str(row['va_comp']):>8}%  ({row['sim_s']}s)", flush=True)
        OUT.write_text(json.dumps({"meta": {"era": ERA, "days": DAYS,
                                            "criterion": "训练段连乘复利(事前注册)"},
                                   "rows": rows}, ensure_ascii=False, indent=1),
                       encoding="utf-8")

    # 事前注册排名: 训练段复利降序取top3, 对比其验证段与BASE
    ok = [r for r in rows if "error" not in r and r.get("tr_comp") is not None]
    ok.sort(key=lambda r: r["tr_comp"], reverse=True)
    base = rows[0]
    print("\n== 训练段(2016-2023)排名 top5 -> 验证段(2024-09后)表现 ==", flush=True)
    for r in ok[:5]:
        print(f"  {r['key']:44s} 训练段{r['tr_comp']:>8}% -> 验证段 avg={r['va_avg']} "
              f"comp={r['va_comp']}% (n={r['va_n']})", flush=True)
    print(f"\n  BASE验证段: avg={base['va_avg']}%/笔 comp={base['va_comp']}% (n={base['va_n']})")
    print("  判据: top验证段avg >= BASE+0.5pp -> 有增量(需再验); ≈或更低 -> 无增量/负贡献", flush=True)
    print(f"== 完成, 结果落盘 {OUT} ==", flush=True)


if __name__ == "__main__":
    main()
