# -*- coding: utf-8 -*-
"""Walk-Forward 样本外检验: 训练段选参 → 冻结 → 测试段评估。

回答的问题: "如果站在当年, 只用当时可得的历史选参数, 下一年的真实表现会怎样?"
与 tools/period_split.py 的区别 —— 后者参数固定(全样本选出来的), 只能看时段依赖;
本工具每次重选参数, 且**选参时不看测试段**, 是真正的时间序列样本外。

方法(预注册, 不因结果调整):
  - 折: 测试年 2017/2019/2021/2023/2025, 训练段=该年之前 900 个交易日(滚动窗口)
  - 网格: 5 个核心参数 × 3 档, 见 GRID —— **预先写死, 看结果后不得增删**
  - 选参准则: 训练段卡玛比率(cagr/|mdd|), 兼顾收益与回撤
  - 单轴扫描(每次只变一个参数), 选出的 5 个值组合后冻结, 跑测试年
  - 对照: 同一测试年上"现行 config"的表现

局限(必须与结果一起读):
  ① 策略有百余个参数, 这里只动 5 个, 其余仍带着全样本选择的印记;
  ② 单轴扫描忽略参数交互, 训练段最优未必是联合最优;
  ③ 5 折样本太少, 单折差异多数落在噪声内 —— 看的是"重选是否系统性优于/劣于现行",
     而不是某一折的胜负。

用法:
    python tools/walk_forward.py
    python tools/walk_forward.py --train-days 750 --folds 2017,2019,2021,2023,2025
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context, run_backtest  # noqa: E402
from src.config import load_config  # noqa: E402

# 预注册网格: 围绕现行值 ±1 档。改动此表须在 CHANGELOG 记录并重跑全部折。
GRID = [
    ("selection.bias20_max", [10.0, 11.0, 12.0]),
    ("selection.dip_vol_ratio_max", [3.5, 4.0, 4.5]),
    ("regime.market_ma", [18, 20, 22]),
    ("regime.vol_max", [2.0, 2.2, 2.4]),
    ("universe.turnover_min", [1.7, 2.0, 2.3]),
]


def setp(cfg: dict, path: str, val) -> None:
    ks = path.split(".")
    cur = cfg
    for k in ks[:-1]:
        cur = cur.setdefault(k, {})
    cur[ks[-1]] = val


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=3377, help="总窗口交易日数")
    ap.add_argument("--train-days", type=int, default=900, help="训练段长度")
    ap.add_argument("--folds", default="2017,2019,2021,2023,2025", help="测试年份, 逗号分隔")
    ap.add_argument("--min-train", type=int, default=400, help="训练段最短长度")
    a = ap.parse_args()

    cfg = load_config(str(ROOT / "config.yaml"))
    print(f"[build] 构建上下文({a.days} 日)...", flush=True)
    ctx = prepare_context(cfg, a.days)
    md = ctx["main_dates"]
    print(f"[build] {md[0]} ~ {md[-1]}  {len(md)} 日\n", flush=True)

    def run(c, sub):
        c2 = dict(ctx)
        c2["main_dates"] = sub
        r = run_backtest(c, None, c2)
        m = r["metrics"]
        del r
        gc.collect()
        return m

    results = []
    for fy in [x.strip() for x in a.folds.split(",") if x.strip()]:
        test = [d for d in md if d[:4] == fy]
        if not test:
            print(f"[{fy}] 无交易日, 跳过", flush=True)
            continue
        idx = md.index(test[0])
        train = md[max(0, idx - a.train_days): idx]
        if len(train) < a.min_train:
            print(f"[{fy}] 训练段过短 {len(train)}, 跳过", flush=True)
            continue
        print("=" * 100, flush=True)
        print(f"[折 {fy}] 训练 {train[0]}~{train[-1]}({len(train)}日) → "
              f"测试 {test[0]}~{test[-1]}({len(test)}日)", flush=True)
        base = run(cfg, train)
        print(f"  训练基线: 卡玛={base['卡玛比率']} 收益={base['总收益率%']}% 笔={base['交易次数']}",
              flush=True)

        chosen = {}
        for path, vals in GRID:
            best_v, best_s = None, -1e9
            line = []
            for v in vals:
                c = copy.deepcopy(cfg)
                setp(c, path, v)
                m = run(c, train)
                s = m.get("卡玛比率") if m.get("卡玛比率") is not None else -99
                line.append(f"{v}:{m['总收益率%']}%/{m['卡玛比率']}")
                if s > best_s:
                    best_s, best_v = s, v
                del c
                gc.collect()
            chosen[path] = best_v
            print(f"  {path:34s} " + "  ".join(line) + f"   → 选 {best_v} (卡玛 {best_s})",
                  flush=True)

        c_new = copy.deepcopy(cfg)
        for p, v in chosen.items():
            setp(c_new, p, v)
        m_new = run(c_new, test)
        m_cur = run(cfg, test)
        print(f"  ▶ 测试段 重选: 收益={m_new['总收益率%']}% 回撤={m_new['最大回撤%']}% "
              f"卡玛={m_new['卡玛比率']} 笔={m_new['交易次数']} "
              f"单笔={m_new['平均单笔收益%']}%", flush=True)
        print(f"  ▶ 测试段 现行: 收益={m_cur['总收益率%']}% 回撤={m_cur['最大回撤%']}% "
              f"卡玛={m_cur['卡玛比率']} 笔={m_cur['交易次数']} "
              f"单笔={m_cur['平均单笔收益%']}%", flush=True)
        results.append(dict(year=fy, chosen=chosen, new=m_new, cur=m_cur))
        del c_new
        gc.collect()

    if not results:
        print("无有效折")
        return 1

    print("\n" + "=" * 100)
    print("【汇总】重选参数 vs 现行参数, 在各自测试年上的表现")
    print("=" * 100)
    print(f"{'年':6s} {'重选收益':>10s} {'现行收益':>10s} {'差':>9s} "
          f"{'重选卡玛':>9s} {'现行卡玛':>9s}")
    for r in results:
        d = r["new"]["总收益率%"] - r["cur"]["总收益率%"]
        print(f"{r['year']:6s} {r['new']['总收益率%']:>10.2f} {r['cur']['总收益率%']:>10.2f} "
              f"{d:>+9.2f} {str(r['new']['卡玛比率']):>9s} {str(r['cur']['卡玛比率']):>9s}")
    diffs = [r["new"]["总收益率%"] - r["cur"]["总收益率%"] for r in results]
    pn = sum(1 for r in results if r["new"]["总收益率%"] > 0)
    pc = sum(1 for r in results if r["cur"]["总收益率%"] > 0)
    print(f"\n正收益年数: 重选 {pn}/{len(results)}  现行 {pc}/{len(results)}")
    print(f"逐折差(pp): " + "  ".join(f"{r['year']}:{d:+.2f}" for r, d in zip(results, diffs)))
    print(f"合计 {sum(diffs):+.2f}pp  中位 {sorted(diffs)[len(diffs) // 2]:+.2f}pp")
    print("\n各参数被选中的值(跨折分布):")
    for path, _ in GRID:
        vals = [r["chosen"][path] for r in results]
        uni = sorted(set(vals), key=lambda x: vals.count(x), reverse=True)
        print(f"  {path:34s} " + " / ".join(f"{v}×{vals.count(v)}" for v in uni))
    print("\n判读: 若同一参数在各折选出的值频繁漂移, 说明它在噪声内不可辨识 —— "
          "此时'训练段最优'没有外推意义, 现行值应取各折最优值的中位数而非众数。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
