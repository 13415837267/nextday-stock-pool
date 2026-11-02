# -*- coding: utf-8 -*-
"""随机参数基线分布 —— 量化"调参"到底贡献了多少。

方法：在参数空间内均匀随机采样 N 组（默认围绕当前值的合理区间），各自跑回测，
把当前 config 放进同一分布看分位。

判读：
    随机中位数已接近当前值 → 收益主要来自**结构**，调参只是边际
    当前值显著高于随机上界 → 当前参数是被多轮选择压力推到尾部的点，外推必须打折
    随机组全部为正 → 策略骨架本身有效，不是"不调参就亏"

注意：本检验不区分"好参数"与"坏参数"，只给出**不做任何优化时的典型表现**。
它与 tools/param_rank_stability.py 互补：前者给水平，后者给可辨识性。

用法：
    python tools/param_random_baseline.py
    python tools/param_random_baseline.py --n 60 --days 1500 --seed 20261102
    python tools/param_random_baseline.py --space "selection.bias20_max=9.5:13.5"  # 只随机这一个
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context, run_backtest  # noqa: E402
from src.config import load_config  # noqa: E402

DEFAULT_SPACE = {
    "selection.bias20_max": (9.5, 13.5),
    "selection.dip_vol_ratio_max": (3.0, 5.0),
    "selection.dip_pct_min": (-7.5, -4.5),
    "regime.market_ma": (16, 26),
    "regime.vol_max": (1.9, 2.6),
    "universe.turnover_min": (1.2, 2.8),
    "universe.turnover_max": (10.0, 16.0),
    "execution.skip_open_gain": (4.0, 12.0),
}


def _set_path(obj: dict, path: str, value) -> None:
    keys = path.split(".")
    cur = obj
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value


def _get_path(obj: dict, path: str):
    cur = obj
    for k in path.split("."):
        cur = cur[k]
    return cur


def _parse_space(items: list[str] | None) -> dict:
    if not items:
        return dict(DEFAULT_SPACE)
    out = {}
    for it in items:
        path, rng = it.split("=", 1)
        lo, hi = rng.split(":")
        out[path.strip()] = (float(lo), float(hi))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="采样组数")
    ap.add_argument("--days", type=int, default=1500, help="回测窗口交易日数")
    ap.add_argument("--seed", type=int, default=20261102)
    ap.add_argument("--space", action="append", default=None,
                    help='path=lo:hi 可多次传入；缺省用内置 8 参数空间')
    ap.add_argument("--json-out", default="", help="明细写入路径（可选）")
    a = ap.parse_args()

    cfg = load_config(str(ROOT / "config.yaml"))
    space = _parse_space(a.space)
    random.seed(a.seed)

    samples = []
    for _ in range(a.n):
        s = {}
        for path, (lo, hi) in space.items():
            v = lo + (hi - lo) * random.random()
            s[path] = int(round(v)) if isinstance(_get_path(cfg, path), int) else round(v, 2)
        samples.append(s)

    print(f"[build] 构建上下文（{a.days} 日）...", flush=True)
    ctx = prepare_context(cfg, a.days)
    print(f"[build] 采样 {a.n} 组，seed={a.seed}，参数空间 {len(space)} 维\n", flush=True)
    print(f"{'#':>3} {'收益%':>10} {'回撤%':>8} {'卡玛':>6} {'笔数':>5}  首个参数", flush=True)

    rows = []
    for i, s in enumerate(samples, 1):
        c = copy.deepcopy(cfg)
        for p, v in s.items():
            _set_path(c, p, v)
        r = run_backtest(c, None, dict(ctx))
        m = r["metrics"]
        del r, c
        gc.collect()
        rows.append({"i": i, "ret": m["总收益率%"], "mdd": m["最大回撤%"],
                     "calmar": m.get("卡玛比率"), "trades": m["交易次数"],
                     "avg": m["平均单笔收益%"], "params": s})
        head = list(s.items())[0]
        print(f"{i:>3} {m['总收益率%']:>10.2f} {m['最大回撤%']:>8.2f} "
              f"{str(m.get('卡玛比率')):>6} {m['交易次数']:>5d}  {head[0].split('.')[-1]}={head[1]}",
              flush=True)

    cur = run_backtest(copy.deepcopy(cfg), None, dict(ctx))["metrics"]
    rets = sorted(r["ret"] for r in rows)
    cals = sorted(r["calmar"] for r in rows if r["calmar"] is not None)

    def pct_of(v, arr):
        return sum(1 for x in arr if x <= v) / len(arr) * 100

    mid = statistics.median(rets)
    print(f"\n当前 config: 收益 {cur['总收益率%']:.2f}%  回撤 {cur['最大回撤%']:.2f}%  "
          f"卡玛 {cur.get('卡玛比率')}  笔数 {cur['交易次数']}", flush=True)

    print("\n" + "=" * 96)
    print(f"随机参数分布（n={len(rets)}）")
    print("=" * 96)
    print(f"  收益  min {rets[0]:>10.2f}%  P10 {rets[int(len(rets)*0.1)]:>10.2f}%  "
          f"中位 {mid:>10.2f}%  P90 {rets[int(len(rets)*0.9)]:>10.2f}%  max {rets[-1]:>10.2f}%")
    print(f"  卡玛  min {cals[0]:>10.2f}   P10 {cals[int(len(cals)*0.1)]:>10.2f}   "
          f"中位 {statistics.median(cals):>10.2f}   P90 {cals[int(len(cals)*0.9)]:>10.2f}   "
          f"max {cals[-1]:>10.2f}")
    print(f"\n  当前值分位：收益 {pct_of(cur['总收益率%'], rets):.1f}%  "
          f"卡玛 {pct_of(cur.get('卡玛比率'), cals):.1f}%")
    print(f"  当前收益 / 随机中位 = {cur['总收益率%'] / mid:.2f}x")
    print(f"  随机组中优于当前的比例："
          f"{sum(1 for x in rets if x >= cur['总收益率%']) / len(rets) * 100:.1f}%")
    print(f"  随机组正收益：{sum(1 for x in rets if x > 0)}/{len(rets)}")
    print("\n判读：随机中位代表\"完全不调参\"的典型水平；当前值与其之比，即调参的名义贡献。")
    print("      若该比值 > 2 且 rank_stability 的 W < 0.3，则名义贡献大部分不可外推。")

    if a.json_out:
        Path(a.json_out).write_text(
            json.dumps({"rows": rows, "current": cur}, ensure_ascii=False, indent=1),
            encoding="utf-8")
        print(f"\n明细已写入 {a.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
