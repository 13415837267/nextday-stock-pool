# -*- coding: utf-8 -*-
"""参数排名跨年一致性检验（Kendall's W）。

回答一个问题：**在当前参数附近继续精调，携带的是跨期信息还是噪声？**

方法：同一参数取 N 个档位，各跑一次全史回测，取**交易级平均单笔收益**（而非复利总收益，
避免单年路径主导），按卖出年份分组，对每个年份给 N 个档位排名，再算一致性系数 W。

判读：
    W >= 0.7   档位优劣跨年一致 → 参数有真实效应，调参有意义
    W <  0.3   排名在年份间随机漂移 → "全样本最优"是凑巧，精调等于拟合噪声
    显著性看 chi2 = m(n-1)W（df = n-1），如 df=4 时临界值 9.49（α=0.05）

补充判读：把"全样本总收益排序"与"年度平均排名"对照。若两者矛盾，说明全样本最优
来自少数年份的路径放大，而非稳定优势 —— 这正是过拟合的典型形态。

用法：
    python tools/param_rank_stability.py
    python tools/param_rank_stability.py --params selection.bias20_max=10.5,11,11.4,12,12.5
    python tools/param_rank_stability.py --days 1500 --min-trades 5
"""
from __future__ import annotations

import argparse
import copy
import gc
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context, run_backtest  # noqa: E402
from src.config import load_config  # noqa: E402

DEFAULT_PARAMS = [
    "selection.bias20_max=10.5,11.0,11.4,12.0,12.5",
    "selection.dip_vol_ratio_max=3.0,3.5,4.0,4.5,5.0",
]


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


def _coerce(template, raw: str):
    """按原值类型解析档位字符串，避免 11 被写成 11.0 后类型漂移。"""
    if isinstance(template, bool):
        return raw.lower() in ("1", "true", "yes")
    if isinstance(template, int):
        return int(float(raw))
    return float(raw)


def _year_stats(trades: list) -> dict:
    d: dict[str, list[float]] = {}
    for t in trades:
        r = t.get("ret_pct")
        if r is None:
            continue
        d.setdefault(str(t.get("sell_date", ""))[:4], []).append(float(r))
    return {y: (len(v), sum(v) / len(v)) for y, v in d.items()}


def _ranks(pairs: list[tuple[float, float]]) -> dict:
    """pairs: [(level, value)] → {level: rank}，1 = 最好，并列取平均。"""
    s = sorted(pairs, key=lambda x: -x[1])
    out, i = {}, 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and abs(s[j + 1][1] - s[i][1]) < 1e-9:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            out[s[k][0]] = avg
        i = j + 1
    return out


def audit(cfg: dict, ctx: dict, path: str, levels: list, years: list, min_trades: int) -> None:
    print("=" * 112)
    print(f"参数 {path}   档位 {levels}")
    print("=" * 112)

    table = {}
    for v in levels:
        c = copy.deepcopy(cfg)
        _set_path(c, path, v)
        r = run_backtest(c, None, dict(ctx))
        m = r["metrics"]
        table[v] = {"m": m, "ys": _year_stats(r.get("trades_all") or [])}
        del r, c
        gc.collect()
        print(f"  档 {str(v):>6}: 全史收益 {m['总收益率%']:>11.2f}%  "
              f"卡玛 {str(m['卡玛比率']):>6}  笔 {m['交易次数']:>4}  "
              f"单笔 {m['平均单笔收益%']:>6.3f}%", flush=True)

    print(f"\n  {'年份':6s}" + "".join(f"{str(v):>9s}" for v in levels) + "   年度最优")
    matrix, best_cnt = [], {}
    for y in years:
        pairs = []
        for v in levels:
            rec = table[v]["ys"].get(y)
            pairs.append((v, rec[1] if rec and rec[0] >= min_trades else -999.0))
        rk = _ranks(pairs)
        valid = [p for p in pairs if p[1] > -900]
        best = max(valid, key=lambda x: x[1])[0] if valid else None
        if best is not None:
            best_cnt[best] = best_cnt.get(best, 0) + 1
        print(f"  {y:6s}" + "".join(f"{('%.3f' % p[1]) if p[1] > -900 else 'NA':>9s}" for p in pairs)
              + f"   {best}", flush=True)
        matrix.append([rk[v] for v in levels])

    m_, n_ = len(matrix), len(levels)
    R = [sum(row[j] for row in matrix) for j in range(n_)]
    rbar = m_ * (n_ + 1) / 2
    S = sum((r - rbar) ** 2 for r in R)
    W = 12 * S / (m_ * m_ * (n_ ** 3 - n_))
    chi2 = m_ * (n_ - 1) * W

    by_total = max(levels, key=lambda v: table[v]["m"]["总收益率%"])
    by_rank = min(levels, key=lambda v: R[levels.index(v)])
    print(f"\n  Kendall W = {W:.3f}   chi2 = {chi2:.2f} (df={n_ - 1})")
    print("  各档排名和（越小越好）: "
          + "  ".join(f"{v}:{R[j]:.1f}" for j, v in enumerate(levels)))
    print(f"  年度最优档位分布: {best_cnt}")
    print(f"  按全样本总收益最优: {by_total}   按年度平均排名最优: {by_rank}"
          + ("   ← 两者矛盾，全样本最优由少数年份路径放大" if by_total != by_rank else ""))
    verdict = "参数有真实效应" if W >= 0.7 else ("较弱，需更多证据" if W >= 0.3 else
                                                 "排名随机，精调等于拟合噪声")
    print(f"  判定: {verdict}\n", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=3377, help="总窗口交易日数")
    ap.add_argument("--params", action="append", default=None,
                    help="path=v1,v2,... 可多次传入；缺省测两个核心参数")
    ap.add_argument("--min-trades", type=int, default=5, help="某年交易数低于此值记 NA")
    ap.add_argument("--years-min-days", type=int, default=200, help="完整年最少交易日")
    a = ap.parse_args()

    cfg = load_config(str(ROOT / "config.yaml"))
    print(f"[build] 构建上下文（{a.days} 日）...", flush=True)
    ctx = prepare_context(cfg, a.days)
    md = ctx["main_dates"]
    years = [y for y in sorted({d[:4] for d in md})
             if sum(1 for d in md if d[:4] == y) >= a.years_min_days]
    print(f"[build] {md[0]} ~ {md[-1]}  {len(md)} 日  完整年 {len(years)} 个: "
          f"{years[0]}~{years[-1]}\n", flush=True)

    specs = a.params or DEFAULT_PARAMS
    for spec in specs:
        if "=" not in spec:
            print(f"跳过无法解析的规格: {spec}")
            continue
        path, raw = spec.split("=", 1)
        path = path.strip()
        template = _get_path(cfg, path)
        levels = [_coerce(template, x.strip()) for x in raw.split(",") if x.strip()]
        audit(cfg, ctx, path, levels, years, a.min_trades)

    print("=" * 112)
    print("汇总口径：W 越接近 0，越说明档位排名在年份间随机，“全样本最优”越可能只是凑巧。")
    print("用法建议：任何参数精调前先跑本检验；W < 0.3 时不应据单点最优改 config。")
    print("=" * 112)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
