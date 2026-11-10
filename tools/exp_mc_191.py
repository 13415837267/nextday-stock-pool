# -*- coding: utf-8 -*-
"""第 191 轮: 从点估计到分布 —— 日频净值块重采样的压力测试。

为什么不用逐笔收益做蒙特卡洛(本轮踩到的坑, 记录在此避免重犯):
  `trades_all` 的 prod(1+ret_pct) = 2,651x, 而引擎在 1.25x 下的净值是 30,502x。
  按 prod(1+1.25·r) 只能得到 13,347x —— 差 2.2 倍。逐笔序列与净值口径不同源
  (净值含买入日收盘浮盈的杠杆放大, 逐笔是"仓位自身收益率"), 因此**逐笔做 MC 会系统性低估**。
  改用引擎实测的**日频净值**做底, 杠杆在日收益上做精确线性换算:
      r_L = L·g - (L-1)·i,   g = (r_1.25 + 0.25·i)/1.25,  i = 0.055/252(仅持仓日)
  这样持仓/空仓结构、浮盈路径、费用全部保留。

爆仓阈值(第 190 轮担保比例公式的逆解): 担保 = L(1+dd)/(L-1) = 1.3
      L=1.25 -> -74.0% / 1.5 -> -56.7% / 1.7 -> -46.5% / 2.0 -> -35.0% / 2.5 -> -22.0%
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_LEV = 1.25
RATE = 0.055
TD = 252.0


def load_daily() -> np.ndarray:
    d = json.load(open(os.path.join(ROOT, "output", "backtest_long.json"), encoding="utf-8"))
    nav = np.array([x["nav"] for x in d["equity"]], dtype=float)
    return nav[1:] / nav[:-1] - 1.0


def to_lev(r: np.ndarray, lev: float) -> np.ndarray:
    """把 1.25x 的日收益精确换算到任意杠杆(持仓日计息, 空仓日不动)。"""
    i = RATE / TD
    hold = np.abs(r) > 1e-12
    g = np.where(hold, (r + (BASE_LEV - 1.0) * i) / BASE_LEV, 0.0)
    return np.where(hold, lev * g - (lev - 1.0) * i, 0.0)


def path_stats(r: np.ndarray) -> tuple[float, float]:
    nav = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(nav)
    return float(nav[-1]), float((nav / peak - 1.0).min() * 100.0)


def crash_dd(lev: float) -> float:
    return (1.3 * (lev - 1.0) / lev - 1.0) * 100.0


def boot(r: np.ndarray, B: int, block: int, horizon: int, seed: int) -> np.ndarray:
    """环形块重采样, 返回 (B, horizon) 的日收益矩阵。"""
    n = len(r)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(horizon / block))
    st = rng.integers(0, n, size=(B, nb))
    idx = (st[:, :, None] + np.arange(block)[None, None, :]) % n
    return r[idx.reshape(B, -1)[:, :horizon]]


def main() -> None:
    r125 = load_daily()
    n_day = len(r125)
    print("=" * 100)
    print(f"日频样本: {n_day} 个交易日, 持仓日 {int((np.abs(r125) > 1e-12).sum())} 天 "
          f"({(np.abs(r125) > 1e-12).mean() * 100:.0f}%)")
    print(f"原始路径 1.25x: {np.prod(1 + r125):,.0f}x, 回撤 {path_stats(r125)[1]:.2f}%")
    # 换算自检: 1.0x 应还原 ~2,651x
    t1, d1 = path_stats(to_lev(r125, 1.0))
    print(f"换算自检 1.00x: {t1:,.0f}x / 回撤 {d1:.2f}%  "
          f"(逐笔 prod(1+ret) 口径为 2,651x —— 差异即'浮盈杠杆化', 见文件头)")
    print("=" * 100)

    LEVS = (1.0, 1.25, 1.5, 1.7, 2.0, 2.5)

    # ---------- ① 原始路径 vs 重采样分布 ----------
    print("\n【A】原始路径处于什么分位(3000 日全额重采样, block=20, B=3000)")
    print(f"{'杠杆':>5} {'原始终值':>14} {'原始回撤':>9} | {'重排中位':>13} {'5%':>10} "
          f"{'25%':>10} {'75%':>12} {'95%':>14} {'原始分位':>8}")
    rows_a = []
    for lev in LEVS:
        r = to_lev(r125, lev)
        t0, dd0 = path_stats(r)
        mat = boot(r, B=3000, block=20, horizon=n_day, seed=101)
        nav = np.cumprod(1 + mat, axis=1)[:, -1]
        pct = float((nav < t0).mean() * 100)
        rows_a.append(dict(lev=lev, total=t0, mdd=dd0,
                           p5=float(np.percentile(nav, 5)), p25=float(np.percentile(nav, 25)),
                           med=float(np.median(nav)), p75=float(np.percentile(nav, 75)),
                           p95=float(np.percentile(nav, 95)), pct=pct))
        print(f"{lev:>5.2f} {t0:>14,.0f}x {dd0:>8.1f}% | {np.median(nav):>13,.0f}x "
              f"{np.percentile(nav, 5):>9,.0f}x {np.percentile(nav, 25):>9,.0f}x "
              f"{np.percentile(nav, 75):>11,.0f}x {np.percentile(nav, 95):>13,.0f}x {pct:>7.0f}%")

    # ---------- ② 回撤分布与爆仓概率 ----------
    print("\n【B】爆仓概率: 维持担保比例 <130% 即被强平(B=3000, block=20)")
    print(f"{'杠杆':>5} {'爆仓阈值':>9} {'原始回撤':>9} {'重排中位':>9} {'5%分位':>9} "
          f"{'1%分位':>9} {'爆仓概率':>9}  {'担保余量':>8}")
    rows_b = []
    for lev in LEVS:
        r = to_lev(r125, lev)
        thr = crash_dd(lev)
        mat = boot(r, B=3000, block=20, horizon=n_day, seed=202)
        nav = np.cumprod(1 + mat, axis=1)
        peaks = np.maximum.accumulate(nav, axis=1)
        dd = (nav / peaks - 1).min(axis=1) * 100
        p_crash = float((dd < thr).mean() * 100)
        _, dd0 = path_stats(r)
        rows_b.append(dict(lev=lev, thr=thr, hist=dd0, dd_med=float(np.median(dd)),
                           dd_p5=float(np.percentile(dd, 5)),
                           dd_p1=float(np.percentile(dd, 1)), p_crash=p_crash))
        print(f"{lev:>5.2f} {thr:>8.1f}% {dd0:>8.1f}% {np.median(dd):>8.1f}% "
              f"{np.percentile(dd, 5):>8.1f}% {np.percentile(dd, 1):>8.1f}% "
              f"{p_crash:>8.2f}%  {dd0 - thr:>7.1f}pp")

    # ---------- ③ 未来 5 年 ----------
    print("\n【C】未来 5 年(1250 个交易日, B=5000)的终值分布")
    print(f"{'杠杆':>5} {'亏钱':>7} {'5%':>9} {'25%':>9} {'中位':>10} {'75%':>11} {'95%':>12} {'爆仓':>7}")
    rows_c = []
    for lev in (1.25, 1.5, 1.7, 2.0):
        r = to_lev(r125, lev)
        thr = crash_dd(lev)
        mat = boot(r, B=5000, block=20, horizon=1250, seed=303)
        nav = np.cumprod(1 + mat, axis=1)
        peaks = np.maximum.accumulate(nav, axis=1)
        dd = (nav / peaks - 1).min(axis=1) * 100
        fin = nav[:, -1]
        rows_c.append(dict(lev=lev, p_loss=float((fin < 1).mean() * 100),
                           p5=float(np.percentile(fin, 5)), p25=float(np.percentile(fin, 25)),
                           med=float(np.median(fin)), p75=float(np.percentile(fin, 75)),
                           p95=float(np.percentile(fin, 95)),
                           p_crash=float((dd < thr).mean() * 100)))
        print(f"{lev:>5.2f} {(fin < 1).mean() * 100:>6.2f}% {np.percentile(fin, 5):>8.2f}x "
              f"{np.percentile(fin, 25):>8.2f}x {np.median(fin):>9.2f}x "
              f"{np.percentile(fin, 75):>10.2f}x {np.percentile(fin, 95):>11.2f}x "
              f"{rows_c[-1]['p_crash']:>6.2f}%")

    # ---------- ④ 块长度敏感性: 重排会不会人为放大回撤 ----------
    print("\n【D】块长度敏感性(B=2000): 块越短破坏的'跌后反弹'越多, 回撤被高估越多")
    print(f"{'块长':>5} " + " ".join(f"{f'L={lv:.2f}':>18}" for lv in (1.25, 1.5, 1.7)))
    print(f"{'':>5} " + " ".join(f"{'回撤中位/爆仓%':>18}" for _ in range(3)))
    rows_d = []
    for blk in (5, 10, 20, 40, 60, 120):
        cells = []
        for lev in (1.25, 1.5, 1.7):
            r = to_lev(r125, lev)
            mat = boot(r, B=2000, block=blk, horizon=n_day, seed=404)
            nav = np.cumprod(1 + mat, axis=1)
            dd = (nav / np.maximum.accumulate(nav, axis=1) - 1).min(axis=1) * 100
            med = float(np.median(dd))
            pc = float((dd < crash_dd(lev)).mean() * 100)
            cells.append((med, pc))
            rows_d.append(dict(block=blk, lev=lev, dd_med=med, p_crash=pc))
        print(f"{blk:>5} " + " ".join(f"{m:>11.1f}% / {p:>4.1f}%" for m, p in cells))

    out = dict(days=n_day, base_lev=BASE_LEV, a=rows_a, b=rows_b, c=rows_c, d=rows_d)
    p = os.path.join(ROOT, "output", "mc_191.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"\n已写 {p}")


if __name__ == "__main__":
    main()
