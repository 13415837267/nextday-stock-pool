# -*- coding: utf-8 -*-
"""第197轮: 主策略 × 副策略的组合权重 —— 静态网格 vs 动态(等风险/动量)。

背景
----
196 轮只测了静态四档, 发现 60/40 的卡玛 7.55 高于 100% 主策略的 6.26,
并解释为"分散降的是主策略仓位, 不是填满空仓"。但没有回答:
**权重该不该动?** 若两条腿的波动率/相关性随时间漂, 固定 60/40 就不是最优。

口径
----
· 两条腿各自满仓运行, 资金按 w 分配; 月度首个交易日调回目标权重, 期间自由漂移
  (日频再平衡会在两条高波动曲线间白交摩擦, 不做)。
· 主腿日收益直接取 output/backtest.json 的 equity(已含 1.0x + 现金 1.5% 计息)。
· 副腿取 output/satellite_bt.json 的 equity(**不含**计息), 按持仓状态补 1.5%/252:
  副策略 889 天空仓, 不补等于系统性低估它 0.9pp/年。
· 动态规则 (只用 t 时点之前的信息):
    A 等风险: w = 1/σ_m / (1/σ_m + 1/σ_s), σ = 过去 60 日日收益标准差
    B 动量: 过去 60 日累计收益高者 70%, 低者 30%
    C 固定 60/40
  权重夹在 [0.2, 0.8], 避免单腿归零后组合变成另一条腿。

输出: output/exp_combo_197_{1500,3000}.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
CASH = 0.015 / 252.0
LB = 60          # 动态权重回看窗口(交易日)
W_LO, W_HI = 0.2, 0.8


def _series(path: Path) -> tuple[list[str], np.ndarray]:
    d = json.loads(path.read_text(encoding="utf-8"))
    eq = d["equity"]
    dates = [str(x["d"]).replace("-", "")[:8] for x in eq]
    nav = np.array([float(x["nav"]) for x in eq])
    return dates, nav


def _ret(nav: np.ndarray) -> np.ndarray:
    r = np.zeros(len(nav))
    r[1:] = nav[1:] / nav[:-1] - 1.0
    return r


def _hold_mask(dates: list[str], trades: list[dict], kb: str, ks: str) -> np.ndarray:
    idx = {d: i for i, d in enumerate(dates)}
    m = np.zeros(len(dates), dtype=bool)
    for t in trades:
        if t.get("ret_pct") is None:
            continue
        b = str(t.get(kb) or "").replace("-", "")[:8]
        s = str(t.get(ks) or "").replace("-", "")[:8]
        if b in idx and s in idx:
            m[idx[b]: idx[s] + 1] = True
    return m


def _metrics(r: np.ndarray) -> dict:
    nav = np.cumprod(1.0 + r)
    n = len(r)
    ann = nav[-1] ** (252.0 / n) - 1
    peak = np.maximum.accumulate(nav)
    dd = float((nav / peak - 1).min())
    calmar = ann / abs(dd) if dd < 0 else float("nan")
    sharpe = r.mean() / (r.std(ddof=1) + 1e-12) * np.sqrt(252)
    return {"总收益率%": round((nav[-1] - 1) * 100, 2), "年化收益率%": round(ann * 100, 2),
            "最大回撤%": round(dd * 100, 2), "卡玛比率": round(calmar, 2),
            "夏普比率": round(sharpe, 2)}


def _weights(rm: np.ndarray, rs: np.ndarray, mode: str, dates: list[str]) -> np.ndarray:
    """返回每日主腿目标权重(只在月度首个交易日重算, 其余日沿用)。"""
    n = len(rm)
    w = np.full(n, 0.6)
    last_ym = None
    for i in range(n):
        ym = dates[i][:6]
        if i < LB or ym == last_ym:
            continue
        last_ym = ym
        # 每月第一个交易日才重算: 用 i 之前的 60 日
        if mode == "C60":
            target = 0.6
        else:
            sm = rm[i - LB:i].std(ddof=1)
            ss = rs[i - LB:i].std(ddof=1)
            if mode == "A":       # 等风险
                if sm <= 0 or ss <= 0:
                    target = 0.6
                else:
                    target = (1.0 / sm) / (1.0 / sm + 1.0 / ss)
            elif mode == "B":     # 动量
                cm = float(np.prod(1 + rm[i - LB:i]) - 1)
                cs = float(np.prod(1 + rs[i - LB:i]) - 1)
                target = 0.7 if cm >= cs else 0.3
            else:
                target = 0.6
        w[i] = float(np.clip(target, W_LO, W_HI))
    return w


def _sim(rm: np.ndarray, rs: np.ndarray, w: np.ndarray, dates: list[str]) -> np.ndarray:
    """月度调仓 + 期间漂移: 每月首个交易日按 w 配资, 月内两腿各自漂移。

    第 197 轮修正: 初版用"w 是否变化"判断调仓, 而 w 是常量数组, 于是只在第一天配资、
    之后一路漂移(buy & hold)。6 年漂移下来主腿权重从 60% 涨到 90%+, 结果介于
    "全主策略"和"月频 60/40"之间, 白算了一版。改成按日期月份跳变。
    """
    n = len(rm)
    out = np.zeros(n)
    cap_m = cap_s = 1.0
    last_ym = None
    for i in range(n):
        ym = dates[i][:6]
        if ym != last_ym:
            tot = cap_m + cap_s
            cap_m, cap_s = tot * w[i], tot * (1 - w[i])
            last_ym = ym
        cap_m *= (1 + rm[i])
        cap_s *= (1 + rs[i])
        out[i] = cap_m + cap_s
    return out[1:] / out[:-1] - 1.0


def _block_boot(diff: np.ndarray, n_boot: int = 2000, block: int = 20, seed: int = 197) -> dict:
    d = diff[np.isfinite(diff)]
    if len(d) < 100:
        return {}
    rng = np.random.default_rng(seed)
    n = len(d)
    nb = int(np.ceil(n / block))
    means = []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block, nb)
        s = np.concatenate([d[i:i + block] for i in starts])[:n]
        means.append(s.mean())
    means = np.array(means)
    return {"年化差_pct": round(float(d.mean()) * 252 * 100, 3),
            "P(差<=0)": round(float((means <= 0).mean()), 4),
            "P5_pct": round(float(np.percentile(means, 5)) * 252 * 100, 3),
            "P95_pct": round(float(np.percentile(means, 95)) * 252 * 100, 3)}


def main(days: int) -> None:
    suffix = "_1500" if days == 1500 else "_3000"
    dm, navm = _series(OUT / f"backtest{suffix if days == 3000 else ''}.json"
                       if (OUT / f"backtest{suffix}.json").exists() else OUT / "backtest.json")
    ds, navs = _series(OUT / f"satellite_bt{suffix if days == 3000 else ''}.json")
    common = sorted(set(dm) & set(ds))
    im = {d: i for i, d in enumerate(dm)}
    is_ = {d: i for i, d in enumerate(ds)}
    navm = np.array([navm[im[d]] for d in common])
    navs = np.array([navs[is_[d]] for d in common])
    rm, rs = _ret(navm), _ret(navs)
    # 两条腿的净值都已含闲置现金 1.5% 计息(第197轮把副腿补齐), 这里不再重复补。
    sat = json.loads((OUT / f"satellite_bt{suffix if days == 3000 else ''}.json").read_text(encoding="utf-8"))
    hold = _hold_mask(common, sat.get("trades", []), "entry_date", "exit_date")
    print(f"[combo197] {days}日 对齐 {len(common)} 天; 副腿在场 {hold.sum()} 天({hold.mean():.0%})",
          flush=True)

    res = {"days": days, "n_days": len(common),
           "main": _metrics(rm), "sat": _metrics(rs), "grid": {}, "dyn": {}, "boot": {}}
    for w0 in (0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0):
        w = np.full(len(rm), w0)
        res["grid"][f"{int(w0 * 100)}/{100 - int(w0 * 100)}"] = _metrics(_sim(rm, rs, w, common))
    base_w = np.full(len(rm), 0.6)
    base_r = _sim(rm, rs, base_w, common)
    res["grid"]["60/40"] = _metrics(base_r)

    for mode, label in (("A", "等风险"), ("B", "动量"), ("C60", "固定60/40")):
        w = _weights(rm, rs, mode, common)
        r = _sim(rm, rs, w, common)
        res["dyn"][label] = _metrics(r)
        res["dyn"][label]["w_mean"] = round(float(w[LB:].mean()), 3)
        res["dyn"][label]["w_min"] = round(float(w[LB:].min()), 3)
        res["dyn"][label]["w_max"] = round(float(w[LB:].max()), 3)
        if mode != "C60":
            res["boot"][label] = _block_boot(r - base_r)

    (OUT / f"exp_combo_197_{days}.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1500)
