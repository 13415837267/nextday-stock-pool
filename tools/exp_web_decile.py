# -*- coding: utf-8 -*-
"""第184轮·F: 十分位收益 —— 为什么"IC 显著"却"策略更差"

IC 是**整个截面**的秩相关: 它只说"因子值小的那半边, 平均比大的那半边好",
并不保证"最小的那 1% 里随便挑一只就是最好的"。单吊策略买的恰恰是第 1 名。

本脚本把每个因子当日横截面分 10 组(D1=最低, D10=最高), 统计各组未来 10 日
平均收益, 并单独给出"最极端 1%(D1 内部)"的收益 —— 用来判断:
  · 单调性是否成立(IC 显著的前提)
  · 极端端是否反转(单吊策略失效的直接原因)

成本参考: 单边 0.15%, 往返 0.30%。D 组收益未扣成本。

用法: python tools/exp_web_decile.py 1500
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context  # noqa: E402
from src.config import load_config  # noqa: E402

FACTORS = ["ivol20", "max20", "imax20", "on_abs10", "ctc20", "to_ma20", "mom20",
           "mom20_ivol"]
H = 10


def _deciles(df: pd.DataFrame, fac: str, tag: str) -> dict:
    d = df["trade_date"]
    v = df[fac].astype(float)
    r = v.groupby(d).rank(pct=True)
    g = np.ceil(r * 10).clip(1, 10)
    m = df["fwd"].groupby([d, g]).mean().groupby(level=1).mean()
    ext = df.loc[r <= 0.01, "fwd"].mean()
    mn = {int(k): round(float(x), 2) for k, x in m.items()}
    print(f"{tag}{fac:11s} D1 {mn.get(1, 0):+.2f} D2 {mn.get(2, 0):+.2f} "
          f"D3 {mn.get(3, 0):+.2f} ... D8 {mn.get(8, 0):+.2f} D9 {mn.get(9, 0):+.2f} "
          f"D10 {mn.get(10, 0):+.2f} | 极端1% {float(ext):+.2f}", flush=True)
    return {"decile": mn, "extreme_low1pct": round(float(ext), 2), "n": int(len(df))}


def main():
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    cfg = load_config()
    ctx = prepare_context(cfg, days)
    # 内存纪律: 容器 cgroup 上限 8G, 全列 concat(60+ 列 x 800 万行)会直接被 OOM 杀掉
    # (本轮踩过两次, 表现为"无输出 failed")。只取本分析要用的 13 列, 并释放 by_date。
    cols = ["ts_code", "trade_date", "is_paused", "close", "close_adj", "pct",
            "ivol20", "max20", "imax20", "on_abs10", "ctc20", "to_ma20", "mom20"]
    cols = [c for c in cols if c in next(iter(ctx["by_date"].values())).columns]
    df = pd.concat([v[cols] for v in ctx["by_date"].values()], ignore_index=True)
    ctx.clear()
    df = df[(df["is_paused"] == 0) & (df["close"] > 0)].drop(columns=["is_paused", "close"])
    ca = df["close_adj"]
    fwd = ca.groupby(df["ts_code"], sort=False).shift(-H) / ca - 1
    df["fwd"] = (fwd.where(fwd.abs() < 0.8) * 100).astype("float32")
    df["trade_date"] = df["trade_date"].astype(str)
    d = df["trade_date"]

    out = {}
    # 全市场 / 低波池内 两套口径。池内口径才是策略的真实战场:
    # 副策略先用 ivol 分位砍到前 10%, 再在池内排序 —— 若池内十分位平坦,
    # 就说明"池边界已经把因子的 alpha 榨干了", 换排序键自然无用。
    df["_ivp"] = df["ivol20"].astype(float).groupby(d).rank(pct=True)
    blocks = {"全市场": df, "低波池内": df[df["_ivp"] <= 0.10]}
    for tag, sub in blocks.items():
        if not len(sub):
            continue
        out[tag] = {}
        for fac in FACTORS:
            if fac not in sub.columns and fac != "mom20_ivol":
                continue
            if fac == "mom20_ivol":
                s = sub.copy()
                s["mom20_ivol"] = -(s["mom20"].astype(float)
                                    / s["ivol20"].astype(float).replace(0, np.nan))
                out[tag][fac] = _deciles(s, fac, tag + " ")
            else:
                out[tag][fac] = _deciles(sub, fac, tag + " ")

    p = ROOT / f"output/exp_web_decile_{days}.json"
    p.write_text(json.dumps({"days": days, "horizon": H, "blocks": out},
                            ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"写出 {p}", flush=True)


if __name__ == "__main__":
    main()
