"""第219轮补: 波动分档 → 凯利效率(μ/σ) 是否随波动率单调?

动机:
  第 170/190/191 轮证明"要到 200% 只能靠杠杆", 而杠杆的安全上限由**爆仓概率**决定,
  爆仓概率又由 μ/σ(单笔收益均值/标准差)决定 —— 1.35x 以上爆仓概率从 0.83% 跳到 5.6%、
  1.70x 到 47.9%。所以真正的问题不是"敢不敢加杠杆", 而是"能不能在不降 μ 的前提下降 σ"。

  主策略从未测过"波动率"这个维度(副策略的 lowvol 是另一种逻辑: 选低波票持有 10 日)。
  本轮只做一件事: 把 353 笔按 **T 日之前的 20 日波动率** 分五档, 看每档的
  单笔均值 μ、单笔标准差 σ、以及凯利分数 f*=μ/σ²。

  如果低波档的 f* 显著高于高波档, 说明"缩池降波"是提高安全杠杆上限的合法手段,
  而不是收益优化 —— 这是一条**间接**路径: 不直接提年化, 而是提"可承受的杠杆倍数"。

口径:
  · 波动率 = signal_date 之前 20 个交易日 pct 的标准差(不含 signal_date 当天? 含, 因为 T 日盘后
    才知道当天涨跌, 属 PIT 可得)。用 panel 的 pct 字段。
  · 分档按全样本五分位, 档内统计 μ/σ。样本 353 笔, 每档约 70 笔 —— 只做方向判断, 不做显著性。
"""
from __future__ import annotations

import copy
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.backtest import run_backtest  # noqa: E402


def load_trades(days: int) -> list[dict]:
    import os

    os.environ["BT_FULL_TRADES"] = "1"
    cfg = load_config()
    res = run_backtest(cfg, days)
    tr = [x for x in (res.get("trades_all") or []) if x.get("ret_pct") is not None]
    print(f"交易 {len(tr)} 笔", flush=True)
    return tr


def vol_lookup(trades: list[dict]) -> dict:
    """(code6, date) -> signal_date 前 20 日 pct 标准差。"""
    need = {}
    for t in trades:
        need.setdefault(t["code"][:6], set()).add(str(t["signal_date"]))
    dates = sorted({d for s in need.values() for d in s})
    # 需要窗口内的所有交易日: 用面板日期序列一次性定位
    files = sorted(glob.glob(str(ROOT / "data" / "panel" / "*.csv.gz")))
    all_dates = [Path(f).name[:8] for f in files]
    idx = {d: i for i, d in enumerate(all_dates)}
    wanted_dates: set[str] = set()
    for d in dates:
        i = idx.get(d)
        if i is None:
            continue
        for j in range(max(0, i - 24), i + 1):
            wanted_dates.add(all_dates[j])
    print(f"需读 {len(wanted_dates)} 个交易日", flush=True)

    rows = []
    for i, f in enumerate(files):
        d = Path(f).name[:8]
        if d not in wanted_dates:
            continue
        x = pd.read_csv(f, usecols=["ts_code", "pct"], dtype={"ts_code": str})
        x["code6"] = x["ts_code"].str[:6]
        x = x[x["code6"].isin(need)]
        if len(x):
            x["date"] = d
            rows.append(x[["code6", "date", "pct"]])
    px = pd.concat(rows, ignore_index=True)
    px = px.sort_values(["code6", "date"])
    px["vol20"] = (px.groupby("code6")["pct"]
                     .transform(lambda s: s.astype(float).rolling(20, min_periods=12).std()))
    m = px.set_index(["code6", "date"])["vol20"].to_dict()
    return {(t["code"][:6], str(t["signal_date"])): m.get((t["code"][:6], str(t["signal_date"])))
            for t in trades}


def main() -> None:
    days = 1500
    tr = load_trades(days)
    vmap = vol_lookup(tr)
    for t in tr:
        t["vol20"] = vmap.get((t["code"][:6], str(t["signal_date"])))
    ok = [t for t in tr if t.get("vol20") is not None and np.isfinite(t["vol20"])]
    print(f"有波动率的 {len(ok)}/{len(tr)} 笔", flush=True)

    df = pd.DataFrame([{"vol20": t["vol20"], "ret": t["ret_pct"]} for t in ok])
    df["q"] = pd.qcut(df["vol20"], 5, labels=False, duplicates="drop")

    mu_all = df["ret"].mean()
    sd_all = df["ret"].std()
    out = {"全样本": {"n": int(len(df)), "mu": round(mu_all, 3), "sigma": round(sd_all, 3),
                    "mu/sigma": round(mu_all / sd_all, 3),
                    "kelly": round(mu_all / sd_all ** 2, 3),
                    # 复利年化近似: 353 笔/6.1 年 = 57.9 笔/年
                    "年化%": round(((1 + mu_all / 100) ** (len(df) / 6.1) - 1) * 100, 1)},
           "分档": []}
    print(f"\n{'档':<4}{'n':>5}{'vol20中位':>11}{'mu%':>8}{'sigma%':>8}{'mu/sigma':>10}{'kelly':>8}{'年化%':>9}")
    for q, g in df.groupby("q"):
        mu, sd = g["ret"].mean(), g["ret"].std()
        row = {"档": int(q) + 1, "n": int(len(g)),
               "vol20中位": round(float(g["vol20"].median()), 3),
               "mu": round(float(mu), 3), "sigma": round(float(sd), 3),
               "mu/sigma": round(float(mu / sd), 4),
               "kelly": round(float(mu / sd ** 2), 3),
               "年化%": round(float(((1 + mu / 100) ** (len(g) / 6.1) - 1) * 100), 1)}
        out["分档"].append(row)
        print(f"{row['档']:<4}{row['n']:>5}{row['vol20中位']:>11.2f}{row['mu']:>8.3f}"
              f"{row['sigma']:>8.3f}{row['mu/sigma']:>10.3f}{row['kelly']:>8.3f}{row['年化%']:>9.1f}")

    # ---- B. 评分分档: 选股排序在顶部是否仍然有效 ----
    #   214 轮扫过评分**权重**, 但从没看过"评分本身与收益的关系形状"。
    sc = [t for t in tr if t.get("score") is not None]
    if len(sc) >= 100:
        d2 = pd.DataFrame([{"score": float(t["score"]), "ret": float(t["ret_pct"])} for t in sc])
        d2["q"] = pd.qcut(d2["score"], 5, labels=False, duplicates="drop")
        out["评分分档"] = []
        print(f"\n评分分档 (n={len(d2)}):")
        print(f"{'档':<4}{'n':>5}{'score中位':>11}{'mu%':>8}{'sigma%':>8}{'胜率%':>8}")
        for q, g in d2.groupby("q"):
            row = {"档": int(q) + 1, "n": int(len(g)),
                   "score中位": round(float(g["score"].median()), 3),
                   "mu": round(float(g["ret"].mean()), 3),
                   "sigma": round(float(g["ret"].std()), 3),
                   "胜率%": round(float((g["ret"] > 0).mean() * 100), 1)}
            out["评分分档"].append(row)
            print(f"{row['档']:<4}{row['n']:>5}{row['score中位']:>11.3f}{row['mu']:>8.3f}"
                  f"{row['sigma']:>8.3f}{row['胜率%']:>8.1f}")
        rho = d2[["score", "ret"]].corr(method="spearman").iloc[0, 1]
        out["评分_收益_秩相关"] = round(float(rho), 4)
        print(f"评分 vs 收益 秩相关 = {rho:+.4f}")

    # 低波 40% / 60% 子集的合并口径
    for frac, name in [(0.4, "低波40%"), (0.6, "低波60%")]:
        thr = df["vol20"].quantile(frac)
        g = df[df["vol20"] <= thr]
        mu, sd = g["ret"].mean(), g["ret"].std()
        out[name] = {"n": int(len(g)), "阈值": round(float(thr), 3),
                     "mu": round(float(mu), 3), "sigma": round(float(sd), 3),
                     "mu/sigma": round(float(mu / sd), 4),
                     "kelly": round(float(mu / sd ** 2), 3),
                     "年化%": round(float(((1 + mu / 100) ** (len(g) / 6.1) - 1) * 100), 1)}
        print(f"{name}: n={len(g)} 阈值 vol20<={thr:.2f} mu={mu:.3f} sigma={sd:.3f} "
              f"mu/sigma={mu/sd:.3f} kelly={mu/sd**2:.3f}")

    p = ROOT / "output" / "exp_kelly_219.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved", p)


if __name__ == "__main__":
    main()
