# -*- coding: utf-8 -*-
"""trend 的可学习性检验: 监督模型 + dip 阳性对照(2026-11-05)。

为什么还要做这一步
------------------
tools/exp_trend_cond.py 在训练集(前 60%)扫了 85 个单变量桶, **没有一个**同时满足
n>=60 / avg>=+0.10% / t>=2.0; 训练集里最好的几个桶在验证集全部归零或转负。
但这仍可能被质疑: "分桶太粗, 也许存在非线性/交互结构没被捕捉到。"

本脚本用监督学习正面回答"有没有可学的信号":
  · 特征同 exp_trend_cond(个股形态 + 市场状态, 全部 T 日收盘可得);
  · 模型 ridge 回归(闭式解, 无超参搜索负担), 目标 = T+1 开盘买 / T+2 收盘卖净收益;
  · 训练集拟合 -> 验证集评估, 看 **AUC / 分档单调性 / top 档期望**, 而不是看拟合优度。

关键设计: **dip 作为阳性对照**
------------------------------
如果只有 trend 学不出来, 无法区分"模型太弱"和"trend 没信号";
把同一套特征、同一个模型套在 dip 上, 若 dip 也学不出来 —— 而 dip 实盘是赚钱的 ——
则说明: 这套 alpha **本来就不住在"单笔方向可预测"里**, 而住在过滤器(资金约束/缺口闸/评分门槛)
造成的选择偏差里(第一百三十三/一百三十一轮结论)。此时"模型学不出 trend"才是可信的否定。

口径
----
  · ridge 闭式解 w = (X'X + lam·I)^-1 X'y, y 为净收益(%); 特征按训练集均值/标准差标准化;
  · lam 网格 0.1/1/10, 在训练集内部按时间做 3 折走前验证选 lam, 验证集只看一次;
  · AUC 用秩公式(不处理并列, 特征连续可忽略);
  · 分档按验证集预测值 5 等分, 报告每档 n/avg/win/t —— 单调性比单点数值更重要。

用法
----
  python3 tools/exp_trend_ml.py
  python3 tools/exp_trend_ml.py --days 3378
"""
import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["BT_FULL_TRADES"] = "1"

from src.backtest import prepare_context  # noqa: E402
from src.config import load_config  # noqa: E402
from src.selector import select_stocks  # noqa: E402

COST = 0.0030
TRAIN_FRAC = 0.60
LAMS = (0.1, 1.0, 10.0)
STOCK_FEATS = ["pct", "mom20", "close_pos", "vol_ratio", "bias20", "dist_high20",
               "dist_low20", "turnover_rate", "turnover_cool", "log_amount", "gap_pct"]
MKT_FEATS = ["breadth", "mkt_pct", "mkt_mom5", "mkt_mom20", "mkt_vol20", "mkt_bias20"]
FEATS = STOCK_FEATS + MKT_FEATS


def _px(sub, d):
    try:
        if d in sub.index:
            return sub.loc[d]
    except Exception:  # noqa: BLE001
        pass
    return None


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return np.nan
    return f if np.isfinite(f) else np.nan


def _stats(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    n = len(v)
    if n == 0:
        return None
    m = float(np.mean(v))
    s = float(np.std(v, ddof=1)) if n > 1 else 0.0
    t = (m / (s / np.sqrt(n))) if s > 0 else 0.0
    return {"n": n, "avg": round(m, 3), "win": round(100 * sum(1 for x in v if x > 0) / n, 1),
            "t": round(t, 2)}


def _fmt(st):
    return ("n=%3d %+6.2f%% 胜%2.0f%% t=%+5.2f" % (st["n"], st["avg"], st["win"], st["t"])) if st else "-"


def _auc(y, s):
    y = np.asarray(y, dtype=float)
    s = np.asarray(s, dtype=float)
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return 0.5
    order = np.argsort(s)
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _collect(cfg0, days, mode):
    """返回 (records, main_dates), 每笔含 T 日特征 + T+1 买 T+2 卖净收益。"""
    if not hasattr(_collect, "_ctx"):
        print("[1/3] 构建上下文 %d 日 ..." % days, flush=True)
        t0 = time.time()
        _collect._ctx = prepare_context(cfg0, days)
        print("      ctx ready %.0fs" % (time.time() - t0), flush=True)
    ctx = _collect._ctx
    by_date, per_code, reg, main_dates = ctx["by_date"], ctx["per_code"], ctx["reg"], ctx["main_dates"]
    dates = sorted(by_date)
    pos = {d: i for i, d in enumerate(dates)}

    reg = reg.copy()
    reg["mkt_mom5"] = reg["mkt_idx"] / reg["mkt_idx"].shift(5) - 1
    reg["mkt_mom20"] = reg["mkt_idx"] / reg["mkt_idx"].shift(20) - 1
    reg["mkt_vol20"] = reg["mkt_pct"].rolling(20).std()
    reg["mkt_bias20"] = reg["mkt_idx"] / reg["mkt_ma"] - 1

    cfg = copy.deepcopy(cfg0)
    cfg["selection"]["fallbacks"] = []
    if mode != "dip":
        cfg["selection"]["mode"] = mode

    recs = []
    t0 = time.time()
    for d in main_dates:
        day = by_date[d]
        i = pos[d]
        if i + 1 >= len(dates):
            continue
        picks, _ = select_stocks(d, day, cfg, None, top_n=1, name_filter=False)
        if not picks:
            continue
        code = picks[0]["code"]
        sub = per_code.get(code)
        if sub is None:
            continue
        d1 = dates[i + 1]
        r1 = _px(sub, d1)
        if r1 is None or r1["is_paused"] == 1 or r1["vol"] <= 0:
            continue
        if r1["open"] >= r1["high_limit"] - 1e-9:
            continue
        buy = float(r1["open_adj"])
        if buy <= 0:
            continue
        sig = day.loc[day["ts_code"] == code, "close_adj"]
        if not len(sig):
            continue
        gap = (buy / float(sig.iloc[0]) - 1) * 100
        d2 = dates[i + 2] if i + 2 < len(dates) else None
        r2 = _px(sub, d2) if d2 else None
        if r2 is None:
            continue
        y = (float(r2["close_adj"]) / buy - 1) * 100 - COST * 100
        row = {"signal_date": d, "code": code, "y": y, "gap_pct": gap}
        srow = day.loc[day["ts_code"] == code]
        for f in STOCK_FEATS:
            if f == "gap_pct":
                continue
            if f == "log_amount":
                a = _num(srow["amount"].iloc[0]) if "amount" in srow else np.nan
                row[f] = float(np.log10(a)) if np.isfinite(a) and a > 0 else np.nan
            else:
                row[f] = _num(srow[f].iloc[0]) if f in srow else np.nan
        mrow = reg.loc[d] if d in reg.index else None
        for f in MKT_FEATS:
            row[f] = _num(mrow[f]) if mrow is not None and f in mrow.index else np.nan
        recs.append(row)
    print("      %s 样本 %d 笔 (%.0fs)" % (mode, len(recs), time.time() - t0), flush=True)
    return recs, main_dates


def _design(tr, te):
    """按训练集标准化; 缺失值先填训练集中位数再用 0(即均值)代替。"""
    Xtr = np.array([[r.get(f, np.nan) for f in FEATS] for r in tr], dtype=float)
    Xte = np.array([[r.get(f, np.nan) for f in FEATS] for r in te], dtype=float)
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    for X in (Xtr, Xte):
        idx = np.where(~np.isfinite(X))
        X[idx] = np.take(med, idx[1])
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0)
    sd = np.where(sd > 1e-9, sd, 1.0)
    return (Xtr - mu) / sd, (Xte - mu) / sd, mu, sd


def _ridge(X, y, lam):
    A = X.T @ X + lam * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ y)


def _fit_predict(tr, te):
    """训练集内部按时间 3 折走前验证选 lam, 再用全训练集重拟合, 预测验证集。"""
    Xtr, Xte, mu, sd = _design(tr, te)
    ytr = np.array([r["y"] for r in tr], dtype=float)
    n = len(tr)
    folds = [(0, int(n * 0.5), int(n * 0.75)), (int(n * 0.5), int(n * 0.75), n)]
    best_lam, best_mse = LAMS[0], None
    for lam in LAMS:
        ses = []
        for a, b, c in folds:
            w = _ridge(Xtr[a:b], ytr[a:b], lam)
            pred = Xtr[b:c] @ w
            ses.append(float(np.mean((pred - ytr[b:c]) ** 2)))
        mse = float(np.mean(ses))
        if best_mse is None or mse < best_mse:
            best_lam, best_mse = lam, mse
    w = _ridge(Xtr, ytr, best_lam)
    return Xte @ w, Xtr @ w, w, best_lam


def _eval(name, tr, va, out):
    print("\n=== %s: 训练 %d 笔 / 验证 %d 笔 ===" % (name, len(tr), len(va)), flush=True)
    pred, pred_tr, w, lam = _fit_predict(tr, va)
    yv = np.array([r["y"] for r in va], dtype=float)
    ytr = np.array([r["y"] for r in tr], dtype=float)
    a_tr = _auc((ytr > 0).astype(int), pred_tr)
    a_va = _auc((yv > 0).astype(int), pred)
    print("  lambda=%.1f | AUC 训练 %.3f / 验证 %.3f  (0.5=无信息)" % (lam, a_tr, a_va), flush=True)

    # 验证集按预测值 5 等分, 看单调性
    order = np.argsort(pred)
    qs = np.array_split(order, 5)
    print("  验证集按预测收益分 5 档(低->高):", flush=True)
    rows = []
    for i, q in enumerate(qs):
        st = _stats([yv[j] for j in q])
        rows.append({"q": i + 1, "stats": st})
        print("     Q%d %s" % (i + 1, _fmt(st)), flush=True)
    top = _stats([yv[j] for j in qs[-1]])
    bot = _stats([yv[j] for j in qs[0]])
    spread = (top["avg"] - bot["avg"]) if top and bot else None
    print("  顶档-底档价差 %s pp" % (("%+.2f" % spread) if spread is not None else "-"), flush=True)

    imp = sorted(zip(FEATS, w), key=lambda x: -abs(x[1]))[:6]
    print("  标准化系数绝对值前 6: " + ", ".join("%s %+.3f" % (k, v) for k, v in imp), flush=True)

    out[name] = {"n_train": len(tr), "n_val": len(va), "lam": lam,
                 "auc_train": round(a_tr, 3), "auc_val": round(a_va, 3),
                 "base": _stats(list(yv)),
                 "buckets": rows, "top": top, "bottom": bot, "spread": spread,
                 "coef": {k: round(float(v), 4) for k, v in zip(FEATS, w)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=1500)
    ap.add_argument("--out", default="output/exp_trend_ml.json")
    args = ap.parse_args()

    cfg0 = load_config()
    out = {"days": args.days, "features": FEATS, "cost_pct": COST * 100}
    main_dates = None
    for mode in ("trend", "dip"):
        recs, main_dates = _collect(cfg0, args.days, mode)
        cut = main_dates[int(len(main_dates) * TRAIN_FRAC)]
        tr = [r for r in recs if r["signal_date"] < cut]
        va = [r for r in recs if r["signal_date"] >= cut]
        out["cut"] = cut
        _eval(mode, tr, va, out)

    # 基准: 不做任何预测, 全验证集均值 —— 顶档若不超过它, 模型就没有信息量
    print("\n=== 参照系: 验证集全样本均值(不做任何筛选) ===", flush=True)
    for mode in ("trend", "dip"):
        if mode in out and out[mode]["top"]:
            base = out[mode]["base"]
            print("  %-6s 全样本 %+0.2f%%  |  模型顶档 %+0.2f%%  |  差 %+.2fpp"
                  % (mode, base["avg"], out[mode]["top"]["avg"], out[mode]["top"]["avg"] - base["avg"]),
                  flush=True)

    op = Path(args.out)
    if not op.is_absolute():
        op = ROOT / op
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print("\n结果已写 %s" % op, flush=True)


if __name__ == "__main__":
    main()
