# -*- coding: utf-8 -*-
"""趋势策略的适用条件搜索: 训练集找条件 -> 验证集只验一次(2026-11-05)。

前情
----
tools/exp_trend.py / exp_trend_hold.py 已确认三件事:
  · trend 单独当主力 1500 日 +1.36%(单笔 -0.083%);
  · "dip 未成交 -> trend 顶上" 91 笔补位均值 -0.35%;
  · 信号与 dip 完全正交(同日同票 0 天), 不是选票重复问题。
但"整体负期望"不等于"任何条件下都负期望"。本脚本回答的是条件问题:
  **在什么市场状态 / 什么个股形态下, trend 票的次日买入是正期望的?**

方法论(为什么必须切分)
----------------------
1500 日、1400+ 笔样本, 若在全样本上扫条件, 扫 100 个桶必然有若干个"显著为正"——
那是抽样噪声, 不是规律。故严格三段式:
  1. 训练集(前 60% 交易日): 单变量 5 分位分桶 + 两两组合, 找 avg/t 双高的条件;
  2. 记录搜索空间大小 M, 用 Bonferroni 校正后的 t 门槛判"训练集显著";
  3. 验证集(后 40%): 候选条件**只跑一次**, 阈值全部沿用训练集分位点, 不回头调参。
同时打印"条件命中率"——命中率 5% 的条件即使正期望, 一年也只有 12 次机会, 做不了副策略。

口径
----
  · 信号 T 日收盘选出 top1; 买入 T+1 开盘(剔除停牌/一字涨停买不到); 卖出 T+2 收盘;
  · 成本 0.30%/笔(滑点千1×2 + 佣金万2.5×2 + 印花税万5), 与 exp_trend_hold 一致;
  · 另记 k=3/5 的持有期收益, 用于判断"是否需要差异化出口"(趋势票可能被 T+2 憋死);
  · 所有特征只用 T 日及之前的数据; 卖出价是仿真必需的事后价格, 不参与选条件。

用法
----
  python3 tools/exp_trend_cond.py               # 1500 日
  python3 tools/exp_trend_cond.py --days 3378   # 全史(样本更多但跨制度, 谨慎)
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
KS = (1, 3, 5)
TRAIN_FRAC = 0.60
MIN_N_TRAIN = 60        # 一个桶至少 60 笔才进入候选(再少 t 值不可信)
MIN_N_VAL = 40          # 验证集至少 40 笔才算"可评估"
MIN_AVG = 0.10          # 候选门槛: 扣成本后单笔至少 +0.10%(否则赚不过摩擦)


# ----------------------------------------------------------------------
def _px(sub, d):
    try:
        if d in sub.index:
            return sub.loc[d]
    except Exception:  # noqa: BLE001
        pass
    return None


def _stats(vals):
    n = len(vals)
    if n == 0:
        return None
    m = float(np.mean(vals))
    s = float(np.std(vals, ddof=1)) if n > 1 else 0.0
    t = (m / (s / np.sqrt(n))) if s > 0 else 0.0
    return {"n": n, "avg": round(m, 3), "win": round(100 * sum(1 for x in vals if x > 0) / n, 1),
            "med": round(float(np.median(vals)), 3), "t": round(t, 2)}


def _fmt(st):
    return ("n=%3d %+6.2f%% 胜%2.0f%% t=%+5.2f" % (st["n"], st["avg"], st["win"], st["t"])) if st else "-"


# ----------------------------------------------------------------------
STOCK_FEATS = ["pct", "mom20", "close_pos", "vol_ratio", "bias20", "dist_high20",
               "dist_low20", "turnover_rate", "turnover_cool", "log_amount"]
MKT_FEATS = ["breadth", "mkt_pct", "mkt_mom5", "mkt_mom20", "mkt_vol20", "mkt_bias20", "pool_n"]
FEATS = STOCK_FEATS + MKT_FEATS


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _collect(cfg0, days):
    """逐日取 dip/trend 的 top1, 展开 T+1 买 / T+k 卖的净收益与全部事前特征。"""
    print("[1/2] 构建上下文 %d 日 ..." % days, flush=True)
    t0 = time.time()
    ctx = prepare_context(cfg0, days)
    print("      ctx ready %.0fs" % (time.time() - t0), flush=True)

    by_date, per_code, reg = ctx["by_date"], ctx["per_code"], ctx["reg"]
    dates = sorted(by_date)
    pos = {d: i for i, d in enumerate(dates)}
    main_dates = ctx["main_dates"]

    # 市场派生特征(仅用截至当日数据)
    reg = reg.copy()
    reg["mkt_mom5"] = reg["mkt_idx"] / reg["mkt_idx"].shift(5) - 1
    reg["mkt_mom20"] = reg["mkt_idx"] / reg["mkt_idx"].shift(20) - 1
    reg["mkt_vol20"] = reg["mkt_pct"].rolling(20).std()
    reg["mkt_bias20"] = reg["mkt_idx"] / reg["mkt_ma"] - 1

    cfg_dip = copy.deepcopy(cfg0)
    cfg_dip["selection"]["fallbacks"] = []
    cfg_tr = copy.deepcopy(cfg0)
    cfg_tr["selection"]["mode"] = "trend"
    cfg_tr["selection"]["fallbacks"] = []

    recs = []
    for d in main_dates:
        day = by_date[d]
        i = pos[d]
        if i + 1 >= len(dates):
            continue
        d1 = dates[i + 1]
        mrow = reg.loc[d] if d in reg.index else None

        picks_dip, _ = select_stocks(d, day, cfg_dip, None, top_n=1, name_filter=False)
        picks_tr, st_tr = select_stocks(d, day, cfg_tr, None, top_n=1, name_filter=False)
        if not picks_tr:
            continue
        p = picks_tr[0]
        code = p["code"]
        sub = per_code.get(code)
        if sub is None:
            continue
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
        sig_px = float(sig.iloc[0])
        row = {"signal_date": d, "code": code, "name": p.get("name", ""),
               "score": p.get("score"), "gap_pct": round((buy / sig_px - 1) * 100, 2),
               "pool_n": int((st_tr or {}).get("scored", 0) or 0),
               "dip_has": bool(picks_dip),
               "dip_code": (picks_dip[0]["code"] if picks_dip else None)}
        srow = day.loc[day["ts_code"] == code]
        for f in STOCK_FEATS:
            if f == "log_amount":
                a = _num(srow["amount"].iloc[0]) if "amount" in srow else None
                row[f] = round(float(np.log10(a)), 3) if a and a > 0 else None
            else:
                row[f] = _num(srow[f].iloc[0]) if f in srow else None
        for f in MKT_FEATS:
            if f == "pool_n":
                continue
            row[f] = _num(mrow[f]) if mrow is not None and f in mrow.index else None
        row["regime_ok"] = bool(mrow["trade_ok"]) if mrow is not None else True
        row["panic"] = bool(mrow["panic"]) if mrow is not None and "panic" in mrow.index else False
        for k in KS:
            j = i + 1 + k
            rk = _px(sub, dates[j]) if j < len(dates) else None
            row["r%d" % k] = (round((float(rk["close_adj"]) / buy - 1) * 100 - COST * 100, 3)
                              if rk is not None else None)
        recs.append(row)
    print("      样本 %d 笔 (%.0fs)" % (len(recs), time.time() - t0), flush=True)
    return recs, main_dates


# ----------------------------------------------------------------------
def _split(recs, main_dates):
    """按交易日时间切分, 训练在前验证在后(不是随机切分——随机切会让同一天的
    市场状态同时出现在两侧, 造成"同一次行情重复计数"的乐观偏差)。"""
    cut = main_dates[int(len(main_dates) * TRAIN_FRAC)]
    tr = [r for r in recs if r["signal_date"] < cut]
    va = [r for r in recs if r["signal_date"] >= cut]
    return tr, va, cut


def _scan_univariate(tr, va, ks):
    """训练集分位分桶, 阈值沿用到验证集。返回每特征每桶的双段统计。"""
    out = {}
    for f in FEATS:
        tv = [r[f] for r in tr if r.get(f) is not None and r.get("r%d" % ks) is not None]
        if len(tv) < MIN_N_TRAIN * 3:
            continue
        qs = [float(np.percentile(tv, q)) for q in (20, 40, 60, 80)]
        edges = [-np.inf] + qs + [np.inf]
        buckets = []
        for b in range(5):
            lo, hi = edges[b], edges[b + 1]
            def _in(r):
                v = r.get(f)
                return v is not None and lo <= v < hi
            a = [r["r%d" % ks] for r in tr if _in(r) and r.get("r%d" % ks) is not None]
            c = [r["r%d" % ks] for r in va if _in(r) and r.get("r%d" % ks) is not None]
            buckets.append({"b": b, "lo": (None if lo == -np.inf else round(lo, 4)),
                            "hi": (None if hi == np.inf else round(hi, 4)),
                            "train": _stats(a), "val": _stats(c)})
        out[f] = buckets
    return out


def _mask(r, cond):
    f, lo, hi = cond
    v = r.get(f)
    if v is None:
        return False
    return (lo is None or v >= lo) and (hi is None or v < hi)


def _combos(tr, va, cands, ks):
    """候选条件两两 AND 组合(仅训练集选, 验证集只读)。"""
    key = "r%d" % ks
    out = []
    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            c1, c2 = cands[i], cands[j]
            if c1[0] == c2[0]:
                continue
            rows = [r for r in tr if _mask(r, c1) and _mask(r, c2) and r.get(key) is not None]
            st = _stats([r[key] for r in rows])
            if not st or st["n"] < MIN_N_TRAIN:
                continue
            vr = [r for r in va if _mask(r, c1) and _mask(r, c2) and r.get(key) is not None]
            out.append({"cond": [list(c1), list(c2)], "train": st,
                        "val": _stats([r[key] for r in vr])})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=1500)
    ap.add_argument("--out", default="output/exp_trend_cond.json")
    args = ap.parse_args()

    cfg0 = load_config()
    recs, main_dates = _collect(cfg0, args.days)
    tr, va, cut = _split(recs, main_dates)
    print("\n=== 切分 ===")
    print("  训练: %s ~ %s  n=%d 笔" % (tr[0]["signal_date"], tr[-1]["signal_date"], len(tr)))
    print("  验证: %s ~ %s  n=%d 笔" % (va[0]["signal_date"], va[-1]["signal_date"], len(va)))
    print("  全样本 trend 均值 r1/r3/r5: %s / %s / %s" % (
        _fmt(_stats([r["r1"] for r in recs if r.get("r1") is not None])),
        _fmt(_stats([r["r3"] for r in recs if r.get("r3") is not None])),
        _fmt(_stats([r["r5"] for r in recs if r.get("r5") is not None]))))

    result = {"days": args.days, "cut": cut, "n_train": len(tr), "n_val": len(va),
              "overall": {("r%d" % k): _stats([r["r%d" % k] for r in recs if r.get("r%d" % k) is not None])
                          for k in KS}}

    for ks in (1, 5):
        print("\n" + "=" * 78)
        print("=== 持有 k=%d 日的单变量分桶(训练集分位, 验证集沿用同一阈值) ===" % ks)
        uni = _scan_univariate(tr, va, ks)
        result["uni_k%d" % ks] = uni
        cands = []
        for f, bs in uni.items():
            line = "  %-14s" % f
            for b in bs:
                t_, v_ = b["train"], b["val"]
                if t_ and t_["n"] >= MIN_N_TRAIN and t_["avg"] >= MIN_AVG and t_["t"] >= 2.0:
                    cands.append((f, b["lo"], b["hi"]))
                line += " | %s" % (_fmt(t_) if t_ else "-")
            print(line + "   [训练集]", flush=True)
            print("  %-14s" % "", end="")
            for b in bs:
                print(" | %s" % (_fmt(b["val"]) if b["val"] else "-"), end="")
            print("   [验证集]", flush=True)
        print("  -> 训练集入选候选 %d 个 (n>=%d, avg>=%.2f%%, t>=2.0)"
              % (len(cands), MIN_N_TRAIN, MIN_AVG), flush=True)

        # 记录搜索空间, 用于多重比较说明
        m_tests = sum(len(v) for v in uni.values())
        # Bonferroni: 对 m_tests 次独立检验, 单次 alpha = 0.05/m_tests, 用正态近似
        try:
            from statistics import NormalDist
            alpha = 0.05 / max(1, m_tests)
            t_bonf = NormalDist().inv_cdf(1 - alpha / 2)
        except Exception:  # noqa: BLE001
            t_bonf = 3.5
        print("  搜索空间 M=%d 个单变量桶; Bonferroni 门槛 t>=%.2f (alpha=0.05/M)" % (m_tests, t_bonf),
              flush=True)

        combos = _combos(tr, va, cands, ks)
        combos.sort(key=lambda x: -x["train"]["avg"])
        result["combos_k%d" % ks] = combos[:40]
        print("  -> 两两组合 %d 个, 训练集 avg 前 10:" % len(combos), flush=True)
        for c in combos[:10]:
            print("     %-12s[%s,%s) x %-12s[%s,%s)  训练 %s | 验证 %s" % (
                c["cond"][0][0], c["cond"][0][1], c["cond"][0][2],
                c["cond"][1][0], c["cond"][1][1], c["cond"][1][2],
                _fmt(c["train"]), _fmt(c["val"])), flush=True)

    # ---- 分层: 副策略的真实适用域 ----
    print("\n" + "=" * 78)
    print("=== 分层: 副策略只在 dip 不成交时启用, 那部分样本长什么样 ===")
    layers = {
        "全部": lambda r: True,
        "非风控": lambda r: r["regime_ok"],
        "风控日": lambda r: not r["regime_ok"],
        "非风控+dip有票": lambda r: r["regime_ok"] and r["dip_has"],
        "非风控+dip无票": lambda r: r["regime_ok"] and not r["dip_has"],
    }
    result["layers"] = {}
    for name, sel in layers.items():
        rows = [r for r in recs if sel(r)]
        line = "  %-16s n=%4d " % (name, len(rows))
        result["layers"][name] = {}
        for k in KS:
            st = _stats([r["r%d" % k] for r in rows if r.get("r%d" % k) is not None])
            result["layers"][name]["r%d" % k] = st
            line += " | k%d %s" % (k, _fmt(st))
        print(line, flush=True)

    op = Path(args.out)
    if not op.is_absolute():
        op = ROOT / op
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(result, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print("\n结果已写 %s" % op, flush=True)


if __name__ == "__main__":
    main()
