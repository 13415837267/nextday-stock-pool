# -*- coding: utf-8 -*-
"""容量天花板探针: 复利 + 单吊 的规模上限在哪。

只读诊断, 不改 config, 不写 latest.json。产物 output/exp_capacity.json。

为什么做这个
------------
第一百二十八轮把成本敏感性扫完后得到一句结论: "每多 10bp 单边滑点 = 单笔 -0.20pp
≈ 累计 -44%"。但滑点是"假设值", 真正决定它的物理量是**参与率**
(投入金额 / 该股当日成交额)。参与率随资金规模线性上升, 复利又把规模推到指数级,
所以"这套策略能装多少钱"是一个可以算出来的硬边界, 而不是参数偏好。

方法
----
1. 跑一次标准回测(置 BT_FULL_TRADES=1 拿全量成交), 得到每笔的
   code / buy_date / ret_pct / pos_frac。
2. 逐笔取 **T+1 买入日**该股全天成交额 amount(元), 参与率
       p = 投入金额 / amount,  投入金额 = 当期净值 × 初始资金 × pos_frac
3. 冲击成本用平方根模型(Almgren 简化, A 股小票常用标定):
       单边 bp = k × sqrt(p × 100)        # k = bp / sqrt(1% 参与率)
   k 取两档: 15(温和) / 30(凶悍)。双边 = 2 × 单边。
4. **迭代口径**(主): 每笔先扣冲击再更新净值, 让规模随亏损自己收敛 —— 避免
   "用无冲击净值算参与率"的系统性高估。k=0 时必须精确复现回测基线, 作为重建校验。
5. 扫初始资金 5 万 ~ 10 亿。

口径说明(必读)
--------------
amount 是 T+1 **全天**成交额; 策略实际只在 9:25 集合竞价后几分钟内建仓, 可成交量
远小于全天(经验 10~20%)。所以本表给出的是**乐观上界**, 真实容量约为表中数字的
1/5 ~ 1/10。JSON 里给出 open_share 折算列。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# run_backtest 仅在置位该变量时返回 trades_all(全量成交), 默认不落盘
os.environ["BT_FULL_TRADES"] = "1"

from src.config import load_config            # noqa: E402
from src.backtest import prepare_context, run_backtest   # noqa: E402

CAPS = [5e4, 1e5, 2e5, 5e5, 1e6, 2e6, 5e6, 1e7, 2e7, 5e7, 1e8, 2e8, 5e8, 1e9]
KS = [15.0, 30.0]
OPEN_SHARE = 0.15          # 开盘集合竞价段成交额占比(经验值, 用于折算真实容量)
TRADING_DAYS = 252.0


def _amount_of(frame: pd.DataFrame | None, code: str) -> float | None:
    """取某日某票的全天成交额(元); 缺失/停牌返回 None。"""
    if frame is None or not len(frame):
        return None
    col = frame["ts_code"].to_numpy()
    hit = np.flatnonzero(col == code)
    if not len(hit):
        return None
    a = float(frame["amount"].to_numpy()[hit[0]])
    return a if (math.isfinite(a) and a > 0) else None


def _maxdd(navs: np.ndarray) -> float:
    if not len(navs):
        return 0.0
    peak = np.maximum.accumulate(navs)
    return float((navs / peak - 1.0).min() * 100.0)


def _walk(trades: list[dict], amounts: np.ndarray, cap: float, k: float,
          max_p: float = 0.05) -> dict:
    """按成交序列复利推演。cap=初始资金, k=单边冲击系数(bp/sqrt(1%)),
    max_p=单笔最多吃掉当日成交额的比例(物理容量约束, 超出按部分成交)。

    注意 1: 这里用**逐笔串行**复利。回测里单吊+持有两天, 同一日"卖出旧仓/买入新仓"
    不会同时满仓(卖出款次日才可用), 串行重建与真实净值路径的差异用 k=0 且
    max_p=inf 的基线复现精度来度量(见 recon_err)。
    注意 2: 平方根模型在 p>10% 后没有实证支撑, 所以必须先截断成交量再算冲击,
    否则"买下整只票"会算出荒谬的成本并让净值假性归零。
    """
    nav = 1.0
    navs = [nav]
    parts, imps, effs = [], [], []
    zero_at = None
    capped = 0
    for i, (t, amt) in enumerate(zip(trades, amounts)):
        pos = float(t.get("pos_frac") or 0.0)
        r = float(t["ret_pct"]) / 100.0
        if pos <= 0 or nav <= 0:
            continue
        wealth = nav * cap
        want = wealth * pos
        amt_ok = bool(amt and math.isfinite(amt) and amt > 0)
        limit = max_p * amt if amt_ok else want
        invested = min(want, limit)
        if invested < want - 1e-6:
            capped += 1
        p = (invested / amt) if amt_ok else 0.0
        eff_pos = (invested / wealth) if wealth > 0 else 0.0
        imp_bp = 2.0 * k * math.sqrt(max(p, 0.0) * 100.0) if k else 0.0
        parts.append(p * 100.0)
        imps.append(imp_bp)
        effs.append(eff_pos)
        nav *= (1.0 + eff_pos * (r - imp_bp / 10000.0))
        navs.append(nav)
        if nav <= 1e-6:
            nav = 0.0
            zero_at = i + 1
            break
    return {
        "nav": nav,
        "parts": np.asarray(parts, dtype=float),
        "imps": np.asarray(imps, dtype=float),
        "effs": np.asarray(effs, dtype=float),
        "navs": np.asarray(navs, dtype=float),
        "zero_at": zero_at,
        "capped": capped,
        "n": len(parts),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=1500)
    ap.add_argument("--out", default="output/exp_capacity.json")
    args = ap.parse_args()

    t0 = time.time()
    cfg = load_config()
    print(f"[1/3] 构建上下文({args.days} 日)...", flush=True)
    ctx = prepare_context(cfg, args.days)
    print(f"[2/3] 回测(k=0 基线, 全量成交)...", flush=True)
    res = run_backtest(cfg, args.days, ctx)
    m = res["metrics"]
    trades = [t for t in res.get("trades_all", []) if t.get("ret_pct") is not None]
    if not trades:
        raise RuntimeError("未取到全量成交: BT_FULL_TRADES 未生效?")
    print(f"     基线 {m['总收益率%']}% / {m['交易次数']} 笔 / 全量 {len(trades)} 条", flush=True)

    # ---- 逐笔成交额 ----
    by_date = ctx["by_date"]
    amounts, miss, noamt = [], 0, []
    for t in trades:
        d = str(t.get("buy_date") or "")
        a = _amount_of(by_date.get(d), str(t.get("code") or ""))
        if a is None:
            miss += 1
            noamt.append({"buy_date": d, "code": t.get("code")})
        amounts.append(np.nan if a is None else a)
    amounts = np.asarray(amounts, dtype=float)
    ok = np.isfinite(amounts)
    print(f"     成交额匹配 {ok.sum()}/{len(trades)} (缺 {miss})", flush=True)

    if miss:
        # 缺失笔按该票可用日的中位成交额填, 避免整笔被当成无限流动性
        fill = float(np.nanmedian(amounts)) if ok.any() else 1e8
        amounts = np.where(ok, amounts, fill)

    lig = {
        "n": int(ok.sum()),
        "amount_median_yi": round(float(np.nanmedian(amounts)) / 1e8, 3),
        "amount_p10_yi": round(float(np.nanpercentile(amounts, 10)) / 1e8, 3),
        "amount_p25_yi": round(float(np.nanpercentile(amounts, 25)) / 1e8, 3),
        "amount_p90_yi": round(float(np.nanpercentile(amounts, 90)) / 1e8, 3),
        "amount_max_yi": round(float(np.nanmax(amounts)) / 1e8, 2),
        "miss": miss,
    }

    # ---- k=0 + 无容量约束 重建校验 ----
    base = _walk(trades, amounts, 1.0, 0.0, max_p=1e9)
    recon_total = (base["nav"] - 1.0) * 100.0
    recon_err = recon_total - float(m["总收益率%"])

    years = args.days / TRADING_DAYS

    # ---- 扫规模 ----
    print("[3/3] 扫资金规模...", flush=True)
    rows = []
    for cap in CAPS:
        for k in KS:
            w = _walk(trades, amounts, cap, k)
            hb = _walk(trades, amounts, cap, k, max_p=1e9)   # 对照: 不计成交量硬买满
            nav = w["nav"]
            parts, imps, effs = w["parts"], w["imps"], w["effs"]
            cagr = (nav ** (1.0 / years) - 1.0) * 100.0 if nav > 0 else -100.0
            rows.append({
                "cap": cap,
                "cap_wan": round(cap / 1e4, 1),
                "k": k,
                "nav_x": round(nav, 3),
                "total_pct": round((nav - 1.0) * 100.0, 1),
                "cagr_pct": round(cagr, 2),
                "mdd_pct": round(_maxdd(w["navs"]), 2),
                "n": w["n"],
                "zero_at": w["zero_at"],
                # 对照口径: 硬买满(可以吃掉超过 5% 成交额), 看冲击成本本身能吃掉多少
                "total_pct_hardbuy": round((hb["nav"] - 1.0) * 100.0, 1),
                "zero_at_hardbuy": hb["zero_at"],
                "p_mean_pct": round(float(parts.mean()), 4) if len(parts) else 0.0,
                "p_med_pct": round(float(np.median(parts)), 4) if len(parts) else 0.0,
                "p95_pct": round(float(np.percentile(parts, 95)), 4) if len(parts) else 0.0,
                "p_max_pct": round(float(parts.max()), 4) if len(parts) else 0.0,
                "imp_mean_bp": round(float(imps.mean()), 2) if len(imps) else 0.0,
                "eff_pos_pct": round(float(effs.mean() * 100.0), 1) if len(effs) else 0.0,
                "capped_share_pct": round(w["capped"] / max(len(parts), 1) * 100.0, 1),
                "over1pct_share": round(float((parts > 1.0).mean() * 100.0), 1) if len(parts) else 0.0,
                # 按开盘 15% 占比折算后的"等效规模"(保守容量)
                "cap_effective": cap * OPEN_SHARE,
            })

    base_total = float(m["总收益率%"])
    tipping = {}
    for k in KS:
        ser = [(r["cap"], r["total_pct"]) for r in rows if r["k"] == k]
        half = next((c for c, v in ser if v < base_total * 0.5), None)
        tenth = next((c for c, v in ser if v < base_total * 0.1), None)
        gone = next((c for c, v in ser if v <= 0), None)
        tipping[f"k{k:g}"] = {"half": half, "tenth": tenth, "zero": gone}

    # ---- 参与率随年份演化(1000 万档, 温和冲击, 同主口径) ----
    by_year = {}
    sel = [t for t in trades]
    nav = 1.0
    per_year: dict[str, list[float]] = {}
    cap_ref, k_ref, max_p_ref = 1e7, 15.0, 0.05
    for t, amt in zip(sel, amounts):
        pos = float(t.get("pos_frac") or 0.0)
        r = float(t["ret_pct"]) / 100.0
        if pos <= 0 or nav <= 0:
            continue
        wealth = nav * cap_ref
        want = wealth * pos
        limit = max_p_ref * amt if amt > 0 else want
        invested = min(want, limit)
        p = (invested / amt * 100.0) if amt > 0 else 0.0
        eff = (invested / wealth) if wealth > 0 else 0.0
        y = str(t.get("signal_date") or "")[:4]
        per_year.setdefault(y, []).append(p)
        imp = 2.0 * k_ref * math.sqrt(max(p, 0.0))
        nav *= (1.0 + eff * (r - imp / 10000.0))
    for y, v in sorted(per_year.items()):
        by_year[y] = {"n": len(v), "p_mean_pct": round(float(np.mean(v)), 3),
                      "p_max_pct": round(float(np.max(v)), 2)}

    out = {
        "meta": {
            "run_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "days": args.days, "years": round(years, 2),
            "start": res["meta"]["start"], "end": res["meta"]["end"],
            "n_trades": len(trades),
            "baseline_total_pct": base_total,
            "baseline_cagr_pct": m["年化收益率%"],
            "baseline_mdd_pct": m["最大回撤%"],
            "baseline_calmar": m["卡玛比率"],
            "recon_total_pct": round(recon_total, 1),
            "recon_err_pp": round(recon_err, 1),
            "k_unit": "单边 bp = k * sqrt(参与率%)",
            "max_p": 0.05,
            "max_p_note": "单笔最多成交当日成交额的 5%, 超出按部分成交(平方根模型在 p>10% 无实证支撑)",
            "open_share": OPEN_SHARE,
            "run_sec": round(time.time() - t0, 1),
        },
        "liquidity": lig,
        "rows": rows,
        "tipping": tipping,
        "by_year_1e7": by_year,
        "notes": [
            "amount 为 T+1 全天成交额, 建仓仅占开盘数分钟 -> 本表为乐观上界, 真实容量约 1/5~1/10",
            "k=0 行是重建校验: recon_total_pct 应等于 baseline_total_pct, 偏差来自逐笔串行复利对并行持仓的近似",
            "capped_share_pct = 仓位被 5% 成交额上限压缩的笔数占比; eff_pos_pct = 平均实际仓位",
            "zero_at 非空表示该规模下净值在某笔归零, 后续交易不再计入",
        ],
    }

    dst = ROOT / args.out
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n→ {dst}")
    print(f"流动性: 中位成交额 {lig['amount_median_yi']} 亿 / P10 {lig['amount_p10_yi']} 亿")
    print(f"重建校验: {recon_total:.1f}% vs 基线 {base_total:.1f}% (偏差 {recon_err:+.1f}pp)")
    print(f"{'资金':>10} {'k':>3} {'净值x':>9} {'总收益%':>11} {'硬买%':>11} {'年化%':>7} {'回撤%':>7} "
          f"{'均参与%':>8} {'P95%':>7} {'冲击bp':>7} {'实仓%':>6} {'压缩%':>6}")
    for r in rows:
        print(f"{r['cap_wan']:>9.0f}万 {r['k']:>3.0f} {r['nav_x']:>9.3f} {r['total_pct']:>11.1f} "
              f"{r['total_pct_hardbuy']:>11.1f} {r['cagr_pct']:>7.2f} {r['mdd_pct']:>7.2f} "
              f"{r['p_mean_pct']:>8.4f} {r['p95_pct']:>7.3f} {r['imp_mean_bp']:>7.1f} "
              f"{r['eff_pos_pct']:>6.1f} {r['capped_share_pct']:>6.1f}")
    print("拐点:", tipping)


if __name__ == "__main__":
    main()
