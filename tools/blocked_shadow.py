# -*- coding: utf-8 -*-
"""十年"风控空仓日"虚拟对照重放(tools/blocked_shadow.py, 第五十轮)。

问题: 风控层拦截买入日后, 若"完全放行风控闸(其余执行闸不变)", 虚拟top1的
盈亏分布如何? —— 为第45轮防线审计"风控era后贡献+1183pp"提供第三只眼睛:
  - 第45轮 no_regime: 整体关闭风控重跑, 复利口径(结论受连败链/资金复用影响);
  - 第49轮 实盘 observe: 空仓日对照账本, 前瞻逐笔积累(样本从零开始);
  - 本脚本: 同口径在十年历史上全量重放, 单日独立均值口径 —— 直接回答
    "平均每个被拦日, 风控是躲过飞刀(-)还是误杀大肉(+)"。

口径(与 daily_run 第四十九轮 blocked 对照逐条一致):
  - 样本集: 买入日 day_frac==0 的全额空仓交易日(超限小仓参与日 frac>0 不在内);
  - 信号日 = 买入日的前一交易日(T日收盘出池 -> T+1开盘买);
  - 放行 = 仅跳过风控闸, 五道非风控闸(停牌无成交/开盘一字涨停/高开放弃/
    低开过深/缺口未达承接线)逐条保留, 空仓日承接线用常规 buy_gap_below
    (daily_run同款: 仅day_ok时才切block_gap_below);
  - max_buy_per_day=1 -> 自top1向下顺延, 虚拟成交第一只过闸票;
  - 卖出日停牌/无成交 -> 不成交(不模拟停牌顺延, 与daily_run口径一致);
  - 虚拟收益 = 卖出日open_adj/买入日open_adj * 费用因子 - 1(gap_fee公式同款);
    注: 不含跌停收盘卖回退(do_sell口径), 因虚拟对照回答的是"买入端放行与否";
  - 只记录不参与决策, 输出不回写任何执行逻辑。

用法: python3 tools/blocked_shadow.py   (依赖本地panel缓存, 约5~10分钟)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config                     # noqa: E402
from src.backtest import prepare_context               # noqa: E402
from src.selector import apply_regime, select_stocks   # noqa: E402

ERA = "20240901"   # era切分锚点, 与第42~49轮一致(按buy_date)

CROSS = ("互验参照: 第45轮 no_regime(整体关风控重跑) era后 2366.2% vs BASE 3549.6%"
         "(关风控更差 -> 风控era后净贡献+1183pp, 复利口径); 本脚本为单日独立均值口径, "
         "两者方向应互洽")


def build_day_frac(reg, cfg) -> dict[str, float]:
    """复刻 run_backtest 的 day_frac 序列(与 daily_run.day_frac_map 同构)。"""
    _rm = cfg.get("regime") or {}
    max_block = int(_rm.get("max_block_days", 0) or 0)
    reenter = float(_rm.get("block_reenter_frac", 0.0) or 0.0)
    rf_ramp = bool(_rm.get("rf_ramp", False))
    day_frac: dict[str, float] = {}
    if max_block > 0 and reenter > 0 and len(reg):
        has_panic = "panic" in reg.columns
        consec = 0
        for d in reg.index:
            if bool(reg.loc[d, "trade_ok"]):
                consec = 0
                day_frac[d] = 1.0
            elif has_panic and bool(reg.loc[d, "panic"]):
                day_frac[d] = 0.0   # 恐慌型波动日: 全额空仓且不计入连续风控天数
            else:
                consec += 1
                if consec <= max_block:
                    day_frac[d] = 0.0
                elif rf_ramp:
                    day_frac[d] = min(1.0, reenter + 0.05 * (consec - max_block))
                else:
                    day_frac[d] = reenter
    return day_frac


def stats_of(rows: list[dict]) -> dict:
    """单日独立口径统计: 均值/胜率/等权和/分位数(非复利)。"""
    if not rows:
        return {"n": 0}
    rets = sorted(float(r["ret_pct"]) for r in rows)
    n = len(rets)
    q = lambda p: rets[min(n - 1, int(p * n))]          # noqa: E731
    return {
        "n": n,
        "avg_pct": round(sum(rets) / n, 3),
        "win_pct": round(100 * sum(r > 0 for r in rets) / n, 1),
        "sum_pct": round(sum(rets), 1),                 # 等权和(每个被拦日独立计1份)
        "median_pct": round(q(0.50), 2),
        "p25_pct": round(q(0.25), 2), "p75_pct": round(q(0.75), 2),
        "worst_pct": round(rets[0], 2), "best_pct": round(rets[-1], 2),
        "bench_avg_pct": round(sum(float(r["bench_pct"]) for r in rows) / n, 3),
    }


def main():
    cfg = load_config()
    days = int(os.environ.get("BLOCKED_SHADOW_DAYS", "2500"))   # 小窗口验证用环境变量
    t0 = time.time()
    print(f"[1/3] prepare_context({days}) ...")
    ctx = prepare_context(cfg, days)
    reg = apply_regime(ctx["reg_base"], cfg) if "reg_base" in ctx else ctx["reg"]
    day_frac = build_day_frac(reg, cfg)
    main_dates = ctx["main_dates"]
    by_date, per_code, names = ctx["by_date"], ctx["per_code"], ctx["names"]
    ex = cfg["execution"]
    comm, stamp, slip = ex["commission_rate"], ex["stamp_tax"], ex["slippage"]
    gap_fee = (1 - slip) * (1 - comm - stamp) / ((1 + slip) * (1 + comm))
    floor = ex.get("buy_gap_below", 0)   # 空仓日承接线=常规线(daily_run同款)
    maxn = int(ex.get("max_buy_per_day", 0))
    pos = {d: i for i, d in enumerate(main_dates)}
    n_zero = sum(1 for d in main_dates if day_frac.get(d, 1.0) <= 0)
    print(f"  全额空仓买入日 {n_zero} 个(共{len(main_dates)}日)")

    print("[2/3] 逐空仓日虚拟对照 ...")
    samples: list[dict] = []
    n_no_pick = n_unfilled = 0
    for d in main_dates[1:]:
        if day_frac.get(d, 1.0) > 0:
            continue
        i = pos[d]
        s = main_dates[i - 1]                            # 信号日=前一交易日
        sell_d = main_dates[i + 1] if i + 1 < len(main_dates) else None
        picks, _ = select_stocks(s, by_date[s], cfg, names)
        if not picks:
            n_no_pick += 1
            continue                                     # 池空: 无top1, 与实盘一致无对照样本
        bench = reg["mkt_idx"] if "mkt_idx" in reg.columns else None
        filled = 0
        for p in picks:
            if maxn > 0 and filled >= maxn:
                break
            sub = per_code.get(p["code"])
            rb = sub.loc[d] if sub is not None and d in sub.index else None
            rs = sub.loc[sell_d] if (sell_d and sub is not None and sell_d in sub.index) else None
            rp = sub.loc[s] if sub is not None and s in sub.index else None
            gap = (float(rb["open_adj"]) / float(rp["close_adj"]) - 1) * 100 \
                if (rb is not None and rp is not None and float(rp["close_adj"]) > 0) else 999.0
            # 五道非风控闸(daily_run._exec_gate同语义, 此处内联避免跨层import实盘模块)
            ok = rb is not None and float(rb["is_paused"]) != 1 and float(rb["vol"]) > 0
            ok = ok and not float(rb["open"]) >= float(rb["high_limit"]) - 1e-9
            ok = ok and gap < ex["skip_open_gain"]
            if ex.get("skip_open_loss", 0):
                ok = ok and gap > -abs(ex["skip_open_loss"])
            ok = ok and (not floor or gap <= floor)
            ok = ok and rs is not None and float(rs["is_paused"]) != 1 and float(rs["vol"]) > 0
            if not ok:
                continue
            vret = (float(rs["open_adj"]) / float(rb["open_adj"])) * gap_fee - 1
            bret = float(reg.loc[sell_d, "mkt_idx"]) / float(reg.loc[d, "mkt_idx"]) - 1 \
                if bench is not None and d in reg.index and sell_d in reg.index else None
            samples.append({
                "signal_date": s, "buy_date": d, "sell_date": sell_d,
                "code": p["code"], "name": p.get("name", ""),
                "ret_pct": round(vret * 100, 2),
                "bench_pct": round(bret * 100, 2) if bret is not None else None,
            })
            filled += 1
        if filled == 0:
            n_unfilled += 1                              # 有池但全部未过闸: 计数不产生样本

    print(f"[3/3] 统计汇总 ... (耗时 {time.time()-t0:.0f}s)")
    era_b = [r for r in samples if r["buy_date"] < ERA]
    era_a = [r for r in samples if r["buy_date"] >= ERA]
    years = sorted({r["buy_date"][:4] for r in samples})
    by_year = {y: stats_of([r for r in samples if r["buy_date"].startswith(y)]) for y in years}
    top5 = sorted(samples, key=lambda r: -r["ret_pct"])[:5]
    bot5 = sorted(samples, key=lambda r: r["ret_pct"])[:5]
    out = {
        "meta": {
            "run_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "window_days": days, "days": len(main_dates), "zero_days": n_zero,
            "no_pick_days": n_no_pick, "unfilled_days": n_unfilled,
            "口径": ("买入日day_frac==0全额空仓日; 信号日=前一交易日; 完全放行风控、"
                     "五道非风控闸不变(承接线=buy_gap_below); top1顺延虚拟成交; "
                     "T+2开盘卖; 与daily_run第49轮blocked对照同口径; 只记录不参与决策"),
        },
        "overall": stats_of(samples),
        "era": {"before_20240901": stats_of(era_b), "after_20240901": stats_of(era_a)},
        "by_year": by_year,
        "top5_win": top5, "top5_loss": bot5,
        "cross_check": {"no_regime_era_after_total_pct": 2366.2,
                        "base_era_after_total_pct": 3549.6, "note": CROSS},
        "samples": samples,
    }
    out_path = ROOT / "output" / "blocked_shadow.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    def _p(tag, st):
        if st.get("n"):
            print(f"  {tag}: n={st['n']} 均值{st['avg_pct']:+.2f}%/笔 胜率{st['win_pct']}% "
                  f"等权和{st['sum_pct']:+.1f}pp | 同期基准均值{st['bench_avg_pct']:+.2f}%/笔")
    print("==== 风控空仓日虚拟对照(完全放行, 其余闸不变) ====")
    _p("全窗口   ", out["overall"])
    _p("era前    ", out["era"]["before_20240901"])
    _p("era后    ", out["era"]["after_20240901"])
    for y in years:
        _p(f"  {y}  ", by_year[y])
    print(f"输出: {out_path}")


if __name__ == "__main__":
    main()
