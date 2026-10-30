# -*- coding: utf-8 -*-
"""历史回放: 按当前实盘策略重放指定区间的选股与成交, 输出逐笔成交明细。

严格复用回测引擎(src/backtest.py::run_backtest), 不重写策略逻辑, 确保口径与实盘一致:
  T日收盘选股 -> T+1开盘竞价买入 -> T+2开盘卖出, 资金复利, 大盘风控空仓。

无未来函数保证:
  - 指标全部为 rolling/shift 向后看(build_indicators), 无 center/shift(-n);
  - 选股只传当日截面(by_date[signal_date]), 不含任何后续日期行;
  - 风控 regime 序列由 rolling 均线/宽度生成, 只依赖当日及之前;
  - 买卖价取对应交易日的真实开盘/收盘价, 不引用未来价格。

用法:
  python3 tools/replay.py --start 20260901 --end 20261031 --out output/replay.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ["BT_FULL_TRADES"] = "1"   # 回测引擎输出全部逐笔(默认只留最近200条)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.config import load_config  # noqa: E402
from src.selector import apply_regime, build_indicators, regime_base, split_by_date  # noqa: E402
from src.backtest import _LazyPerCode, _downcast_float, fetch_panel, run_backtest  # noqa: E402

PANEL_DIR = ROOT / "data" / "panel"


class LocalClient:
    """只读本地 data/panel 缓存的极简数据端, 回放不触网。"""

    def market_day(self, date: str, use_cache: bool = True) -> pd.DataFrame:
        p = PANEL_DIR / f"{date}.csv.gz"
        if not p.exists():
            return pd.DataFrame()
        return pd.read_csv(p, dtype={"ts_code": "str", "trade_date": "str"})


def build_ctx(cfg: dict, start: str, end: str, warmup: int) -> tuple[dict, list[str]]:
    dates = sorted(p.name[:8] for p in PANEL_DIR.glob("*.csv.gz"))
    targets = [d for d in dates if start <= d <= end]
    if not targets:
        raise RuntimeError(f"本地无 {start}~{end} 的面板数据")
    i0 = dates.index(targets[0])
    if i0 < warmup:
        raise RuntimeError(f"预热不足: 需要{warmup}日, 起点前仅{i0}日(请补下载更早面板)")
    # 主窗口比目标区间早 5 个交易日: 保证区间首日就有 pending_buy(来自前一交易日信号),
    # 否则首日无持仓可卖, 会漏掉区间第一天的成交
    main_start = max(0, i0 - 5)
    warmup_eff = min(warmup, main_start)   # 本地数据不足时压低预热(仍需 >= listed_min_bars)
    panel_dates = dates[max(0, main_start - warmup_eff):]
    main_dates = dates[main_start:]

    client = LocalClient()
    print(f"[1/3] 读取面板 {len(panel_dates)} 个交易日({panel_dates[0]}~{panel_dates[-1]})...")
    panel = fetch_panel(client, panel_dates, float32=True)
    if not len(panel):
        raise RuntimeError("面板为空")
    print(f"[2/3] 计算指标 {len(panel):,} 行...")
    panel = build_indicators(panel)
    panel = _downcast_float(panel)
    by_date = split_by_date(panel)
    reg_base = regime_base(panel, cfg)
    reg_base = reg_base.astype("float64")
    reg = apply_regime(reg_base, cfg)
    del panel

    names_path = ROOT / "data" / "names.json"
    names = json.loads(names_path.read_text(encoding="utf-8")) if names_path.exists() else {}

    ctx = {"main_dates": main_dates, "by_date": by_date, "reg": reg, "reg_base": reg_base,
           "per_code": _LazyPerCode(by_date), "names": names, "warmup_eff": warmup_eff}
    return ctx, panel_dates


def main() -> None:
    ap = argparse.ArgumentParser(description="按当前策略重放历史区间成交")
    ap.add_argument("--start", default="20260901")
    ap.add_argument("--end", default="20261031")
    ap.add_argument("--warmup", type=int, default=160, help="指标预热交易日数(需>=listed_min_bars)")
    ap.add_argument("--out", default="output/replay.json")
    args = ap.parse_args()

    cfg = load_config()
    ctx, panel_dates = build_ctx(cfg, args.start, args.end, args.warmup)
    print(f"[3/3] 逐日模拟 {len(ctx['main_dates'])} 个交易日...")
    result = run_backtest(cfg, ctx=ctx)

    trades = [t for t in result.get("trades_all", [])
              if t.get("ret_pct") is not None and t.get("buy_date", "") >= args.start]
    trades.sort(key=lambda t: (t["buy_date"], t.get("rank") or 0))

    curve = result.get("equity") or []

    # 区间绩效: 以区间首日之前最后一个净值为基准, 避免把预热段的收益算进来
    def _dash(s: str) -> str:
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"

    seg = [p for p in curve if p["d"] >= _dash(args.start)]
    base = None
    for p in curve:
        if p["d"] <= _dash(args.start):
            base = p["nav"]
    if base and seg:
        navs = [base] + [p["nav"] for p in seg]
        seg_ret = (navs[-1] / navs[0] - 1) * 100
        peak, mdd = navs[0], 0.0
        for v in navs:
            peak = max(peak, v)
            mdd = min(mdd, v / peak - 1)
        seg_perf = {"range_ret_pct": round(seg_ret, 2), "range_mdd_pct": round(mdd * 100, 2),
                    "base_date": _dash(args.start), "base_nav": base, "end_nav": navs[-1]}
    else:
        seg_perf = {}

    wins = [t for t in trades if t["ret_pct"] > 0]

    # 逐日决策明细: 说明"这天为什么没买"(风控空仓/无候选/缺口未达标)
    by_day = {}
    for t in result.get("trades_all", []):
        if t.get("buy_date"):
            by_day.setdefault(t["buy_date"], []).append(t)
    reg = ctx["reg"]
    # 仓位系数(与 backtest 同口径): 连续风控超过 max_block_days 后按 reenter_frac 小仓参与,
    # 逐日 rf_ramp 递增。0=完全空仓, 1=正常满仓, 0<x<1=超限参与
    _rm = cfg.get("regime") or {}
    _mb = int(_rm.get("max_block_days", 0) or 0)
    _re = float(_rm.get("block_reenter_frac", 0.0) or 0.0)
    _ramp = bool(_rm.get("rf_ramp", False))
    frac: dict[str, float] = {}
    _consec = 0
    for _d in reg.index:
        if bool(reg.loc[_d, "trade_ok"]):
            _consec, frac[_d] = 0, 1.0
        elif _mb > 0 and _re > 0:
            _consec += 1
            if _consec <= _mb:
                frac[_d] = 0.0
            elif _ramp:
                frac[_d] = min(1.0, _re + 0.05 * (_consec - _mb))
            else:
                frac[_d] = _re
        else:
            frac[_d] = 0.0

    daily = []
    for d in ctx["main_dates"]:
        if d < args.start:
            continue
        items = by_day.get(d, [])
        daily.append({
            "date": d,
            "trade_ok": bool(reg.loc[d, "trade_ok"]) if d in reg.index else True,
            "alloc": round(frac.get(d, 1.0), 2),
            "fills": [{"code": t["code"], "name": t.get("name", ""),
                       "ret_pct": t["ret_pct"]} for t in items if t.get("ret_pct") is not None],
            "skipped": [{"code": t["code"], "name": t.get("name", ""),
                         "note": t.get("note", "")} for t in items if t.get("ret_pct") is None],
        })

    payload = {
        "meta": {
            "start": args.start, "end": args.end,
            "panel_first": panel_dates[0], "panel_last": panel_dates[-1],
            "warmup_days": ctx.get("warmup_eff", args.warmup),
            "strategy": "T日收盘选股 → T+1开盘竞价买入 → T+2开盘卖出",
            "initial_capital": float(cfg["backtest"]["initial_capital"]),
            "note": "复用回测引擎, 指标仅用截至当日数据, 买卖用当日真实开盘价, 无未来函数",
        },
        "summary": {
            "n_trades": len(trades),
            "n_win": len(wins),
            "win_rate": round(len(wins) / len(trades) * 100, 1) if trades else None,
            "avg_ret_pct": round(sum(t["ret_pct"] for t in trades) / len(trades), 2) if trades else None,
            "sum_ret_pct": round(sum(t["ret_pct"] for t in trades), 2),
            "best": max((t["ret_pct"] for t in trades), default=None),
            "worst": min((t["ret_pct"] for t in trades), default=None),
            **seg_perf,
        },
        "trades": [{
            "signal_date": t["signal_date"], "buy_date": t["buy_date"], "sell_date": t["sell_date"],
            "code": t["code"], "name": t.get("name", ""), "rank": t.get("rank"),
            "pool": t.get("pool", "main"), "strategy": t.get("strategy", "dip"),
            "buy_px": round(float(t["buy_px"]), 2) if t.get("buy_px") is not None else None,
            "sell_px": round(float(t["sell_px"]), 2) if t.get("sell_px") is not None else None,
            "ret_pct": t["ret_pct"], "note": t.get("note", ""),
        } for t in trades],
        "curve": [p for p in curve if p["d"] >= _dash(args.start)],
        "base_nav": (base if base else None),
        "daily": daily,
        "monthly": [m for m in result.get("monthly", []) if m["month"] >= args.start[:6]],
        "perf": result.get("metrics", {}),
    }

    outp = ROOT / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"\n成交 {len(trades)} 笔, 胜率 {payload['summary']['win_rate']}%, "
          f"均值 {payload['summary']['avg_ret_pct']}% → {outp}")


if __name__ == "__main__":
    main()
