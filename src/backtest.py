# -*- coding: utf-8 -*-
"""实盘口径复利回测引擎。

执行规则(与实盘完全一致):
  - T日收盘后按 selector 选出候选池(次日计划)
  - T+1 开盘竞价买入: 先按竞价缺口等条件筛出实际可买名单(保持评分排序),
    每笔买入占组合总资产 alloc_frac_pct%(与达标数量解耦, 防恐慌晨逆向选择),
    按评分排序每日最多买入 max_buy_per_day 只(0=不限)
    * 可选: 仅买开盘缺口 <= buy_gap_below% 的深低开票(dip 承接过滤; 0=不过滤)
    * T+1 停牌/无成交/开盘一字涨停/开盘涨幅 >= skip_open_gain% -> 放弃该票
    * 可选: 开盘跌幅 <= -skip_open_loss%(低开过深, 反转失败概率高) -> 放弃该票
  - T+2 开盘竞价卖出
    * T+2 停牌 -> 顺延至后续交易日开盘卖出(最多5日, 否则按最后收盘价)
    * T+2 开盘跌停且开启保守模式 -> 按当日收盘价卖出
  - 成本: 佣金万2.5(双边) + 印花税万5(卖出) + 滑点千1(双边)
  - 资金全部复利滚动, 卖出资金当日开盘即可再投入
  - 大盘风控触发时次一交易日空仓(不买入)

接口:
  - prepare_context(cfg, days): 构建行情面板/指标/风控上下文(耗时主体, 可复用)
  - run_backtest(cfg, days, ctx): 执行逐日模拟与绩效汇总; 传入 ctx 可跳过重复构建
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ROOT, load_config
from .data_api import ZZClient
from .selector import build_indicators, regime_series, select_stocks, split_by_date


# ----------------------------------------------------------------------
def fetch_panel(client: ZZClient, dates: list[str], log_every: int = 50) -> pd.DataFrame:
    """按交易日抓取全市场日线并合并为长表(带缓存)。空数据日(未来日期)自动跳过。"""
    frames = []
    t0 = time.time()
    for i, d in enumerate(dates, 1):
        df = client.market_day(d)
        if len(df):
            frames.append(df)
        if i % log_every == 0:
            print(f"  已抓取 {i}/{len(dates)} 个交易日, 累计 {time.time()-t0:.0f}s", flush=True)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def prepare_context(cfg: dict, days: int | None = None) -> dict:
    """构建回测所需上下文(行情面板/指标/风控/逐票索引), 供多次模拟复用。"""
    bt_cfg = cfg["backtest"]
    n_days = days or bt_cfg["trading_days"]
    warmup = bt_cfg["warmup_days"]

    client = ZZClient(cfg)
    print(f"[1/4] 获取交易日历与行情面板(主窗口{n_days}日+预热{warmup}日)...")
    all_days = client.trade_days(days=n_days + warmup + 5)
    if len(all_days) < n_days + warmup:
        raise RuntimeError(f"交易日不足: 需要{n_days + warmup}日, 仅获取{len(all_days)}日")
    panel_dates = all_days[-(n_days + warmup):]
    panel = fetch_panel(client, panel_dates)
    print(f"[2/4] 计算指标(共{len(panel):,}行)...")
    panel = build_indicators(panel)
    by_date = split_by_date(panel)
    reg = regime_series(panel, cfg)
    # 主窗口仅保留实际有行情的日期(剔除日历中的未来日/无数据日)
    main_dates = [d for d in panel_dates[warmup:] if d in by_date]

    per_code: dict[str, pd.DataFrame] = {}
    for code, sub in panel.groupby("ts_code", sort=False):
        per_code[code] = sub.set_index("trade_date").sort_index()

    names = {}
    try:
        sb = client.stock_names()
        names = dict(zip(sb["ts_code"], sb["name"]))
    except Exception:
        pass
    return {"main_dates": main_dates, "by_date": by_date, "reg": reg,
            "per_code": per_code, "names": names}


def run_backtest(cfg: dict, days: int | None = None, ctx: dict | None = None) -> dict:
    if ctx is None:
        ctx = prepare_context(cfg, days)
    main_dates, by_date = ctx["main_dates"], ctx["by_date"]
    reg, per_code, names = ctx["reg"], ctx["per_code"], ctx["names"]
    bt_cfg = cfg["backtest"]

    ex = cfg["execution"]
    comm, stamp, slip = ex["commission_rate"], ex["stamp_tax"], ex["slippage"]
    cash = float(bt_cfg["initial_capital"])
    curve: dict[str, float] = {}
    trades: list[dict] = []
    pending_buy: tuple[str, list[dict]] | None = None   # (buy_date, picks)
    holding: list[dict] = []                            # 未平仓持仓
    empty_days = 0
    fee_total = 0.0

    def do_buy(picks: list[dict], d: str):
        nonlocal cash, fee_total
        if not picks:
            return
        # 第一遍: 先按竞价条件筛出实际可买名单(保持评分排序, 与实盘一致)
        qualified = []
        for p in picks:
            sub = per_code.get(p["code"])
            row = sub.loc[d] if sub is not None and d in sub.index else None
            if row is None or row["is_paused"] == 1 or row["vol"] <= 0:
                trades.append(_trade(p, "skip", d, note="停牌/无成交"))
                continue
            if row["open"] >= row["high_limit"] - 1e-9:
                trades.append(_trade(p, "skip", d, note="开盘一字涨停"))
                continue
            sig_row = by_date[p["signal_date"]]
            prev_close = sig_row.loc[sig_row["ts_code"] == p["code"], "close_adj"]
            gap = row["open_adj"] / float(prev_close.iloc[0]) - 1 if len(prev_close) else 0
            if gap * 100 >= ex["skip_open_gain"]:
                trades.append(_trade(p, "skip", d, note=f"开盘高开{gap*100:.1f}%"))
                continue
            if ex.get("skip_open_loss", 0) and gap * 100 <= -abs(ex["skip_open_loss"]):
                trades.append(_trade(p, "skip", d, note=f"开盘低开{gap*100:.1f}%"))
                continue
            # dip 承接过滤: 仅买缺口 <= buy_gap_below% 的深低开票(0=不过滤)
            floor = ex.get("buy_gap_below", 0)
            if floor and gap * 100 > floor:
                trades.append(_trade(p, "skip", d, note=f"缺口{gap*100:.1f}%未达承接线{floor}%"))
                continue
            qualified.append({**p, "_row": row, "_gap": gap})
        if not qualified:
            return
        # 第二遍: 每笔固定占组合总资产 frac%(与达标数量解耦, 防恐慌晨逆向选择),
        # 按评分排序最多买 max_buy_per_day 只
        max_n = int(ex.get("max_buy_per_day", 0))
        if max_n > 0:
            qualified = qualified[:max_n]
        frac = ex.get("alloc_frac_pct", 5.0) / 100.0
        equity = cash + sum(
            h["units"] * float(per_code[h["code"]].loc[d]["open_adj"])
            for h in holding if d in per_code[h["code"]].index
        )
        for p in qualified:
            if cash <= 0:
                break
            row = p.pop("_row")
            alloc = min(frac * equity, cash)
            if alloc < frac * equity * 0.3:  # 剩余资金不足单笔最低规模(30%)则放弃
                break
            buy_eff = row["open_adj"] * (1 + slip)
            units = alloc / (buy_eff * (1 + comm))
            fee = alloc * comm
            fee_total += fee
            cash -= alloc
            holding.append({
                **p, "buy_date": d, "units": units, "buy_eff": buy_eff,
                "buy_px_raw": float(row["open"]),
            })
        # 未成交部分留在现金中, 复利滚动
        # T+1 日内止损(可选): 买入当日尾盘浮亏超阈值 -> 按当日收盘价卖出
        # (决策与成交均用收盘时点信息, 实盘对应14:50尾盘判断执行, 无前视)
        stop = ex.get("intraday_stop_loss", 0)
        if stop:
            kept = []
            for h in holding:
                if h["buy_date"] != d:
                    kept.append(h)
                    continue
                row2 = per_code[h["code"]].loc[d]
                close_eff = float(row2["close_adj"]) * (1 - slip)
                proceeds = h["units"] * close_eff * (1 - stamp) * (1 - comm)
                rr = proceeds / (h["units"] * h["buy_eff"] * (1 + comm)) - 1
                if rr * 100 <= -abs(stop):
                    fee = h["units"] * close_eff * (stamp + comm)
                    fee_total += fee
                    cash += proceeds
                    trades.append({
                        "signal_date": h["signal_date"], "code": h["code"], "name": h.get("name", ""),
                        "buy_date": d, "buy_px": h["buy_px_raw"],
                        "sell_date": d, "sell_px": float(row2["close"]),
                        "ret_pct": round(rr * 100, 2), "note": "T+1尾盘止损",
                    })
                else:
                    kept.append(h)
            holding[:] = kept

    def do_sell(d: str):
        nonlocal cash, fee_total
        remaining = []
        for h in holding:
            sub = per_code.get(h["code"])
            row = sub.loc[d] if sub is not None and d in sub.index else None
            sell_px_adj, sell_d, note = None, d, ""
            if row is not None and row["is_paused"] == 1:
                row = None
            if row is not None and row["vol"] <= 0:
                row = None
            if row is None:
                # 停牌顺延: 最多向后找5个交易日
                idx = sub.index
                if sub is not None and d in idx:
                    pos = idx.get_loc(d)
                    for j in range(1, 6):
                        if pos + j >= len(idx):
                            break
                        r2 = sub.iloc[pos + j]
                        if r2["is_paused"] == 0 and r2["vol"] > 0:
                            row = r2
                            sell_d, note = str(idx[pos + j]), "停牌顺延"
                            break
                if row is None:  # 退市等极端情况: 按最后已知收盘估算
                    if sub is not None and len(sub):
                        row = sub.iloc[-1]
                        sell_d, note = str(sub.index[-1]), "退市按末收盘"
                    else:
                        remaining.append(h)
                        continue
            open_adj, close_adj = float(row["open_adj"]), float(row["close_adj"])
            sell_px_adj = open_adj
            if ex.get("sell_limit_down_fallback", True) and float(row["open"]) <= float(row["low_limit"]) + 1e-9:
                sell_px_adj = close_adj
                note = (note + " 开盘跌停按收盘").strip()
            sell_eff = sell_px_adj * (1 - slip)
            proceeds = h["units"] * sell_eff * (1 - stamp) * (1 - comm)
            fee = h["units"] * sell_eff * (stamp + comm)
            fee_total += fee
            cash += proceeds
            r = proceeds / (h["units"] * h["buy_eff"] * (1 + comm)) - 1
            trades.append({
                "signal_date": h["signal_date"], "code": h["code"], "name": h.get("name", ""),
                "buy_date": h["buy_date"], "buy_px": h["buy_px_raw"],
                "sell_date": sell_d, "sell_px": float(row["open"]),
                "ret_pct": round(r * 100, 2), "note": note,
            })
        holding[:] = remaining

    print("[3/4] 逐日模拟交易...")
    t0 = time.time()
    for d in main_dates:
        # 开盘: 先卖出昨日到期持仓(T-2信号的池), 再买入(T-1信号的池)
        do_sell(d)
        if pending_buy is not None:
            _, picks = pending_buy
            do_buy(picks, d)
            pending_buy = None
        # 收盘估值
        total = cash + sum(h["units"] * float(per_code[h["code"]].loc[d]["close_adj"])
                           for h in holding if d in per_code[h["code"]].index)
        curve[d] = total
        # 收盘后: 决定明日计划
        reg_row = reg.loc[d] if d in reg.index else None
        if reg_row is None or not bool(reg_row["trade_ok"]):
            empty_days += 1
            pending_buy = (d, [])
        else:
            picks, _ = select_stocks(d, by_date[d], cfg, names)
            pending_buy = (d, [{**p, "signal_date": d} for p in picks])
        # 若持仓未在开盘卖出(停牌顺延中), 继续每日估值
    print(f"  模拟完成, 耗时 {time.time()-t0:.0f}s")

    print("[4/4] 汇总绩效...")
    nav = pd.Series(curve).sort_index()
    if nav.empty:
        raise RuntimeError("回测主窗口为空, 无净值数据(请检查交易日/预热配置)")
    ret = nav.pct_change().dropna()
    n = len(nav)
    total_ret = nav.iloc[-1] / nav.iloc[0] - 1
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (252 / max(n, 1)) - 1
    dd = nav / nav.cummax() - 1
    mdd = float(dd.min())
    sharpe = float(ret.mean() / ret.std() * np.sqrt(252)) if ret.std() else 0.0
    real = [t for t in trades if t["ret_pct"] is not None]
    wins = [t for t in real if t["ret_pct"] > 0]
    losses = [t for t in real if t["ret_pct"] <= 0]
    gross_win = sum(t["ret_pct"] for t in wins)
    gross_loss = -sum(t["ret_pct"] for t in losses)
    bench = reg.loc[reg.index.isin(main_dates), "mkt_idx"]
    bench_norm = (bench / bench.iloc[0] * float(bt_cfg["initial_capital"])) if len(bench) else bench

    result = {
        "meta": {
            "run_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "start": main_dates[0], "end": main_dates[-1], "days": n,
            "initial_capital": bt_cfg["initial_capital"],
            "rules": "T+1开盘买入/T+2开盘卖出/复利/佣金万2.5+印花税万5+滑点千1",
        },
        "metrics": {
            "总收益率%": round(total_ret * 100, 2),
            "年化收益率%": round(cagr * 100, 2),
            "最大回撤%": round(mdd * 100, 2),
            "卡玛比率": round(cagr / abs(mdd), 2) if mdd else None,
            "夏普比率": round(sharpe, 2),
            "交易次数": len(real),
            "胜率%": round(len(wins) / len(real) * 100, 2) if real else 0,
            "平均单笔收益%": round(np.mean([t["ret_pct"] for t in real]), 3) if real else 0,
            "盈亏比": round(gross_win / gross_loss, 2) if gross_loss else None,
            "空仓天数": empty_days,
            "跳过买入次数": sum(1 for t in trades if t["ret_pct"] is None),
            "总费用(元)": round(fee_total, 0),
            "同期市场等权基准%": round(float(bench_norm.iloc[-1] / bt_cfg["initial_capital"] - 1) * 100, 2) if len(bench_norm) else None,
        },
        "equity": [{"d": _fmt_date(d), "nav": round(float(v), 0)} for d, v in nav.items()],
        "benchmark": [{"d": _fmt_date(d), "nav": round(float(v), 0)} for d, v in bench_norm.items()] if len(bench_norm) else [],
        "trades_recent": trades[-60:],
        "monthly": _monthly(nav),
    }
    return result


def _trade(p: dict, kind: str, d: str, note: str = "") -> dict:
    return {"signal_date": p.get("signal_date", ""), "code": p["code"], "name": p.get("name", ""),
            "buy_date": d, "buy_px": None, "sell_date": None, "sell_px": None,
            "ret_pct": None, "note": note}


def _fmt_date(d: str) -> str:
    return f"{d[:4]}-{d[4:6]}-{d[6:]}"


def _monthly(nav: pd.Series) -> list[dict]:
    m = nav.copy()
    m.index = [str(i)[:6] for i in m.index]
    g = m.groupby(level=0).agg(["first", "last"])
    out = []
    prev_last = None
    for idx, row in g.iterrows():
        base = prev_last if prev_last else row["first"]
        out.append({"month": idx, "ret_pct": round((row["last"] / base - 1) * 100, 2)})
        prev_last = row["last"]
    return out


def save_result(result: dict, out: str | None = None):
    out_path = Path(out) if out else ROOT / "output" / "backtest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return out_path


def main():
    ap = argparse.ArgumentParser(description="A股次日候选池策略回测")
    ap.add_argument("--days", type=int, default=None, help="回测交易日数")
    ap.add_argument("--config", type=str, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    result = run_backtest(cfg, args.days)
    path = save_result(result)
    m = result["metrics"]
    print("\n========== 回测结果 ==========")
    print(f"区间: {result['meta']['start']} ~ {result['meta']['end']} ({result['meta']['days']}个交易日)")
    for k, v in m.items():
        print(f"  {k}: {v}")
    print(f"报告已写入: {path}")


if __name__ == "__main__":
    main()
