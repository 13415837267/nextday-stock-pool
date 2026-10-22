# -*- coding: utf-8 -*-
"""每日运行入口(供 GitHub Actions 每日收盘后调用)。

流程:
  1. 确定最新有完整数据的交易日 T
  2. 抓取近 regime.warmup_days 个交易日全市场日线 -> 计算指标与大盘风控
  3. 在 T 日横截面上执行选股, 得到"次日(T+1)计划候选池"
  4. 对最终候选逐票核查上市天数(过滤次新股)
  5. 结算 T-2 日发布池的实盘战绩(T-1开盘买入, T开盘卖出)
  6. 输出 output/latest.json / output/history.json, 并在 CHANGELOG.md 追加记录
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from .config import ROOT, load_config
from .data_api import ZZClient
from .selector import build_indicators, regime_series, select_stocks, split_by_date
from .backtest import fetch_panel

DISCLAIMER = "本候选池为量化模型自动生成, 仅供研究参考, 不构成任何投资建议。股市有风险, 入市需谨慎。"


def _fmt(d: str) -> str:
    return f"{d[:4]}-{d[4:6]}-{d[6:]}"


def _position_note(ex: dict) -> str:
    """按执行配置生成仓位说明(与回测口径一致)。"""
    frac = float(ex.get("alloc_frac_pct", 5.0))
    maxn = int(ex.get("max_buy_per_day", 0))
    gap = float(ex.get("buy_gap_below", 0) or 0)
    if maxn == 1:
        pos = f"若开仓: 买入评分第1名 1 只, 约 {frac:g}% 总资产仓位"
    elif maxn > 1:
        pos = f"若开仓: 按评分买入最多 {maxn} 只, 每笔约 {frac:g}% 总资产"
    else:
        pos = f"若开仓: 达标票每笔约 {frac:g}% 总资产"
    if gap:
        pos += f"; 仅买开盘缺口 <= {gap:g}% 的深低开票"
    pos += "; T+1买入, T+2开盘卖出(遵守A股T+1制度, 无当日卖出)"
    return pos


def main():
    cfg = load_config()
    client = ZZClient(cfg)
    t0 = time.time()

    # ---- 1. 最新完整交易日 ----
    cal = client.trade_days(days=cfg["regime"].get("warmup_days", 30) + 45)
    signal_date = None
    day_df = None
    for d in reversed(cal[-8:]):
        df = client.market_day(d)
        if len(df) >= 3000:
            signal_date = d
            day_df = df
            break
    if signal_date is None:
        raise RuntimeError("近8个交易日均无完整行情数据, 终止")
    pos = cal.index(signal_date)
    print(f"信号日(最新完整交易日): {_fmt(signal_date)}")

    # ---- 2. 近期面板与指标 ----
    warm = max(cfg["regime"].get("warmup_days", 30), 25)
    panel_dates = cal[max(0, pos - warm + 1): pos + 1]
    panel = fetch_panel(client, panel_dates)
    panel = build_indicators(panel)
    by_date = split_by_date(panel)
    reg = regime_series(panel, cfg)
    names_df = client.stock_names()
    names = dict(zip(names_df["ts_code"], names_df["name"]))

    # ---- 3. 选股(多取缓冲用于次新核查) ----
    # 注意: 日常面板仅warmup天, 无法用bars判断上市天数, 此处关闭该硬过滤,
    #       改由第4步逐票拉取扩展历史核查(回测路径不受影响)。
    sel_cfg = cfg["selection"]
    cfg_sel = json.loads(json.dumps(cfg))
    cfg_sel["universe"]["listed_min_bars"] = 0
    # 发布池数量: 人工审核用, 默认取评分前 display_top_n 只(与回测的 top_n 资金部署口径解耦)
    disp_n = int(sel_cfg.get("display_top_n") or sel_cfg["top_n"])
    ranked, stats = select_stocks(signal_date, by_date[signal_date], cfg_sel,
                                  names, top_n=sel_cfg["top_n"] + 8)
    # ---- 4. 次新股/上市天数核查 ----
    start_far = (pd.Timestamp(signal_date) - pd.Timedelta(days=540)).strftime("%Y%m%d")
    verified = []
    for p in ranked:
        if len(verified) >= disp_n:
            break
        hist = client.stock_history(p["code"], start_far, signal_date)
        bars = len(hist)
        if bars < cfg["universe"]["listed_min_bars"]:
            print(f"  剔除次新/数据不足: {p['code']} {p['name']} ({bars}根K线)")
            continue
        verified.append(p)

    # ---- 5. 结算 T-2 发布池 ----
    out_dir = ROOT / "output"
    out_dir.mkdir(exist_ok=True)
    hist_path = out_dir / "history.json"
    if hist_path.exists():
        _data = json.loads(hist_path.read_text(encoding="utf-8"))
        history = _data.get("records", []) if isinstance(_data, dict) else _data
    else:
        history = []
    ex = cfg["execution"]

    # 买入日风控状态(与回测口径一致): 超限参与日用专属承接线与参与系数(含渐近恢复)
    _rm = cfg["regime"]
    max_block = int(_rm.get("max_block_days", 0) or 0)
    reenter = float(_rm.get("block_reenter_frac", 0) or 0)
    rf_ramp = bool(_rm.get("rf_ramp", False))
    day_frac_map: dict[str, float] = {}
    _consec = 0
    for _d in reg.index:
        if bool(reg.loc[_d, "trade_ok"]):
            _consec = 0
            day_frac_map[_d] = 1.0
        else:
            _consec += 1
            if _consec <= max_block:
                day_frac_map[_d] = 0.0
            elif rf_ramp:
                day_frac_map[_d] = min(1.0, reenter + 0.05 * (_consec - max_block))
            else:
                day_frac_map[_d] = reenter

    def row_of(d: str, code: str):
        sub = by_date.get(d)
        if sub is None:
            return None
        r = sub[sub["ts_code"] == code]
        return r.iloc[0] if len(r) else None

    for rec in history:
        if rec.get("status") == "已结算":
            continue
        pdate = rec["publish_date"]
        if pdate not in cal:
            continue
        ip = cal.index(pdate)
        buy_d = cal[ip + 1] if ip + 1 < len(cal) else None
        sell_d = cal[ip + 2] if ip + 2 < len(cal) else None
        if not sell_d or sell_d > signal_date:
            continue  # 卖出日数据尚未产生
        # 买入日仓位系数与承接线(与回测口径一致)
        bfrac = day_frac_map.get(buy_d, 1.0) if buy_d else 0.0
        day_ok = bfrac > 0
        floor = ex.get("buy_gap_below", 0)
        if not day_ok:
            floor = ex.get("block_gap_below", floor)
        rets, skipped, chain = [], [], []
        executed = None
        maxn = int(ex.get("max_buy_per_day", 0))
        for rank_i, p in enumerate(rec.get("picks", []), 1):
            if maxn > 0 and len(rets) >= maxn:
                break  # 与回测口径一致: 竞价过滤后仅按评分买入前 maxn 只
            rb = row_of(buy_d, p["code"]) if buy_d else None
            rs = row_of(sell_d, p["code"])
            rp = row_of(pdate, p["code"])
            gap = (rb["open_adj"] / rp["close_adj"] - 1) * 100 \
                if (rb is not None and rp is not None and rp["close_adj"] > 0) else 999
            reason = None
            if not day_ok:
                reason = "大盘风控空仓"
            elif rb is None or rb["is_paused"] == 1 or rb["vol"] <= 0:
                reason = "停牌/无成交"
            elif rb["open"] >= rb["high_limit"] - 1e-9:
                reason = "开盘一字涨停"
            elif gap >= ex["skip_open_gain"]:
                reason = f"高开{gap:.1f}%放弃"
            elif ex.get("skip_open_loss", 0) and gap <= -abs(ex["skip_open_loss"]):
                reason = f"低开{gap:.1f}%过深"
            elif floor and gap > floor:
                reason = f"缺口{gap:.1f}%未达承接线{floor}%"
            if reason:
                chain.append(f"第{rank_i}名 {p.get('name', p['code'])}: {reason}")
                skipped.append(p["code"])
                continue
            if rs is None or rs["is_paused"] == 1 or rs["vol"] <= 0:
                continue  # 卖出日停牌: 留待后续结算
            gap_fee = (1 - ex["slippage"]) * (1 - ex["commission_rate"] - ex["stamp_tax"]) \
                      / ((1 + ex["slippage"]) * (1 + ex["commission_rate"]))
            r = (rs["open_adj"] / rb["open_adj"]) * gap_fee - 1
            rets.append(float(r))
            executed = {
                "rank": rank_i, "code": p["code"], "name": p.get("name", p["code"]),
                "buy_date": _fmt(buy_d), "buy_px": round(float(rb["open"]), 2),
                "sell_date": _fmt(sell_d), "sell_px": round(float(rs["open"]), 2),
                "ret_pct": round(float(r) * 100, 2),
                "position_pct": round(bfrac * ex.get("alloc_frac_pct", 100.0), 1),
                "gap_pct": round(gap, 2),
                "chain": chain,
            }
        if rets:
            rec["realized"] = {
                "buy_date": _fmt(buy_d), "sell_date": _fmt(sell_d),
                "n_traded": len(rets), "n_skipped": len(skipped),
                "avg_ret_pct": round(sum(rets) / len(rets) * 100, 2),
                "win": sum(1 for r in rets if r > 0),
            }
            rec["executed"] = executed
        elif executed is None and (chain or not day_ok):
            rec["skip_chain"] = chain if day_ok else ["大盘风控空仓"]
        rec["status"] = "已结算" if rets else "无成交"
        rec["skipped_codes"] = skipped

    # ---- 6. 写出候选池与历史(空仓时也展示观察池, 并明确标注风控状态) ----
    reg_row = reg.loc[signal_date] if signal_date in reg.index else None
    # 最大连续空仓上限: 统计截至信号日的连续风控天数, 超限后按小仓位参与
    _rm = cfg["regime"]
    max_block, reenter = int(_rm.get("max_block_days", 0) or 0), float(_rm.get("block_reenter_frac", 0) or 0)
    consec = 0
    if reg_row is not None and not bool(reg_row["trade_ok"]):
        for d_i in reversed(list(reg.index[:reg.index.get_loc(signal_date) + 1])):
            if bool(reg.loc[d_i, "trade_ok"]):
                break
            consec += 1
    over_limit = max_block > 0 and reenter > 0 and consec > max_block
    regime_info = {
        "ok": bool(reg_row["trade_ok"]) if reg_row is not None else True,
        "breadth": round(float(reg_row["breadth"]), 3) if reg_row is not None else None,
        "mkt_above_ma": bool(reg_row["mkt_idx"] >= reg_row["mkt_ma"]) if reg_row is not None else None,
        "block_days": consec,
        "max_block_days": max_block,
        "reenter_frac": reenter if over_limit else None,
        "note": "宽度/市场指数达标, 可正常开仓" if (reg_row is not None and reg_row["trade_ok"])
                else (f"大盘风控连续第{consec}日, 超过{max_block}日上限: 次日按{reenter:.0%}仓位参与候选池"
                      f"(执行承接线{cfg['execution'].get('block_gap_below', cfg['execution'].get('buy_gap_below'))}%)"
                      if over_limit else
                      (f"大盘风控触发(连续第{consec}日): 次日建议空仓观望" if reg_row is not None else "")),
    }
    buy_d = cal[pos + 1] if pos + 1 < len(cal) else None
    sell_d = cal[pos + 2] if pos + 2 < len(cal) else None

    latest = {
        "update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "signal_date": _fmt(signal_date),
        "plan": {
            "buy_date": _fmt(buy_d) if buy_d else "",
            "buy_point": "开盘集合竞价(09:15-09:25挂单, 09:25成交)",
            "sell_date": _fmt(sell_d) if sell_d else "",
            "sell_point": "次日开盘集合竞价卖出",
            "position": _position_note(ex),
        },
        "strategy": "次日开盘买入-再次日开盘卖出({})".format(
            "趋势内缩量回调+次日深低开承接" if cfg["selection"].get("mode", "momentum") == "dip" else "隔夜动量(备用)"
        ),
        "regime": regime_info,
        "stats": {k: v for k, v in stats.items()},
        "picks": verified,
        "disclaimer": DISCLAIMER,
    }
    (out_dir / "latest.json").write_text(
        json.dumps(latest, ensure_ascii=False, indent=1), encoding="utf-8")

    new_rec = {
        "publish_date": _fmt(signal_date),
        "buy_date": _fmt(buy_d) if buy_d else "",
        "sell_date": _fmt(sell_d) if sell_d else "",
        "regime_ok": regime_info["ok"],
        "picks": [{"code": p["code"], "name": p["name"], "close": p["close"],
                   "pct": p["pct"], "reasons": p["reasons"]} for p in verified],
        "status": "待结算",
    }
    history = [r for r in history if r["publish_date"] != new_rec["publish_date"]]
    history.insert(0, new_rec)
    keep = cfg["pages"].get("history_keep_days", 60)
    # 汇总仅统计风控通过(实际可开仓)的已结算池; 空仓期观察池不计入胜率
    done = [r for r in history
            if r.get("status") == "已结算" and r.get("realized") and r.get("regime_ok", True) is not False]
    win_n = sum(1 for r in done if r["realized"]["avg_ret_pct"] > 0)
    summary = {
        "n_done": len(done),
        "win_rate": round(win_n / len(done) * 100, 1) if done else None,
        "avg_ret": round(sum(r["realized"]["avg_ret_pct"] for r in done) / len(done), 2) if done else None,
    }
    hist_path.write_text(json.dumps({"summary": summary, "records": history[:keep]},
                                    ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- 7. CHANGELOG 追加运行记录 ----
    cl = ROOT / "CHANGELOG.md"
    if regime_info["ok"]:
        regime_tag = "风控通过"
    elif regime_info.get("reenter_frac") is not None:
        regime_tag = f"超限小仓参与({regime_info['reenter_frac']:.0%})"
    else:
        regime_tag = "触发空仓"
    line = (f"- {datetime.now().strftime('%Y-%m-%d %H:%M')} 自动运行: 信号日 {_fmt(signal_date)}, "
            f"候选 {len(verified)} 只, {regime_tag}, "
            f"耗时 {time.time()-t0:.0f}s\n")
    text = cl.read_text(encoding="utf-8") if cl.exists() else "# 更新日志\n\n## 自动运行记录\n"
    marker = "## 自动运行记录\n"
    if marker in text:
        text = text.replace(marker, marker + line, 1)
    else:
        text += "\n## 自动运行记录\n" + line
    cl.write_text(text, encoding="utf-8")

    print(f"候选池({len(verified)}只): " + ", ".join(f"{p['code']} {p['name']}" for p in verified))
    print(f"风控: {regime_info['note']}")
    print(f"完成, 耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
