# -*- coding: utf-8 -*-
"""每日运行入口(供 GitHub Actions 每日收盘后调用)。

流程:
  0. 交易日历闸门: 非交易日(周末/法定假日)或同信号日已处理则跳过
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
from datetime import datetime, timedelta
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


def _refresh_auction_positions(out_dir: Path, day_df: pd.DataFrame, now_str: str) -> None:
    """盘后将 auction.json 的持仓浮盈刷新为收盘价口径。

    9:26竞价确认写入的positions为实时现价(9:26时=开盘价, 即买入价, 浮盈恒0);
    收盘后应以T日收盘价刷新, 避免网页盘后一直显示买入开盘价。
    auction.json 不存在时跳过(不伪造竞价确认数据)。
    """
    ap = out_dir / "auction.json"
    if not ap.exists():
        return
    try:
        a = json.loads(ap.read_text(encoding="utf-8"))
        hist = json.loads((out_dir / "history.json").read_text(encoding="utf-8"))
        holds = [(r.get("executed") or {}) for r in hist.get("records", [])
                 if (r.get("executed") or {}).get("holding") and (r.get("executed") or {}).get("code")]
        close_of = day_df.set_index("ts_code")["close"].to_dict() if day_df is not None and len(day_df) else {}
        positions = []
        for e in holds:
            close = close_of.get(e["code"])
            ok = close is not None and pd.notna(close) and close > 0 and e.get("buy_px")
            positions.append({"code": e["code"], "name": e.get("name", ""),
                              "buy_px": e.get("buy_px"), "buy_date": e.get("buy_date"),
                              "cur_px": round(float(close), 2) if ok else None,
                              "float_pct": round((close / e["buy_px"] - 1) * 100, 2) if ok else None})
        a["positions"] = positions
        a["positions_time"] = now_str
        ap.write_text(json.dumps(a, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"持仓浮盈已刷新为收盘价口径({len(positions)}只, 截至 {now_str})")
    except Exception as e:  # noqa: BLE001 刷新失败不影响主流程
        print(f"auction.json持仓浮盈刷新失败(不影响主流程): {e}")


def main():
    cfg = load_config()
    client = ZZClient(cfg)
    t0 = time.time()

    # ---- 0. 交易日历闸门: 非交易日(周末/法定假日)直接空转, 防周期轮询在非交易日重复计算 ----
    try:
        from zoneinfo import ZoneInfo
        _now = datetime.now(ZoneInfo("Asia/Shanghai"))
    except Exception:  # noqa: BLE001 时区库不可用时退回系统本地时间
        _now = datetime.now()
    _today = _now.strftime("%Y%m%d")
    if not client.is_trading_day(_today):
        print(f"今日{_fmt(_today)}非A股交易日(周末或法定假日), 按交易日历跳过")
        return
    print(f"今日{_fmt(_today)}为A股交易日, 开始运行")

    # ---- 1. 最新完整交易日 ----
    # 历史: trade_days只含已过去交易日; 未来: 确定信号日后用区间查询补齐——
    # 否则最新发布日的T+1/T+2越界为None → 持仓sell_date为空,
    # 竞价确认卖出侧永不触发(2026-10-24修复)
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
    try:
        _fut = client.trade_calendar(_today, (_now + timedelta(days=20)).strftime("%Y%m%d"))
        cal = sorted(set(cal) | set(_fut))
    except Exception as _e:  # noqa: BLE001 未来日历失败退回旧行为(仅最新一期卖出日为空)
        print(f"⚠ 未来交易日历获取失败(最新持仓卖出日可能为空): {_e}")
    pos = cal.index(signal_date)
    print(f"信号日(最新完整交易日): {_fmt(signal_date)}")

    # ---- 1.4 同信号日幂等跳过: 今日收盘信号已处理过则不重复计算(防重复触发) ----
    # 注: 仅当解析出的信号日就是今天(收盘数据已完整)且与latest.json一致时跳过;
    #     信号日<今天(盘前运行/数据延迟)时不跳过, 保留既有补算与恢复路径
    _latest_path = ROOT / "output" / "latest.json"
    _prev_sig = ""
    if _latest_path.exists():
        try:
            _prev_sig = str((json.loads(_latest_path.read_text(encoding="utf-8")) or {}).get("signal_date") or "")
        except Exception:  # noqa: BLE001 latest.json损坏时视为未处理, 正常重算
            _prev_sig = ""
    if _prev_sig == _fmt(signal_date) and signal_date >= _today:
        print(f"信号日{_fmt(signal_date)}今日已处理过(latest.json一致), 幂等跳过")
        return

    # ---- 1.5 数据质量校验(异常检测, 防脏数据进信号) ----
    anomalies = []
    if len(day_df) < 3000:
        anomalies.append(f"全市场行数仅{len(day_df)}(<3000)")
    # 新股上市首日: pre_close=0, pct为脏值(如+7199%)属正常现象, 单独列示不计为异常
    ipo_rows = day_df[day_df["pre_close"] <= 0] if "pre_close" in day_df.columns else day_df.iloc[:0]
    if len(ipo_rows):
        _codes = ", ".join(ipo_rows["ts_code"].head(3))
        anomalies.append(f"新股上市首日{len(ipo_rows)}只({_codes}等, pct脏值已在指标层置NaN, 正常)")
    # 真实异常: 收盘价超出自身涨跌停价(按各板自身限制判断, 北交所±30%/科创板±20%不再误报;
    # 限价=0为无涨跌幅限制的次新股, 不参与判定)
    lim = day_df[(day_df["pre_close"] > 0) & (day_df["high_limit"] > 0) & (day_df["low_limit"] > 0)]
    out_lim = lim[(lim["close"] > lim["high_limit"] + 0.005) | (lim["close"] < lim["low_limit"] - 0.005)]
    if len(out_lim):
        _codes = ", ".join(out_lim["ts_code"].head(5))
        anomalies.append(f"真实异常: 收盘超出自身涨跌停价共{len(out_lim)}条({_codes})")
    # 疑似除权: pct与收盘/昨收隐含收益矛盾但收盘在限价内(除权日quote_rate对未复权昨收, 属正常)
    mism = lim[lim["close"] > 0]
    _implied = (mism["close"] / mism["pre_close"] - 1) * 100
    mism = mism[((mism["pct"] - _implied).abs() > 1.5)
                & (mism["close"] <= mism["high_limit"] + 0.005)
                & (mism["close"] >= mism["low_limit"] - 0.005)]
    if len(mism):
        _codes = ", ".join(mism["ts_code"].head(5))
        anomalies.append(f"疑似除权: pct与收盘昨收不一致共{len(mism)}条({_codes}等, 除权日属正常)")
    nan_close = int(day_df["close"].isna().sum())
    if nan_close:
        anomalies.append(f"收盘价缺失{nan_close}条")
    dup = int(day_df["ts_code"].duplicated().sum())
    if dup:
        anomalies.append(f"代码重复{dup}条")
    if anomalies:
        print("⚠ 数据质量异常: " + "; ".join(anomalies))
    else:
        print("数据质量校验通过")

    # ---- 2. 近期面板与指标 ----
    warm = max(cfg["regime"].get("warmup_days", 30), 25)
    panel_dates = cal[max(0, pos - warm + 1): pos + 1]
    panel = fetch_panel(client, panel_dates)
    if not len(panel):
        raise RuntimeError("面板抓取为空, 指标无法计算, 终止(请检查数据源)")
    got = set(panel["trade_date"].astype(str))
    miss = [d for d in panel_dates if str(d) not in got]
    if miss:
        print(f"⚠ 面板缺日 {len(miss)}/{len(panel_dates)} 天: {', '.join(miss[:5])}"
              + (" ..." if len(miss) > 5 else "")
              + " —— 指标与风控基于实际可得日期计算, 请核查数据源")
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
    day_block_map: dict[str, bool] = {}   # 该日是否处于风控期(trade_ok=False), 供承接线口径判定
    _consec = 0
    for _d in reg.index:
        if bool(reg.loc[_d, "trade_ok"]):
            _consec = 0
            day_frac_map[_d] = 1.0
        else:
            day_block_map[_d] = True
            _consec += 1
            if _consec <= max_block:
                day_frac_map[_d] = 0.0
            elif rf_ramp:
                day_frac_map[_d] = min(1.0, reenter + 0.05 * (_consec - max_block))
            else:
                day_frac_map[_d] = reenter

    def row_of(d: str, code: str):
        # 日期归一化: 调用方可能传入带横杠(history)或无横杠(cal)格式
        sub = by_date.get(str(d).replace("-", ""))
        if sub is None:
            return None
        r = sub[sub["ts_code"] == code]
        return r.iloc[0] if len(r) else None

    def _settle_holding(rec, buy_d, sell_d, signal_date, cal, day_frac_map, row_of, ex, cfg):
        """买入日已发生但卖出日未到: 按当日真实竞价判定成交, 返回持仓中明细(无卖出价)。"""
        bfrac = day_frac_map.get(buy_d, 1.0) if buy_d else 0.0
        day_ok = bfrac > 0
        floor = ex.get("buy_gap_below", 0)
        if day_ok and day_block_map.get(buy_d, False):
            # 超限小仓参与日(风控期但允许小仓): 用专属承接线, 与回测口径一致
            floor = ex.get("block_gap_below", floor)
        maxn = int(ex.get("max_buy_per_day", 0))
        chain = []
        for rank_i, p in enumerate(rec.get("picks", []), 1):
            rb = row_of(buy_d, p["code"]) if buy_d else None
            rp = row_of(rec["publish_date"], p["code"])
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
                continue
            return {
                "rank": rank_i, "code": p["code"], "name": p.get("name", p["code"]),
                "buy_date": _fmt(buy_d), "buy_px": round(float(rb["open"]), 2),
                "sell_date": _fmt(sell_d) if sell_d else "", "sell_px": None,
                "ret_pct": None,
                "position_pct": round(bfrac * ex.get("alloc_frac_pct", 100.0), 1),
                "gap_pct": round(gap, 2),
                "chain": chain,
                "holding": True,
            }
        rec["skip_chain"] = chain   # 总是覆盖: 顺延链为空即清空, 防历史运行残留与executed矛盾
        return None

    for rec in history:
        if rec.get("status") == "已结算":
            continue
        pdate = rec["publish_date"]
        # 日期格式归一化: history 的 publish_date 带横杠, cal/by_date/reg 为无横杠
        # (此前格式不一致导致结算循环静默跳过所有记录 —— 2026-10-23 发现并修复)
        pdate_n = str(pdate).replace("-", "")
        if pdate_n not in cal:
            continue
        ip = cal.index(pdate_n)
        buy_d = cal[ip + 1] if ip + 1 < len(cal) else None
        sell_d = cal[ip + 2] if ip + 2 < len(cal) else None
        if not sell_d or sell_d > signal_date:
            # 卖出日未到: 若买入日数据已产生, 记录"持仓中"(买入事实当日可见)
            if buy_d and buy_d <= signal_date and rec.get("picks"):
                _holding = _settle_holding(rec, buy_d, sell_d, signal_date, cal, day_frac_map, row_of, ex, cfg)
                if _holding:
                    rec["executed"] = _holding
                    rec["status"] = "持仓中"
                    rec["skipped_codes"] = []
                    rec.pop("skip_chain", None)   # 清除历史运行残留的旧顺延链(与executed矛盾)
                else:
                    rec.pop("executed", None)     # 判定未成交: 清除可能的旧持仓记录
            continue
        # 买入日仓位系数与承接线(与回测口径一致)
        bfrac = day_frac_map.get(buy_d, 1.0) if buy_d else 0.0
        day_ok = bfrac > 0
        floor = ex.get("buy_gap_below", 0)
        if day_ok and day_block_map.get(buy_d, False):
            # 超限小仓参与日(风控期但允许小仓): 用专属承接线, 与回测口径一致
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
            rec.pop("skip_chain", None)   # 顺延链已在executed.chain内, 清除顶层残留
        elif executed is None and (chain or not day_ok):
            rec["skip_chain"] = chain if day_ok else ["大盘风控空仓"]
        else:
            rec.pop("skip_chain", None)
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
    if buy_d is None:   # 信号日=日历最后一天: 推算下一交易日(跳过周末; 法定节假日顺延, 误差可接受)
        _bd = pd.bdate_range(start=_fmt(signal_date), periods=3)
        buy_d = _bd[1].strftime("%Y%m%d")
        sell_d = _bd[2].strftime("%Y%m%d")
        print(f"交易日历无未来日, 推算买入日{buy_d}/卖出日{sell_d}(遇节假日顺延)")

    # ---- T+1 执行判定卡(每晚生成, 次日照卡执行) ----
    regime_ok_now = bool(reg_row["trade_ok"]) if reg_row is not None else True
    over_limit_now = over_limit
    panic_day = False
    if reg_row is not None and cfg["regime"].get("vol_max"):
        _vol20 = float(reg["mkt_pct"].rolling(20).std().loc[signal_date]) if signal_date in reg.index else 0.0
        _m20 = float((reg["mkt_idx"].loc[signal_date] / reg["mkt_idx"].shift(20).loc[signal_date] - 1) * 100) \
            if signal_date in reg.index else 0.0
        panic_day = (_vol20 > cfg["regime"]["vol_max"]) and (_m20 < cfg["regime"].get("vol_mkt_ret_min", 0.0))
    if panic_day:
        allow_buy = False
        buy_note = "恐慌型波动日(高波动+市场20日下跌): 全额空仓, 不参与任何买入"
    elif regime_ok_now:
        allow_buy = True
        buy_note = "风控通过: 可正常开仓(承接线见下)"
    elif over_limit_now:
        allow_buy = True
        buy_note = f"超限参与日(连续风控第{consec}日): 按{reenter:.0%}仓位参与, 执行更严承接线"
    else:
        allow_buy = False
        buy_note = f"大盘风控中(连续第{consec}日): 空仓观望"

    floor_now = ex.get("buy_gap_below", 0)
    if allow_buy and not regime_ok_now:
        floor_now = ex.get("block_gap_below", floor_now)
    steps = []
    for i, p in enumerate(verified[:5], 1):
        steps.append({
            "rank": i, "code": p["code"], "name": p["name"], "score": p.get("score"),
            "close": p.get("close"), "pct": p.get("pct"),
            "rule": f"9:25竞价缺口 ≤ {floor_now}% 且 < 9% 且未停牌未一字 → 买入全仓",
            "fallback": f"不达标 → 顺延第{i+1}名" if i < min(5, len(verified)) else "",
        })
    if not verified:
        steps.append({"rank": 0, "code": "", "name": "（今日无合格候选）", "rule": "明日空仓, 无需操作", "fallback": ""})
    t1_decision = {
        "allow_buy": allow_buy,
        "buy_note": buy_note,
        "panic_day": panic_day,
        "floor_pct": floor_now if allow_buy else None,
        "position_pct": round((reenter if (over_limit_now and not regime_ok_now) else 1.0) * ex.get("alloc_frac_pct", 100.0), 1),
        "steps": steps,
        "sell_plan": "T+2 开盘集合竞价卖出(开盘跌停则当日收盘卖出)",
        "empty_funnel": ({"全市场": stats.get("universe"), "流动性/次新": stats.get("liquidity"),
                          "回调形态": stats.get("price_action"), "趋势条件": stats.get("trend_ok"),
                          "评分入池": stats.get("scored")} if not verified else None),
    }

    def _audit_vs_auction(out_dir, today_fmt, history):
        """两道判定一致性对账: 9:26竞价确认(auction.json, 腾讯实时) vs 晚间panel结算(权威)。
        两道判定应一致; 不一致说明数据源异常, 返回match=False供告警。"""
        ap = out_dir / "auction.json"
        if not ap.exists():
            return {"checked": False, "note": "无竞价确认记录"}
        try:
            a = json.loads(ap.read_text(encoding="utf-8"))
        except Exception:
            return {"checked": False, "note": "auction.json 解析失败"}
        if a.get("buy_date", "").replace("-", "") != today_fmt.replace("-", ""):
            return {"checked": False, "note": f"auction非今日数据({a.get('buy_date')})"}
        # 今日实际结算结果(executed: 已结算或持仓中)
        actual = None
        for rec in history:
            e = rec.get("executed") or {}
            if str(e.get("buy_date", "")).replace("-", "") == today_fmt.replace("-", "") and not e.get("holding"):
                actual = e  # 已结算(卖出也完成了)
                break
            if str(e.get("buy_date", "")).replace("-", "") == today_fmt.replace("-", ""):
                actual = e
                break
        a_rank, a_code = a.get("rank"), a.get("code")
        if a.get("action") == "buy":
            if actual and str(actual.get("code")) == str(a_code):
                return {"checked": True, "match": True,
                        "note": f"两道判定一致: 买入{a.get('name')} 第{a_rank}名 (竞价缺口{a.get('gap_pct')}%)"}
            if actual:
                return {"checked": True, "match": False,
                        "note": f"不一致! 竞价确认买{a.get('name')}, panel结算为{actual.get('name')} —— 请检查数据源"}
            return {"checked": True, "match": False,
                    "note": "不一致! 竞价确认有买入但panel未结算到成交 —— 请检查数据源"}
        # auction 认为空仓
        if actual:
            return {"checked": True, "match": False,
                    "note": f"不一致! 竞价确认空仓但panel结算有成交({actual.get('name')}) —— 请检查数据源"}
        return {"checked": True, "match": True, "note": "两道判定一致: 今日空仓"}

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
        "t1_decision": t1_decision,
        "audit": _audit_vs_auction(out_dir, _fmt(signal_date), history),
        "data_quality": {"anomalies": anomalies, "rows": int(len(day_df))},
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

    # ---- 6.5 盘后刷新 auction.json 持仓浮盈为收盘价口径 ----
    _refresh_auction_positions(out_dir, day_df, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

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
