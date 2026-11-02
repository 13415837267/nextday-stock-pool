# -*- coding: utf-8 -*-
"""向前台账(paper tape): 从今天起冻结每日信号, 用未来验证, 同时实测执行假设。

为什么要有这个文件
------------------
第一百二十九轮把容量算完后的结论是: 参数/特征/日历/持有期/分散度/容量, 能测的都测完了,
回测再往下挖只会挖到噪声。唯一还能提供**新信息**的是"向前看" —— README 里那句
"真正的 OOS 只有一个: 从今天起冻结每日信号, 用未来 6~12 个月检验"。

同时它解决第二个问题: 第一百二十九轮的容量表依赖 k=15/30 两个经验系数, 因为回测只有
日频数据、无法知道"开盘价到底是不是日内低位"。台账每笔都记 open_rank 与 VWAP 溢价,
积累 30~60 笔就能用真实分布替换掉这两个系数。

铁律
----
1. **只增不改**: 已写入的记录永不修改, 只发生状态流转 planned→holding→closed/void。
   这是"冻结信号"的全部意义 —— 允许事后修正的台账不是 OOS。
2. 幂等: 同一 signal_date 重复运行不会重复追加(每日有 5 个重试槽)。
3. 只读输出: 不碰 config.yaml, 不碰 latest.json, 只维护 output/paper.json。

状态机
------
planned  信号已记录, 等待 T+1 开盘
holding  已按 T+1 开盘价成交, 等待 T+2 收盘
closed   已按 T+2 收盘价结算(net 收益已扣佣金/印花税/滑点)
void     未成交(停牌/一字涨停/风控未触发/评分不达标), 附原因
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config                      # noqa: E402
from src.data_api import ZZClient                       # noqa: E402
from src.backtest import fetch_panel                    # noqa: E402
from src.selector import split_by_date                  # noqa: E402

TAPE = ROOT / "output" / "paper.json"
LATEST = ROOT / "output" / "latest.json"
LOOKBACK_DAYS = 45
EPS = 1e-6


def _d(x: str) -> str:
    """2026-11-02 -> 20261102"""
    return str(x or "").replace("-", "").strip()


def load_tape() -> dict:
    if TAPE.exists():
        try:
            return json.loads(TAPE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"meta": {"created": datetime.now().strftime("%Y-%m-%d"), "version": 1},
            "records": [], "stats": {}}


def _row_of(by_date: dict, date: str, code: str):
    frame = by_date.get(date)
    if frame is None or not len(frame):
        return None
    col = frame["ts_code"].to_numpy()
    hit = np.flatnonzero(col == code)
    if not len(hit):
        return None
    return frame.iloc[int(hit[0])]


def _exec_profile(row, pos: float) -> dict:
    """买入日的执行画像: 开盘价在日内区间的位置、VWAP 溢价、参与率。"""
    o, h, l, c = (float(row["open"]), float(row["high"]),
                  float(row["low"]), float(row["close"]))
    amt, vol = float(row["amount"]), float(row["vol"])
    rng = h - l
    vwap = (amt / vol) if vol > 0 else None
    return {
        "open": round(o, 3), "high": round(h, 3), "low": round(l, 3), "close": round(c, 3),
        # 0 = 当日最低价开盘, 1 = 当日最高价开盘
        "open_rank": round((o - l) / rng, 4) if rng > EPS else None,
        # >0 表示全天均价高于开盘价 —— 开盘买优于日内均匀买
        "vwap_prem_pct": round((vwap / o - 1) * 100, 3) if vwap and o else None,
        "amount_yi": round(amt / 1e8, 3),
        # 单笔投入 / 当日成交额, 按 100 万 / 1000 万 两个标尺各算一次
        "part_100w_pct": round(100e4 * pos / amt * 100, 4) if amt > 0 else None,
        "part_1000w_pct": round(1000e4 * pos / amt * 100, 4) if amt > 0 else None,
    }


def fill_buy(rec: dict, by_date: dict, ex: dict) -> None:
    row = _row_of(by_date, rec["buy_date"], rec["code"])
    if row is None:
        rec.update(state="void", note="买入日无行情(停牌/退市)")
        return
    if bool(row.get("is_paused", False)):
        rec.update(state="void", note="买入日停牌")
        return
    o, hl = float(row["open"]), float(row["high_limit"])
    if o >= hl - EPS:
        rec.update(state="void", note="一字涨停, 开盘即封板买不到")
        return
    pre = float(row["pre_close"])
    gap = (o / pre - 1) * 100 if pre > 0 else 0.0
    rec["gap_pct"] = round(gap, 3)

    # 风控日: 只有缺口够深才接(与 t1_decision 的 floor_pct 同口径)
    if not rec.get("regime_ok", True):
        floor = rec.get("floor_pct")
        if floor is None or gap > float(floor) + EPS:
            rec.update(state="void",
                       note=f"风控未触发(缺口 {gap:+.2f}% 未达 {floor}%), 按规则空仓")
            return

    if not rec.get("allow_buy", True):
        rec.update(state="void", note="信号日决策为不买")
        return

    pos = float(rec.get("plan_pos_pct") or 0.0) / 100.0
    if pos <= 0:
        rec.update(state="void", note="计划仓位为 0")
        return
    # 高开减仓: 缺口 >= 0.5% 且评分未达满仓档时压到 30%(与 execution.gap_above_tiers 同口径)
    for tier in (ex.get("gap_above_tiers") or []):
        try:
            if (gap >= float(tier.get("above", 99))
                    and float(rec.get("score") or 0) < float(tier.get("score_below", 0))):
                pos = min(pos, float(tier.get("mult", 1.0)))
                rec["gap_tier_note"] = (f"高开 {gap:.2f}%≥{tier.get('above')} → "
                                        f"仓位×{tier.get('mult')}")
                break
        except (TypeError, ValueError):
            continue

    rec.update(state="holding", pos_frac=round(pos, 4),
               buy_px=round(o, 3), exec=_exec_profile(row, pos),
               note="按 T+1 开盘价成交(回测同口径)")


def fill_sell(rec: dict, by_date: dict, ex: dict) -> None:
    row = _row_of(by_date, rec["sell_date"], rec["code"])
    if row is None:
        rec["note"] = (rec.get("note") or "") + " | 卖出日无行情, 待补"
        return
    c, ll = float(row["close"]), float(row["low_limit"])
    if c <= ll + EPS:
        # 跌停卖不出: 顺延(最多 3 日), 与 sell_limit_down_fallback 同向
        rec["defer"] = int(rec.get("defer") or 0) + 1
        if rec["defer"] > 3:
            rec.update(state="closed", sell_px=round(c, 3), note="连续跌停, 按第4日收盘强制结算")
        else:
            dates = sorted(by_date.keys())
            try:
                nxt = dates[dates.index(rec["sell_date"]) + 1]
            except (ValueError, IndexError):
                return
            rec["sell_date"] = nxt
            rec["note"] = f"卖出日跌停, 顺延至 {nxt}"
            return
    else:
        rec["sell_px"] = round(c, 3)

    comm = float(ex["commission_rate"])
    stamp = float(ex["stamp_tax"])
    slip = float(ex["slippage"])
    buy_px = float(rec["buy_px"])
    sell_px = float(rec["sell_px"])
    gross = sell_px / buy_px - 1
    net = (sell_px * (1 - slip) * (1 - stamp) * (1 - comm)) / \
          (buy_px * (1 + slip) * (1 + comm)) - 1
    pos = float(rec.get("pos_frac") or 1.0)
    rec.update(state="closed",
               ret_gross_pct=round(gross * 100, 3),
               ret_net_pct=round(net * 100, 3),
               weighted_pct=round(net * 100 * pos, 3),
               sell_exec={"close": round(c, 3),
                          "close_rank": round((c - float(row["low"])) /
                                              max(float(row["high"]) - float(row["low"]), EPS), 4)})


def append_signal(tape: dict, latest: dict) -> bool:
    sig = _d(latest.get("signal_date"))
    if not sig:
        return False
    if any(r["signal_date"] == sig for r in tape["records"]):
        return False
    plan = latest.get("plan") or {}
    picks = latest.get("picks") or []
    dec = latest.get("t1_decision") or {}
    if not picks:
        return False
    top = picks[0]
    rec = {
        "signal_date": sig,
        "buy_date": _d(plan.get("buy_date")),
        "sell_date": _d(plan.get("sell_date")),
        "code": top.get("code"), "name": top.get("name"),
        "rank": 1, "score": top.get("score"),
        "strategy": top.get("strategy"),
        "regime_ok": bool((latest.get("regime") or {}).get("ok", True)),
        "allow_buy": bool(dec.get("allow_buy", True)),
        "plan_pos_pct": dec.get("position_pct"),
        "floor_pct": dec.get("floor_pct"),
        "condition": (dec.get("buy_note") or "")[:160],
        "fallbacks": [{"code": p.get("code"), "name": p.get("name"), "score": p.get("score")}
                      for p in picks[1:4]],
        "state": "planned",
        "recorded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "buy_px": None, "sell_px": None, "gap_pct": None,
        "ret_gross_pct": None, "ret_net_pct": None, "weighted_pct": None,
        "exec": None, "note": "",
    }
    tape["records"].append(rec)
    return True


def compute_stats(tape: dict) -> dict:
    rs = tape["records"]
    closed = [r for r in rs if r.get("state") == "closed" and r.get("ret_net_pct") is not None]
    void = [r for r in rs if r.get("state") == "void"]
    holding = [r for r in rs if r.get("state") == "holding"]
    planned = [r for r in rs if r.get("state") == "planned"]
    out = {
        "n_signal": len(rs), "n_closed": len(closed),
        "n_holding": len(holding), "n_planned": len(planned), "n_void": len(void),
        "void_reasons": {},
    }
    for r in void:
        k = (r.get("note") or "其它").split("(")[0][:20]
        out["void_reasons"][k] = out["void_reasons"].get(k, 0) + 1
    if closed:
        rets = np.asarray([r["ret_net_pct"] for r in closed], dtype=float)
        wts = np.asarray([r.get("weighted_pct") or 0.0 for r in closed], dtype=float)
        nav = 1.0
        navs = [1.0]
        for w in wts:
            nav *= (1 + w / 100.0)
            navs.append(nav)
        peak = np.maximum.accumulate(np.asarray(navs))
        out.update({
            "win_pct": round(float((rets > 0).mean() * 100), 1),
            "avg_ret_pct": round(float(rets.mean()), 3),
            "med_ret_pct": round(float(np.median(rets)), 3),
            "sum_weighted_pct": round(float(wts.sum()), 2),
            "nav_x": round(nav, 4),
            "mdd_pct": round(float((np.asarray(navs) / peak - 1).min() * 100), 2),
            "best_pct": round(float(rets.max()), 2),
            "worst_pct": round(float(rets.min()), 2),
        })
        ex_rows = [r["exec"] for r in closed if r.get("exec")]
        if ex_rows:
            def _avg(k):
                v = [x[k] for x in ex_rows if x.get(k) is not None]
                return round(float(np.mean(v)), 4) if v else None
            out["exec"] = {
                "open_rank_mean": _avg("open_rank"),
                "open_rank_median": round(float(np.median(
                    [x["open_rank"] for x in ex_rows if x.get("open_rank") is not None])), 4)
                if any(x.get("open_rank") is not None for x in ex_rows) else None,
                "vwap_prem_mean_pct": _avg("vwap_prem_pct"),
                "vwap_prem_pos_share": round(float(np.mean(
                    [1.0 if (x.get("vwap_prem_pct") or 0) > 0 else 0.0 for x in ex_rows])) * 100, 1),
                "amount_yi_median": round(float(np.median(
                    [x["amount_yi"] for x in ex_rows])), 3) if ex_rows else None,
                "part_1000w_mean_pct": _avg("part_1000w_pct"),
            }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true", help="只打印观察摘要, 不写文件")
    args = ap.parse_args()

    cfg = load_config()
    ex = cfg["execution"]
    tape = load_tape()

    # 1) 追加当日信号(幂等)
    added = False
    if LATEST.exists():
        latest = json.loads(LATEST.read_text(encoding="utf-8"))
        added = append_signal(tape, latest)

    # 2) 取最近行情面板, 回填状态
    client = ZZClient(cfg)
    days = client.trade_days(days=LOOKBACK_DAYS)
    panel = fetch_panel(client, days, float32=False)
    by_date = split_by_date(panel)

    for rec in tape["records"]:
        try:
            if rec.get("state") == "planned" and rec.get("buy_date") in by_date:
                fill_buy(rec, by_date, ex)
            if rec.get("state") == "holding" and rec.get("sell_date") in by_date:
                fill_sell(rec, by_date, ex)
        except Exception as e:      # 单条记录失败不应中断整份台账
            rec["note"] = (rec.get("note") or "") + f" | 回填异常: {type(e).__name__}: {e}"

    tape["stats"] = compute_stats(tape)
    tape["meta"]["updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tape["meta"]["n"] = len(tape["records"])

    if args.report:
        print(json.dumps(tape["stats"], ensure_ascii=False, indent=1))
        return

    TAPE.parent.mkdir(parents=True, exist_ok=True)
    TAPE.write_text(json.dumps(tape, ensure_ascii=False, indent=1), encoding="utf-8")

    st = tape["stats"]
    print(f"台账已更新: 信号 {st['n_signal']} / 已结算 {st['n_closed']} / "
          f"持仓 {st['n_holding']} / 待买 {st['n_planned']} / 未成交 {st['n_void']}"
          + (" [本次新增]" if added else " [无新增]"))
    if st.get("n_closed"):
        print(f"  已结算: 胜率 {st['win_pct']}% / 均笔 {st['avg_ret_pct']:+.3f}% / "
              f"累计 {st['sum_weighted_pct']:+.2f}% / 净值 {st['nav_x']:.4f} / 回撤 {st['mdd_pct']:.2f}%")
        if st.get("exec"):
            e = st["exec"]
            print(f"  执行画像: 开盘价分位 {e.get('open_rank_mean')} (中位 {e.get('open_rank_median')}) / "
                  f"VWAP 溢价 {e.get('vwap_prem_mean_pct')}% / "
                  f"均价高于开盘的笔数占比 {e.get('vwap_prem_pos_share')}%")
    if st.get("void_reasons"):
        print("  未成交原因:", st["void_reasons"])


if __name__ == "__main__":
    main()
