# -*- coding: utf-8 -*-
"""T+1 竞价确认: 每交易日 9:26 用实时行情自动判定今日买入标的。
数据源: 腾讯 qt.gtimg.cn(主) / 新浪 hq.sinajs.cn(备)。
输出 output/auction.json, 供网页「竞价确认」卡展示。"""
import json
import os
import re
import sys
import time
import urllib.request
from datetime import datetime, timezone, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
CST = timezone(timedelta(hours=8))

from src.config import load_config   # noqa: E402  (需先注入ROOT)
from src.daily_run import _gap_tier_mult   # noqa: E402  (第九十七轮: 高开降仓同口径)


def limit_ratio(code):
    """A股涨跌停幅度: 创业板300/301与科创板688/689为20%, 其余主板10%(北交所未参与)。"""
    return 0.20 if str(code).startswith(("300", "301", "688", "689")) else 0.10


def http_get(url, headers=None, timeout=15):
    req = urllib.request.Request(url, headers=headers or {"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def fetch_quotes(codes):
    """codes: ['000504.SZ', ...] -> {code: {name, prev_close, open, last}}
    last=现价(集合竞价后=开盘价, 盘中=实时价, 收盘后=收盘价)。
    主源(腾讯)失败/空返回自动重试3次(9:26窗口网络抖动常见), 仍缺的交由备源(新浪)补齐。"""
    quotes = {}
    # 主源: 腾讯(自动重试) 字段: f[1]名称 f[3]现价 f[4]昨收 f[5]今开
    q = codes and "q=" + ",".join(("sh" if c.endswith(".SH") else "sz") + c[:6] for c in codes)
    for attempt in range(1, 4):
        try:
            raw = http_get(f"https://qt.gtimg.cn/{q}").decode("gbk", "ignore")
            for m in re.finditer(r'v_(?:sh|sz)(\d{6})="([^"]*)"', raw):
                f = m.group(2).split("~")
                if len(f) > 5 and float(f[4] or 0) > 0:
                    quotes[f"{m.group(1)}.{'SH' if m.group(1).startswith(('6', '9')) else 'SZ'}"] = {
                        "name": f[1], "prev_close": float(f[4]), "open": float(f[5]),
                        "last": float(f[3]) if len(f) > 3 and float(f[3] or 0) > 0 else None}
            if quotes or not codes:
                break
            print(f"腾讯源第{attempt}次返回空", flush=True)
        except Exception as e:
            print(f"腾讯源第{attempt}次失败: {e}", flush=True)
        if attempt < 3:
            time.sleep(2)
    # 备源: 新浪(补缺) 字段: f[0]名称 f[1]今开 f[2]昨收 f[3]现价
    miss = [c for c in codes if c not in quotes]
    if miss:
        try:
            q = "list=" + ",".join(("sh" if c.endswith(".SH") else "sz") + c[:6] for c in miss)
            raw = http_get(f"https://hq.sinajs.cn/{q}",
                           headers={"Referer": "https://finance.sina.com.cn"}).decode("gbk", "ignore")
            for m in re.finditer(r'hq_str_(?:sh|sz)(\d{6})="([^"]*)"', raw):
                f = m.group(2).split(",")
                if len(f) > 2 and float(f[2] or 0) > 0:
                    quotes[f"{m.group(1)}.{'SH' if m.group(1).startswith(('6', '9')) else 'SZ'}"] = {
                        "name": f[0], "open": float(f[1]), "prev_close": float(f[2]),
                        "last": float(f[3]) if len(f) > 3 and float(f[3] or 0) > 0 else None}
        except Exception as e:
            print(f"新浪源失败: {e}", flush=True)
    return quotes


def main():
    cfg = load_config()
    ex = cfg["execution"]
    latest_path = os.path.join(ROOT, "output", "latest.json")
    out_path = os.path.join(ROOT, "output", "auction.json")
    latest = json.load(open(latest_path, encoding="utf-8"))
    plan = latest.get("plan", {})
    td = latest.get("t1_decision", {})
    buy_date = (plan.get("buy_date") or "").replace("-", "")
    today = datetime.now(CST).strftime("%Y%m%d")
    now = datetime.now(CST).strftime("%Y-%m-%d %H:%M:%S")

    result = {
        "confirm_time": now,
        "buy_date": plan.get("buy_date", ""),
        "signal_date": latest.get("signal_date", ""),
        "for_today": buy_date == today,
        "allow_buy": bool(td.get("allow_buy")),
        "floor_pct": td.get("floor_pct"),
        "position_pct": td.get("position_pct"),
        "data_ok": True,
        "action": "skip", "name": "", "code": "", "rank": None,
        "open_px": None, "gap_pct": None, "chain": [],
        "gap_tier_note": None,   # 第九十七轮: 高开减仓说明(仅当触发时写入)
    }
    cands = []
    if buy_date != today:
        result["note"] = f"今日({today})非计划买入日(计划买入日{plan.get('buy_date')}), 无需竞价确认"
    elif not result["allow_buy"]:
        result["note"] = td.get("buy_note") or "风控判定: 今日不买入"
    else:
        cands = [(s.get("rank"), s.get("code"), s.get("name")) for s in td.get("steps", []) if s.get("rank")]
        codes = [c for _, c, _ in cands if c]
        quotes = fetch_quotes(codes)
        floor = result["floor_pct"]
        skip_gain = float(ex.get("skip_open_gain", 9.0) or 9.0)   # 与config执行口径同步, 防硬编码漂移
        for rank, code, name in cands:
            q = quotes.get(code)
            if q is None or q.get("open", 0) <= 0:
                result["chain"].append(f"第{rank}名 {name}: 无行情/停牌 → 顺延")
                continue
            prev, op = q["prev_close"], q["open"]
            gap = (op / prev - 1) * 100 if prev else 0.0
            ratio = limit_ratio(code)
            limit_up = round(prev * (1 + ratio), 2)
            if op >= limit_up - 0.001:
                result["chain"].append(f"第{rank}名 {name}: 一字涨停({gap:+.2f}%) → 顺延")
                continue
            if gap >= skip_gain:
                result["chain"].append(f"第{rank}名 {name}: 高开{gap:.2f}%≥{skip_gain:g}% → 顺延")
                continue
            if floor is not None and gap > floor:
                result["chain"].append(f"第{rank}名 {name}: 缺口{gap:.2f}%未达承接线{floor}% → 顺延")
                continue
            result.update({"action": "buy", "name": name, "code": code, "rank": rank,
                           "open_px": op, "gap_pct": round(gap, 2)})
            break
        if result["action"] == "skip" and not any("顺延" in c for c in result["chain"]) and cands:
            result["note"] = "候选数据异常, 请人工复核"
        elif result["action"] == "skip":
            result["note"] = ("全部候选未达标 → 今日空仓" if cands
                              else "空池日(无合格候选) → 今日空仓")
        if result["action"] == "buy":
            # 第九十七轮: 高开降仓 —— 盘前仓位只含评分档, 9:25 拿到真实竞价缺口后再乘缺口档,
            # 与回测 do_buy / daily_run._settle_holding 同口径(评分 ≥1.6 的高开票不减)。
            _sc = next((s.get("score") for s in td.get("steps", [])
                        if s.get("code") == result["code"]), None)
            _gm, _gt = _gap_tier_mult(result["gap_pct"], ex, _sc)
            if _gm < 1.0:
                result["position_pct"] = round(float(result["position_pct"] or 0) * _gm, 1)
                result["gap_tier_note"] = _gt
            result["note"] = (f"确认买入第{result['rank']}名 {name}({code}), "
                              f"竞价缺口{result['gap_pct']:.2f}%, 开盘价{result['open_px']}, "
                              f"仓位{result['position_pct']}%"
                              + (f"（{_gt}，高开减仓）" if _gm < 1.0 else ""))

        # ---- 行情全失败保护(2026-10-24新增): 双源均无数据时绝不误报"今日空仓" ----
        # 退出码2 → 工作流失败 → 自动建Issue告警; 不写入误导性auction.json
        if cands and not quotes:
            print("⚠ 行情源(腾讯+新浪)全部失败, 竞价确认不可用 —— 请人工复核, 切勿按空仓处理", flush=True)
            sys.exit(2)

    # ---- 卖出侧确认(2026-10-24新增): 「持仓中」记录的卖出日==今日 → 生成卖出执行卡 ----
    # 输出实际竞价开盘价/预计实现收益(含双边费用口径与回测一致)/开盘跌停→收盘卖出提示
    try:
        hist_path = os.path.join(ROOT, "output", "history.json")
        sells = []
        if os.path.exists(hist_path):
            hist = json.load(open(hist_path, encoding="utf-8"))
            sell_recs = []
            for rec in hist.get("records", []):
                e = rec.get("executed") or {}
                if e.get("holding") and str(e.get("sell_date", "")).replace("-", "") == today:
                    sell_recs.append(e)
            if sell_recs:
                q = fetch_quotes([e["code"] for e in sell_recs])
                fee = (1 - ex.get("slippage", 0)) \
                    * (1 - ex.get("commission_rate", 0) - ex.get("stamp_tax", 0)) \
                    / ((1 + ex.get("slippage", 0)) * (1 + ex.get("commission_rate", 0)))
                for e in sell_recs:
                    cur = q.get(e["code"], {})
                    op, prev = cur.get("open") or 0, cur.get("prev_close") or 0
                    item = {"code": e["code"], "name": e.get("name", ""),
                            "buy_date": e.get("buy_date"), "buy_px": e.get("buy_px"),
                            "sell_date": e.get("sell_date"), "open_px": op or None,
                            "est_ret_pct": None, "limit_down": False}
                    if op and e.get("buy_px"):
                        item["est_ret_pct"] = round((op / e["buy_px"] * fee - 1) * 100, 2)
                        if prev:
                            ratio = limit_ratio(e["code"])
                            item["limit_down"] = op <= round(prev * (1 - ratio), 2) + 0.001
                    sells.append(item)
        if sells:
            result["sell_today"] = sells
            print("卖出确认:", [(s["name"], s.get("est_ret_pct"), s["limit_down"]) for s in sells], flush=True)
    except Exception as e:
        print(f"卖出确认失败(不影响买入确认): {e}", flush=True)

    # ---- 持仓实时浮盈: 扫描 history 中「持仓中」记录, 拉现价 ----
    try:
        hist_path = os.path.join(ROOT, "output", "history.json")
        positions = []
        if os.path.exists(hist_path):
            hist = json.load(open(hist_path, encoding="utf-8"))
            hold_codes, hold_recs = [], []
            for rec in hist.get("records", []):
                e = rec.get("executed") or {}
                if e.get("holding") and e.get("code"):
                    hold_codes.append(e["code"])
                    hold_recs.append(e)
            if hold_codes:
                q = fetch_quotes(hold_codes)
                for e in hold_recs:
                    cur = q.get(e["code"], {})
                    # 现价优先级: last(9:26=开盘价/盘中=实时价/盘后=收盘价) -> open -> prev_close
                    px = cur.get("last") or cur.get("open") or cur.get("prev_close")
                    fl = round((px / e["buy_px"] - 1) * 100, 2) if (px and e.get("buy_px")) else None
                    positions.append({"code": e["code"], "name": e.get("name", ""),
                                      "buy_px": e.get("buy_px"), "buy_date": e.get("buy_date"),
                                      "cur_px": px, "float_pct": fl})
        if positions:
            result["positions"] = positions
            result["positions_time"] = now
            print("持仓浮盈:", [(p["name"], p["float_pct"]) for p in positions])
    except Exception as e:
        print(f"持仓浮盈计算失败(不影响确认): {e}", flush=True)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(result, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("竞价确认:", result["note"])
    for c in result["chain"]:
        print("  ", c)
    print(f"SAVED {out_path}")


if __name__ == "__main__":
    main()
