"""第二百一十七轮: 数据源能力实测 —— 哪个源真能喂进回测

为什么需要它
-----------
第二百一十六轮之后, 主策略在现有数据上已到近似上限(全史年化 75.10% / 近 6 年 111.87%)。
要继续往上, 唯一没验证过的方向是"引入维度外的数据": 封板时间、涨停原因、游资席位、分钟级。
但**"能拿到"和"能用"是两件事**: 一个接口只覆盖最近 3 年, 就无法支撑 14 年回测,
只能做样本外验证。所以本轮不猜文档, 直接打接口, 测四个事实:

  1) 最早能取到哪一天(历史深度) —— 决定能否进主回测
  2) 返回哪些字段(信息密度)     —— 决定能构造什么因子
  3) 单日行数量级               —— 决定回补 3400 个交易日的请求量/耗时
  4) 是否依赖"未来信息"          —— 涨停原因/席位是 T 日盘后公布, 与 dip 同口径, 可用

探测方式
-------
对每个接口给一组年份锚点(每年 1 月首个交易日), 二分收敛到"最早有数据的年"。
不做全历史拉取(几百万行, 无必要), 只判可达性。

产物: output/probe_ds_217.json  —— 供文档引用, 不参与任何策略数值。

用法: python3 tools/probe_ds_217.py
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "probe_ds_217.json"

# 每年 1 月首个交易日附近(允许个别年份元旦前后差异, 只用于探测可达性)
ANCHORS = [
    "20100104", "20120104", "20130104", "20140102", "20150105", "20160104",
    "20170103", "20180102", "20190102", "20200102", "20210104", "20220104",
    "20230103", "20240102", "20250102", "20260105",
]

SLEEP = 0.35  # 匿名/低档 token 的保守间隔


def _len(x) -> int:
    if x is None:
        return 0
    try:
        return len(x)
    except TypeError:
        return 0


def _sample(x, n: int = 1):
    """取前 n 条样本并转成可 JSON 序列化的 dict。"""
    if x is None:
        return []
    if isinstance(x, list):
        rows = x[:n]
        return [r if isinstance(r, dict) else str(r) for r in rows]
    try:
        return x.head(n).to_dict("records")
    except Exception:  # noqa: BLE001
        return [str(x)[:400]]


def _fields(x) -> list[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return sorted(x[0].keys()) if x and isinstance(x[0], dict) else []
    try:
        return [str(c) for c in x.columns]
    except Exception:  # noqa: BLE001
        return []


def call(fn, **kw):
    """统一调用 + 异常兜底, 返回 (行数, 字段, 样本, 错误)。"""
    try:
        x = fn(**kw)
        time.sleep(SLEEP)
        return _len(x), _fields(x), _sample(x), None
    except Exception as e:  # noqa: BLE001
        time.sleep(SLEEP)
        return 0, [], [], f"{type(e).__name__}: {e}"[:200]


def probe_by_date(name: str, fn, extra: dict | None = None):
    """按年份锚点扫描, 返回最早有数据的锚点与首个可用字段。"""
    extra = extra or {}
    hits, first_ok, first_fields, first_sample = [], None, [], []
    for d in ANCHORS:
        n, f, s, err = call(fn, date1=d, **extra)
        hits.append({"date": d, "rows": n, "err": err})
        if n > 0 and first_ok is None:
            first_ok, first_fields, first_sample = d, f, s
        if err and "429" in str(err):
            time.sleep(3)
    return {
        "kind": "by_date",
        "earliest_hit": first_ok,
        "fields": first_fields,
        "sample": first_sample,
        "sweep": hits,
    }


def main() -> None:
    from zzshare.client import DataApi

    api = DataApi()
    report: dict = {"probed_at": time.strftime("%Y-%m-%d %H:%M:%S"), "interfaces": {}}

    targets = [
        # 涨停复盘: 涨停原因(个股逻辑文本)
        ("review_uplimit_reason_open", api.review_uplimit_reason_open, {}),
        # 涨停复盘: 分页版, 可能带板块/连板字段
        ("review_uplimit_reason", api.review_uplimit_reason, {"page": 1, "page_size": 20}),
        # 当日涨停列表(可能含封板时间)
        ("uplimit_stocks", getattr(api, "uplimit_stocks", None), {}),
        # 涨停梯队(连板高度 + 板块)
        ("uplimit_hot", api.uplimit_hot, {"board": ""}),
        # 龙虎榜概览
        ("lhb_list", api.lhb_list, {}),
        # 市场情绪序列(连续区间, 单独测)
        ("sentiment_trend", getattr(api, "sentiment_trend", None), {}),
    ]

    for name, fn, extra in targets:
        if fn is None:
            report["interfaces"][name] = {"kind": "by_date", "fatal": "方法不存在"}
            continue
        try:
            report["interfaces"][name] = probe_by_date(name, fn, extra)
        except Exception:  # noqa: BLE001
            report["interfaces"][name] = {"kind": "by_date", "fatal": traceback.format_exc()[-400:]}
        print(f"[done] {name} -> {report['interfaces'][name].get('earliest_hit')}")

    # ---- 龙虎榜席位明细: 依赖当日上榜股, 先取列表再取席位 ----
    seat = {"kind": "seat", "earliest_hit": None, "fields": [], "sample": [], "sweep": []}
    for d in ANCHORS:
        n, f, s, err = call(api.lhb_list, date1=d)
        code = None
        if n:
            lst = api.lhb_list(date1=d)
            if isinstance(lst, list) and lst:
                code = str(lst[0].get("stock_code") or lst[0].get("code") or "").strip()
            time.sleep(SLEEP)
        if not code:
            seat["sweep"].append({"date": d, "lhb_list_rows": n, "seat_rows": 0, "err": err})
            continue
        sn, sf, ss, serr = call(api.lhb_detail, date1=d, stock_code=code)
        seat["sweep"].append(
            {"date": d, "lhb_list_rows": n, "code": code, "seat_rows": sn, "err": serr}
        )
        if sn > 0 and seat["earliest_hit"] is None:
            seat["earliest_hit"], seat["fields"], seat["sample"] = d, sf, ss
    report["interfaces"]["lhb_detail"] = seat
    print(f"[done] lhb_detail -> {seat.get('earliest_hit')}")

    # ---- 营业部历史战绩: 按已知游资名探测 ----
    trader = {"kind": "trader", "earliest_hit": None, "fields": [], "sample": []}
    for tname in ["华鑫证券有限责任公司上海分公司", "中国银河证券股份有限公司绍兴证券营业部"]:
        n, f, s, err = call(api.lhb_trader_history, trader_name=tname, page=1, per_page=5)
        if n > 0:
            trader.update({"earliest_hit": "n/a(按名检索)", "fields": f, "sample": s, "name": tname})
            break
        trader.setdefault("errs", []).append(err)
    report["interfaces"]["lhb_trader_history"] = trader

    # ---- 分钟线历史: 封板时间/竞价缺口的前提 ----
    mins = {"kind": "minutes", "earliest_hit": None, "fields": [], "sample": [], "sweep": []}
    for d in ["20150105", "20180102", "20200102", "20230103", "20250102"]:
        n, f, s, err = call(
            api.stk_mins, ts_code="000001.SZ", freq="1min",
            start_time=f"{d[:4]}-{d[4:6]}-{d[6:]} 09:30:00",
            end_time=f"{d[:4]}-{d[4:6]}-{d[6:]} 15:00:00",
        )
        mins["sweep"].append({"date": d, "rows": n, "err": err})
        if n > 0 and mins["earliest_hit"] is None:
            mins["earliest_hit"], mins["fields"], mins["sample"] = d, f, s
    report["interfaces"]["stk_mins_1min"] = mins

    # ---- 个股资金流: 大单/小单结构(方法名在不同版本有差异, 逐个试) ----
    mf = {"kind": "moneyflow", "fields": [], "sample": [], "errs": []}
    fn_mf = getattr(api, "stock_moneyflow", None) or getattr(api, "market_mf", None)
    if fn_mf is None:
        mf["errs"].append("0.4.13 无 stock_moneyflow / market_mf 方法")
    else:
        for kw in [{"stock_id": "000001"}, {"stock_id": "000001", "m_type": "day"}, {"stock": "000001"}]:
            try:
                n, f, s, err = call(fn_mf, **kw)
            except Exception as e:  # noqa: BLE001
                mf["errs"].append(f"{type(e).__name__}: {e}"[:160])
                continue
            if n > 0:
                mf.update({"fields": f, "sample": s, "rows": n, "kw": kw})
                break
            mf["errs"].append(err)
    report["interfaces"]["stock_moneyflow"] = mf

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # 控制台摘要(进 Actions Summary 用)
    print("\n=== 数据源可达性速查 ===")
    for k, v in report["interfaces"].items():
        print(f"{k:28s} 最早={v.get('earliest_hit')}  字段数={len(v.get('fields') or [])}")
    print(f"\n写入 {OUT}")


if __name__ == "__main__":
    main()
