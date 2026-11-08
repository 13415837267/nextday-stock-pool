# -*- coding: utf-8 -*-
"""第一百七十五轮(诊断): 979 次"被跳过"的信号, 到底误杀了多少 alpha。

为什么做这个
------------
`output/backtest.json` 里一直挂着一个没人追问的数字: **跳过买入次数 979**, 而成交只有 354 笔。
也就是说每 4 个"想买"的信号里有 3 个被规则挡掉。这些规则(缺口承接线、崩塌线、评分门槛、
一字板、停牌)每一条都是当年某一轮回测"加进来更好"才留下的, 但**没有人回头算过它们各自
一共误杀了多少 alpha** —— 加进来时只看了净效果, 没看被剔除的那部分本身值多少钱。

方法
----
1. 跑一次 base, 导出全量 trades(含 ret_pct=None 的跳过记录)。每条跳过记录都带 note
   (剔除原因)和 buy_date(T+1)。
2. 对每条跳过记录, 用 per_code 重算"假如买了"的收益: T+1 开盘买 -> T+2 开盘卖,
   扣除与正式回测相同的双边成本(FEE)。
3. 按原因分组输出 n / 单笔均值 / 胜率 / 正期望的置信度。

判读标准
--------
某条规则若"被它剔除的票"均值显著为负 -> 规则是对的, 保留;
若均值显著为正 -> 规则在误杀, 应当放宽或改成降仓。
注意这是**反事实**, 不构成直接落地依据: 真放宽后池子会变, 还可能挤占后续更好的信号。

用法: python tools/exp_skipped.py [days]
输出: output/exp_skipped.json
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ["BT_FULL_TRADES"] = "1"

from src.backtest import prepare_context, run_backtest       # noqa: E402
from src.config import load_config                          # noqa: E402

OUT = os.path.join(ROOT, "output")


def fee_factor(ex):
    slip = float(ex.get("slippage", 0.001))
    comm = float(ex.get("commission_rate", 0.00025))
    stamp = float(ex.get("stamp_tax", 0.0005))
    return (1 - slip) * (1 - comm - stamp) / ((1 + slip) * (1 + comm))


def bucket(note: str) -> str:
    """把 note 归一化成稳定的类别名(去掉具体数值)。"""
    if not note:
        return "未标注"
    n = note
    n = re.sub(r"[-+]?\d+\.\d+%", "<x>%", n)
    n = re.sub(r"[-+]?\d+\.\d+", "<x>", n)
    return n


def main(days: int = 1500) -> None:
    cfg = load_config()
    print(f"\n[prepare] days={days}", flush=True)
    ctx = prepare_context(cfg, days)
    cfg["execution"]["leverage"] = 1.25

    r = run_backtest(cfg, days, ctx)
    m = r["metrics"]
    print("[base] 年化 %.2f%% 回撤 %.2f%% 笔数 %d 跳过 %d"
          % (m["年化收益率%"], m["最大回撤%"], m["交易次数"], m["跳过买入次数"]), flush=True)

    trades = r.get("trades_all") or []
    skips = [t for t in trades if t.get("ret_pct") is None]
    print(f"[diag] 跳过记录 {len(skips)} 条", flush=True)

    ff = fee_factor(cfg["execution"])
    bd = ctx["by_date"]
    cal = sorted(bd)                      # 全局交易日, 字典序即时间序
    cal_i = {d: i for i, d in enumerate(cal)}

    def px(d: str, code: str, col: str = "open_adj") -> float | None:
        df = bd.get(d)
        if df is None:
            return None
        m = df[df["ts_code"] == code]
        if m.empty:
            return None
        try:
            v = float(m.iloc[0][col])
        except Exception:
            return None
        return v if np.isfinite(v) and v > 0 else None

    rows = []
    miss: dict[str, int] = defaultdict(int)
    for t in skips:
        code, d = t["code"], t.get("buy_date") or ""
        note = t.get("note") or ""
        # 一字涨停/停牌当天本来就买不到, 这类"反事实"没有意义, 单独归类
        if "停牌" in note or "一字" in note:
            rows.append({"kind": bucket(note), "ret": None, "buy": d})
            continue
        i = cal_i.get(d)
        if i is None:
            miss["日期不在日历(停牌日)"] += 1
            continue
        b = px(d, code)
        if b is None:
            miss["T+1 取不到价"] += 1
            continue
        # T+2 = 下一个有该票行情的交易日(最多顺延 5 日, 与正式卖出逻辑一致)
        s_, d2 = None, None
        for j in range(i + 1, min(i + 7, len(cal))):
            v = px(cal[j], code)
            if v is not None:
                s_, d2 = v, cal[j]
                break
        if s_ is None:
            miss["T+2 取不到价"] += 1
            continue
        rows.append({"kind": bucket(note), "ret": float(s_ / b * ff - 1.0), "buy": d,
                     "gap": note})

    print(f"[diag] 可重算 {sum(1 for x in rows if x['ret'] is not None)} 条, "
          f"停牌/一字 {sum(1 for x in rows if x['ret'] is None)} 条, "
          f"取价失败 {sum(miss.values())} 条 {dict(miss) if miss else ''}\n", flush=True)

    by: dict[str, list] = defaultdict(list)
    for x in rows:
        if x["ret"] is not None:
            by[x["kind"]].append(x["ret"])

    out_rows = []
    print("%-34s %5s %9s %8s %7s %8s" % ("剔除原因", "n", "反事实单笔%", "std%", "胜率%", "t值"))
    print("-" * 78)
    for k, v in sorted(by.items(), key=lambda kv: -len(kv[1])):
        a = np.array(v)
        n = len(a)
        mu = a.mean()
        sd = a.std(ddof=1) if n > 1 else 0.0
        t = mu / (sd / np.sqrt(n)) if sd > 0 and n > 1 else 0.0
        win = (a > 0).mean()
        out_rows.append({"kind": k, "n": n, "mean_pct": round(float(mu) * 100, 3),
                         "std_pct": round(float(sd) * 100, 3),
                         "win_pct": round(float(win) * 100, 1), "t": round(float(t), 2)})
        flag = "  <<< 误杀?" if (mu > 0 and t > 2) else ""
        print("%-34s %5d %+9.3f %8.3f %7.1f %8.2f%s"
              % (k[:34], n, mu * 100, sd * 100, win * 100, t, flag), flush=True)

    allv = np.array([x["ret"] for x in rows if x["ret"] is not None])
    if len(allv):
        print("-" * 78)
        print("%-34s %5d %+9.3f %8.3f %7.1f %8.2f"
              % ("合计", len(allv), allv.mean() * 100, allv.std(ddof=1) * 100,
                 (allv > 0).mean() * 100,
                 allv.mean() / (allv.std(ddof=1) / np.sqrt(len(allv)))))

    # 分年: 误杀是否只在某一段
    yr: dict[str, list] = defaultdict(list)
    for x in rows:
        if x["ret"] is not None:
            yr[str(x["buy"])[:4]].append(x["ret"])
    print("\n=== 分年(全部跳过信号的反事实单笔) ===")
    for y in sorted(yr):
        a = np.array(yr[y])
        print("  %s  n=%3d  %+7.3f%%" % (y, len(a), a.mean() * 100))

    os.makedirs(OUT, exist_ok=True)
    fn = "exp_skipped.json" if days <= 1600 else f"exp_skipped_{days}.json"
    with open(os.path.join(OUT, fn), "w", encoding="utf-8") as f:
        json.dump({"days": days, "base": m, "rows": out_rows,
                   "by_year": {y: {"n": len(v), "mean_pct": round(float(np.mean(v)) * 100, 3)}
                               for y, v in yr.items()}}, f, ensure_ascii=False, indent=1)
    print(f"\nsaved output/{fn}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1500)
