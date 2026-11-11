# -*- coding: utf-8 -*-
"""第 196 轮: 把"空仓的 76% 时间"变成第二条腿 —— 双引擎资金复用。

背景
----
第 195 轮关掉杠杆后, 主策略 1500 日: 总收益 15879.65%、年化 134.66%、回撤 -17.61%、
卡玛 7.65、365 笔。但 **空仓天数 1133 / 1499 = 75.6%** —— 一笔只在场 1 个交易日
(T+1 开盘买 / T+2 开盘卖), 其余四分之三的时间里, 全部资金只拿 1.5% 的货币收益。

同期副策略(lowvol + mom20/ivol, H10, 单吊)在场率约 70%, 两者**不是同一天进场**:
主策略吃"缩量阴线回调"的次日反弹, 副策略吃"低波强势股"的 10 日趋势。相关性低。

所以本轮问一个从未问过的问题:
    **资金能不能在主策略空档期借给副策略, 有信号时立刻收回?**

这不是第 185 轮的"静态权重网格"(70% 主 / 30% 副 同时持有), 而是**时间维度上的复用**:
同一笔钱在不同日子干不同的活, 任一时点只有一条腿在场。与"分散"是两件事。

口径(必须写清, 否则数字会被误读)
----
1. 日收益拼接, 不是重跑引擎: 主策略日收益 r_m(t) 取自 equity, 用逐笔 pos_frac 重建
   持仓区间 w_m(t); 副策略同理(entry_date~exit_date, 单吊 w_s=1)。
2. 主策略日收益里含现金计息 1.5%。为了把"复用"与"计息"分开, 先反解持仓日毛收益
   r_pos = (r_m - (1-w_m)·cash_d) / w_m, 再按新的资金分配重组。
3. 副策略部分按 (1-w_m) 缩放, 即"主策略占多少, 副策略就只能用剩下的"。
   这是**偏乐观**的口径: 现实中主策略插队时副策略要卖出再回补, 有 0.3%×2 的换手成本,
   脚本单独做成本敏感性(每次让位扣 0.3%/0.6%/1.0%)。
4. 副策略本身不计息(与线上一致), 复用部分空仓时按 cash_yield=1.5% 计息。

用法: python3 -u tools/exp_reuse_196.py [days]
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest import prepare_context, run_backtest   # noqa: E402
from src.config import ROOT, load_config                # noqa: E402
from tools import satellite_backtest as SB              # noqa: E402
from src.satellite import entry_ok_series               # noqa: E402

OUT = ROOT / "output"


def _d8(s) -> str:
    return str(s).replace("-", "")[:8]


def prep(days: int) -> tuple[dict, dict]:
    """跑两条腿并落盘(带全量逐笔)。已有缓存则复用。"""
    fp_m = OUT / f"reuse_196_m{days}.json"
    fp_s = OUT / f"reuse_196_s{days}.json"
    os.environ["BT_FULL_TRADES"] = "1"
    cfg = load_config()
    # 复用实验固定用第 195 轮末的口径(boost 可透支), 与已落盘的 1500 日缓存一致;
    # 第 196 轮新加的 pos_frac_cap 只影响"能不能透支", 不改变持仓区间, 对归因无影响。
    cfg["execution"]["pos_frac_cap"] = 0
    ctx = prepare_context(cfg, days)
    print(f"窗口 {len(ctx['main_dates'])} 日 {ctx['main_dates'][0]}~{ctx['main_dates'][-1]}", flush=True)

    if fp_m.exists():
        res_m = json.loads(fp_m.read_text(encoding="utf-8"))
        print(f"复用主策略 {fp_m.name}", flush=True)
    else:
        res_m = run_backtest(cfg, days, ctx)
        fp_m.write_text(json.dumps(res_m, ensure_ascii=False, default=float), encoding="utf-8")
        print(f"主策略落盘 {fp_m.name}", flush=True)

    if fp_s.exists():
        res_s = json.loads(fp_s.read_text(encoding="utf-8"))
        print(f"复用副策略 {fp_s.name}", flush=True)
    else:
        _entry = entry_ok_series(ctx.get("reg"), cfg)
        res_s = SB.run(ctx, cfg, entry_ok=_entry)
        fp_s.write_text(json.dumps(res_s, ensure_ascii=False, default=float), encoding="utf-8")
        print(f"副策略落盘 {fp_s.name}", flush=True)
    return res_m, res_s


def _series(res: dict) -> tuple[list[str], dict[str, float]]:
    eq = res["equity"]
    dates = [_d8(x["d"]) for x in eq]
    nav = {_d8(x["d"]): float(x["nav"]) for x in eq}
    return dates, nav


def _leg(res: dict, key_buy: str, key_sell: str, pos_key: str | None) -> tuple[list, dict, dict]:
    """返回 (dates, 日收益 r(t), 持仓权重 w(t))。"""
    dates, nav = _series(res)
    r = {}
    for i in range(1, len(dates)):
        prev = nav[dates[i - 1]]
        r[dates[i]] = nav[dates[i]] / prev - 1.0 if prev else 0.0
    w = {d: 0.0 for d in dates}
    idx = {d: i for i, d in enumerate(dates)}
    n_hold = 0
    for t in res.get("trades_all") or res.get("trades") or []:
        if t.get("ret_pct") is None:
            continue
        b, s = _d8(t.get(key_buy) or ""), _d8(t.get(key_sell) or "")
        if b not in idx or s not in idx:
            continue
        pf = float(t.get(pos_key) or 1.0) if pos_key else 1.0
        # 卖出日的隔夜跳空仍属于持仓收益(T+2 开盘卖), 必须含卖出日;
        # 若只算 [buy, sell) 会漏掉 365 个隔夜, 拼接口径会比引擎少一半以上。
        for i in range(idx[b], idx[s] + 1):
            w[dates[i]] = max(w[dates[i]], pf)
            n_hold += 1
    return dates, r, w


def _metrics(dates: list[str], r: dict[str, float], w: dict[str, float] | None = None) -> dict:
    seq = [r[d] for d in dates[1:] if d in r]
    nav = np.cumprod([1.0 + x for x in seq])
    total = nav[-1] - 1.0
    years = len(seq) / 250.0
    cagr = (1 + total) ** (1 / years) - 1 if (1 + total) > 0 and years > 0 else -1.0
    peak = np.maximum.accumulate(nav)
    mdd = float((nav / peak - 1).min())
    vol = float(np.std(seq) * np.sqrt(252))
    kama = cagr / abs(mdd) if mdd else None
    expo = float(np.mean([w[d] for d in dates[1:] if d in w])) if w else None
    return {"总收益率%": round(total * 100, 2), "年化收益率%": round(cagr * 100, 2),
            "最大回撤%": round(mdd * 100, 2), "卡玛比率": round(kama, 2) if kama else None,
            "年化波动%": round(vol * 100, 2), "平均在场": round(expo, 4) if expo else None,
            "净值序列": nav}


def combine(res_m: dict, res_s: dict, cash_yield: float, switch_cost: float = 0.0,
            mode: str = "reuse", cap: float = 1.0) -> dict:
    dates, rm, wm = _leg(res_m, "buy_date", "sell_date", "pos_frac")
    _, rs, ws = _leg(res_s, "entry_date", "exit_date", None)
    cash_d = cash_yield / 252.0
    nav = 1.0
    out_r, out_w, hits = {}, {}, 0
    for i, d in enumerate(dates):
        if i == 0:
            out_r[d] = 0.0
            out_w[d] = 0.0
            continue
        a = wm.get(d, 0.0)                      # 主策略仓位(0 或 pos_frac)
        if mode == "main":
            a = a
            b = 0.0
        else:
            b = ws.get(d, 0.0)
            if a > 0 and b > 0:
                hits += 1                       # 同日两条腿都想在场: 主策略优先, 副策略让位
        avail = max(0.0, 1.0 - a)
        rp_m = (rm.get(d, 0.0) - (1 - a) * cash_d) / a if a > 0 else 0.0
        rp_s = rs.get(d, 0.0) / b if b > 0 else 0.0
        w_s = b * avail * cap if mode != "main" else 0.0
        r = a * rp_m + avail * (w_s * rp_s + (1 - w_s) * cash_d)
        if switch_cost and a > 0 and b > 0:
            r -= switch_cost                    # 让位换手成本(单次, 按总资产)
        out_r[d] = r
        out_w[d] = a + w_s
    m = _metrics(dates, out_r, out_w)
    m["冲突日"] = hits
    m["日收益"] = out_r
    return m


def block_boot(diff: list[float], block: int = 20, n: int = 2000, seed: int = 196) -> dict:
    rng = np.random.default_rng(seed)
    a = np.asarray(diff, dtype=float)
    T = len(a)
    if T < block * 2:
        return {}
    nb = int(np.ceil(T / block))
    means = []
    for _ in range(n):
        starts = rng.integers(0, T - block, size=nb)
        s = np.concatenate([a[i:i + block] for i in starts])[:T]
        means.append(s.mean())
    means = np.array(means)
    # 日收益差的年化近似: 均值 × 250
    return {"日均差_bp": round(float(a.mean()) * 1e4, 2),
            "年化差_pct": round(float(a.mean()) * 250 * 100, 2),
            "P(差<=0)": round(float((means <= 0).mean()), 4),
            "95%区间_bp": [round(float(np.percentile(means, 2.5)) * 1e4, 2),
                          round(float(np.percentile(means, 97.5)) * 1e4, 2)]}


def main() -> None:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    cash_yield = float(load_config().get("execution", {}).get("cash_yield", 0.015) or 0.0)
    print(f"== 第196轮 资金时间复用: {days} 日, cash_yield={cash_yield} ==", flush=True)
    res_m, res_s = prep(days)

    mm = res_m["metrics"]
    ms = res_s["metrics"]
    print(f"主策略 引擎 {mm['总收益率%']:.2f}% 年化 {mm['年化收益率%']:.2f}% 回撤 {mm['最大回撤%']:.2f}% "
          f"卡玛 {mm['卡玛比率']} 笔 {mm['交易次数']}", flush=True)
    print(f"副策略 引擎 {ms['总收益率%']:.2f}% 年化 {ms['年化收益率%']:.2f}% 回撤 {ms['最大回撤%']:.2f}% "
          f"卡玛 {ms['卡玛比率']} 笔 {ms['交易次数']}", flush=True)

    base = combine(res_m, res_s, cash_yield, 0.0, "main")
    print(f"基线(纯主/日收益拼接) {base['总收益率%']:.2f}% 年化 {base['年化收益率%']:.2f}% "
          f"回撤 {base['最大回撤%']:.2f}% 卡玛 {base['卡玛比率']} 在场 {base['平均在场']}", flush=True)

    rows = []
    for sc in (0.0, 0.003, 0.006, 0.010):
        c = combine(res_m, res_s, cash_yield, sc, "reuse")
        diff = [c["日收益"][d] - base["日收益"][d] for d in c["日收益"] if d in base["日收益"]]
        bt = block_boot(diff[1:])
        rows.append({"切换成本": sc, **{k: v for k, v in c.items() if k not in ("净值序列", "日收益")}, **bt})
        print(f"复用 成本{sc * 100:.1f}%: {c['总收益率%']:.2f}% 年化 {c['年化收益率%']:.2f}% "
              f"回撤 {c['最大回撤%']:.2f}% 卡玛 {c['卡玛比率']} 在场 {c['平均在场']:.3f} "
              f"冲突日 {c['冲突日']} | 年化差 {bt.get('年化差_pct')}pp P(<=0)={bt.get('P(差<=0)')}", flush=True)

    fp = OUT / f"exp_reuse_196_{days}.json"
    fp.write_text(json.dumps({"days": days, "cash_yield": cash_yield,
                              "main": {k: v for k, v in base.items() if k not in ("净值序列", "日收益")},
                              "variants": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"写出 {fp}", flush=True)


if __name__ == "__main__":
    main()
