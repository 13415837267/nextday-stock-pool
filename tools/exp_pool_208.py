"""第 208 轮: 钱放一起 —— 共享资金池 vs 固定权重切分.

起因
----
第 197 轮做过组合层的**权重**网格(20/80 … 80/20)和两条动态规则(等风险/动量), 结论是
"动态不如固定 60/40, 固定 60/40 的卡玛 7.56 > 纯主腿 6.26"(1,500 日口径)。但 3,000 日
口径下卡玛是**单调向主腿走的**(60/40 = 3.54 < 80/20 = 3.86 < 100/0 = 4.04) —— 同一份
研究在两个窗口给出相反的答案, 说明"权重"这个提法本身就漏了东西。

漏掉的是: 切分权重假设**两腿的钱互不相通**, 于是各自常年在空仓。实测:
  · 主腿 844 笔 × 持 1 晚(T+1 买 / T+2 卖) ≈ 1,688 个在仓日 / 3,645 天 = **46%**
  · 副腿 265 笔 × 持 10 日 ≈ 2,650 个在仓日 / 3,645 天 = **73%**
两者加总 119% —— 也就是说**只有 19% 的时间两条腿同时想要钱**。这意味着一个共享账户
几乎可以永远满仓, 而切分账户两条腿各自有 54% / 27% 的时间在吃 1.5% 的现金利息。

本轮的提法不是"给副策略多少钱"(权重), 而是"**谁有信号谁用钱**"(调度)。三条路线:
  A. 切分(197 轮口径的延伸): 每日按目标权重再平衡, 补 3,645 日全史。
  B. 共享池 · 主腿优先: 主腿在仓日副腿不许开新仓; 主腿空仓时副腿用满仓。
  C. 共享池 · 副腿让半仓: 同上, 但副腿只用 50% 资金(剩下的留现金), 看回撤换不换得回。

纪律
----
· 只读现有 config, 不改任何策略参数; 副腿走 satellite_backtest.run(block_dates=...),
  该参数默认 None 时与旧口径逐位一致。
· **无未来函数**: 判断"明天主腿是否在仓"用的是主腿**已经平仓**的交易记录, 而主腿的
  买入信号在 T 日收盘就已确定, 与副腿同日收盘出信号, 两者信息集相同。
· 共享池的收益序列是**拼接**出来的, 不是两腿各自净值的加权: 主腿在仓日取主腿日收益,
  其余日取副腿(受阻挡版)日收益 —— 因为同一天只可能有一条腿在花钱。

输出: output/exp_pool_208.json (+ 逐年表)
用法: python3 tools/exp_pool_208.py [days]
"""
from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context, run_backtest   # noqa: E402
from src.config import load_config                      # noqa: E402
from src.satellite import entry_ok_series               # noqa: E402
from tools.satellite_backtest import run as sat_run     # noqa: E402

OUT = ROOT / "output"
os.environ["BT_FULL_TRADES"] = "1"
CASH = 0.015 / 252.0


def to_nav(eq: list) -> pd.Series:
    return pd.Series({str(p["d"]).replace("-", "")[:8]: float(p["nav"]) for p in eq}).sort_index()


def stats(nav: pd.Series, label: str) -> dict:
    nav = nav.dropna()
    n = len(nav)
    years = n / 250.0
    tot = float(nav.iloc[-1] / nav.iloc[0] - 1)
    cagr = (1 + tot) ** (1 / years) - 1 if years > 0 and (1 + tot) > 0 else -1.0
    mdd = float((nav / nav.cummax() - 1).min())
    r = nav.pct_change().dropna()
    shp = float(r.mean() / r.std() * np.sqrt(252)) if r.std() else 0.0
    return {"label": label, "days": n, "总收益率%": round(tot * 100, 2),
            "总倍数": round(float(nav.iloc[-1] / nav.iloc[0]), 2),
            "年化收益率%": round(cagr * 100, 2), "最大回撤%": round(mdd * 100, 2),
            "卡玛比率": round(cagr / abs(mdd), 3) if mdd else 0.0,
            "夏普比率": round(shp, 3)}


def monthly_curve(nav: pd.Series) -> dict:
    """每月最后一个交易日的净值 —— 配图只要形状, 不必存 3,645 个点。"""
    g = nav.groupby([str(i)[:6] for i in nav.index])
    return {k: round(float(v.iloc[-1]), 1) for k, v in g}


def yearly(nav: pd.Series) -> dict:
    out = {}
    for y, g in nav.groupby([str(i)[:4] for i in nav.index]):
        if len(g) < 20:
            continue
        out[y] = round((float(g.iloc[-1]) / float(g.iloc[0]) - 1) * 100, 2)
    return out


def filled(trades: list) -> list:
    """trades_all 里混着**未成交的尝试**(缺口未达承接线/评分低于门槛), 只留真成交的。"""
    return [t for t in trades if t.get("buy_date") and t.get("sell_date")
            and t.get("ret_pct") is not None]


def holding_days(trades: list, dates: list) -> set:
    """主腿在仓日: 买入日到卖出日之间的全部交易日(含两端)。"""
    idx = {d: i for i, d in enumerate(dates)}
    hold: set = set()
    for t in trades:
        b = str(t.get("buy_date") or t.get("entry_date") or "")[:8]
        s = str(t.get("sell_date") or t.get("exit_date") or "")[:8]
        if b in idx and s in idx:
            hold.update(dates[idx[b]: idx[s] + 1])
        elif b in idx:
            hold.add(b)
    return hold


def main() -> None:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 3645
    cfg = load_config()
    print(f"== 第208轮 共享资金池 vs 切分权重 {days} 日 ==", flush=True)

    ctx = prepare_context(cfg, days)
    dates = [str(d) for d in ctx["main_dates"]]

    # ---------- 1) 主腿: 全史回测 + 在仓日 ----------
    print("跑主腿…", flush=True)
    rm = run_backtest(cfg, days, ctx)
    main_nav = to_nav(rm["equity"])
    main_tr = filled(rm.get("trades_all") or [])
    H = holding_days(main_tr, dates)
    main_nav = main_nav.reindex(dates).ffill()
    print(f"  主腿 {len(main_tr)} 笔, 在仓 {len(H)} 天 = {len(H) / len(dates) * 100:.1f}%",
          flush=True)

    # ---------- 2) 副腿: 无约束 + 受阻挡 ----------
    ok = entry_ok_series(ctx.get("reg"), cfg)
    # 信号日 T 出票 -> T+1 才买; 若 T+1 主腿在仓, 这份信号作废。
    nxt = {dates[i]: dates[i + 1] for i in range(len(dates) - 1)}
    block_sig = {d for d in dates if nxt.get(d) in H}

    print("跑副腿 base…", flush=True)
    sat0 = sat_run(ctx, cfg, entry_ok=ok)
    print("跑副腿 blocked(主腿在仓日禁开)…", flush=True)
    sat1 = sat_run(ctx, cfg, entry_ok=ok, block_dates=block_sig)

    sat0_nav = to_nav(sat0["equity"]).reindex(dates).ffill()
    sat1_nav = to_nav(sat1["equity"]).reindex(dates).ffill()
    print(f"  副腿 base {sat0['metrics']['交易次数']:.0f} 笔 / "
          f"blocked {sat1['metrics']['交易次数']:.0f} 笔", flush=True)

    r_m = main_nav.pct_change().fillna(0.0)
    r_s0 = sat0_nav.pct_change().fillna(0.0)
    r_s1 = sat1_nav.pct_change().fillna(0.0)
    in_h = pd.Series([d in H for d in dates], index=dates)

    # ---------- 3) 三种口径 ----------
    # 副腿在仓日: 用成交记录算(第一百零八轮: 不能用 nav.diff() != 0, 空仓日的现金
    # 利息也会让净值变动, 会把"在仓率"虚报成 99%)
    S = holding_days(sat0["trades"], dates)
    in_s = pd.Series([d in S for d in dates], index=dates)
    res: dict = {"days": days, "n_days": len(dates),
                 "main_hold_pct": round(len(H) / len(dates) * 100, 2),
                 "sat_hold_pct": round(len(S) / len(dates) * 100, 2),
                 "corr_daily": round(float(r_m.corr(r_s0)), 4)}

    res["legs"] = {"main": stats(main_nav, "主腿 100%"),
                   "sat": stats(sat0_nav, "副腿 100%")}

    # A. 切分: 每日再平衡
    grid = {}
    for w in range(0, 101, 10):
        rr = (w / 100.0) * r_m + (1 - w / 100.0) * r_s0
        nav = (1 + rr).cumprod() * 100000.0
        grid[f"{w}/{100 - w}"] = stats(nav, f"{w}/{100 - w}")
    res["grid"] = grid

    # B. 共享池 · 主腿优先: 主腿在仓日取主腿, 其余取副腿(受阻挡版)
    r_sh = r_m.where(in_h, r_s1)
    nav_sh = (1 + r_sh).cumprod() * 100000.0
    res["shared"] = stats(nav_sh, "共享池 · 主腿优先")

    # C. 共享池 · 副腿只用半仓(其余 50% 吃现金息)
    r_half = r_m.where(in_h, 0.5 * r_s1 + 0.5 * CASH)
    nav_half = (1 + r_half).cumprod() * 100000.0
    res["shared_half"] = stats(nav_half, "共享池 · 副腿半仓")

    # D. 反例: 副腿优先(主腿在主腿=副腿在仓日被挡) —— 只为证明"优先级该给谁"
    sat_in = in_s
    r_rev = r_s0.where(sat_in, r_m)
    nav_rev = (1 + r_rev).cumprod() * 100000.0
    res["shared_sat_first"] = stats(nav_rev, "共享池 · 副腿优先(反例)")

    res["curves"] = {
        "main": monthly_curve(main_nav),
        "sat": monthly_curve(sat0_nav),
        "shared": monthly_curve(nav_sh),
        "shared_half": monthly_curve(nav_half),
        "grid_60_40": monthly_curve((1 + (0.6 * r_m + 0.4 * r_s0)).cumprod() * 100000.0),
    }
    res["yearly"] = {
        "main": yearly(main_nav),
        "shared": yearly(nav_sh),
        "shared_half": yearly(nav_half),
        "grid_60_40": yearly((1 + (0.6 * r_m + 0.4 * r_s0)).cumprod() * 100000.0),
    }
    # 关键分解: 副腿在主腿"在仓日"和"空仓日"的收益是否一样?
    # 如果主腿空仓 = 市场弱 = 副腿也在亏, 那"共享池填满空仓"就是拿低质量仓位换回撤。
    res["split"] = {
        "sat0_ann_when_main_in": round(float(r_s0[in_h].mean()) * 250 * 100, 3),
        "sat0_ann_when_main_out": round(float(r_s0[~in_h].mean()) * 250 * 100, 3),
        "sat1_ann_when_main_in": round(float(r_s1[in_h].mean()) * 250 * 100, 3),
        "sat1_ann_when_main_out": round(float(r_s1[~in_h].mean()) * 250 * 100, 3),
        "main_ann_when_sat_in": round(float(r_m[in_s].mean()) * 250 * 100, 3),
        "main_ann_when_sat_out": round(float(r_m[~in_s].mean()) * 250 * 100, 3),
        "n_main_out": int((~in_h).sum()),
    }
    res["occupancy"] = {
        "main_pct": round(float(in_h.mean()) * 100, 2),
        "sat_pct": round(float(sat_in.mean()) * 100, 2),
        "both_pct": round(float((in_h & sat_in).mean()) * 100, 2),
        "shared_full_pct": round(float((in_h | sat_in).mean()) * 100, 2),
    }

    path = OUT / f"exp_pool_208_{days}.json"
    path.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

    print("\n== 汇总 ==", flush=True)
    rows = [res["legs"]["main"], res["legs"]["sat"]] + list(grid.values()) + \
           [res["shared"], res["shared_half"], res["shared_sat_first"]]
    for v in rows:
        print(f"{v['label']:22s} {v['总倍数']:>12,.1f}×  年化 {v['年化收益率%']:>7.2f}%  "
              f"回撤 {v['最大回撤%']:>8.2f}%  卡玛 {v['卡玛比率']:>6.3f}  "
              f"夏普 {v['夏普比率']:>5.2f}", flush=True)
    print("\n占用:", res["occupancy"], flush=True)
    print("日收益相关:", res["corr_daily"], flush=True)
    print("写出", path, flush=True)


if __name__ == "__main__":
    main()
