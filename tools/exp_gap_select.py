"""第一百七十八轮: 用"开盘缺口"替代"评分"做最终选择 —— 人工观察入场的可执行版本。

依据
----
exp_factor_ic 的结论是现行评分在可买样本里 IC ≈ 0 (t=+0.15), 而**开盘缺口**是单调有效的:
  gap<=-4% +0.992% / -4~-3 +0.821% / -3~-2 +0.548% / -2~-1 +0.318% / -1~-0.5 +0.007%
(全部为 T+1开→T+2收 净收益, 样本 20,430 条)

用户的工作流本身就是"每日盘后给候选, 人工观察次日开盘入场" —— 也就是说,
**开盘缺口是人工在决策那一刻真的看得见的信息**, 而评分只是盘后的预测。
若"挑缺口最深的那只"优于"挑评分最高的那只", 这个替代就是免费且可执行的。

变体
----
  score_top1   : 现行 —— 评分第 1 名, 缺口达标才买
  gap_deep_N   : 在当日全部候选里, 挑缺口 <= -N 且最深的一只 (没有就不买)
  gap_deep_any : 挑缺口最深的一只, 不设门槛
  gap_band     : 只在 [-N, -M] 区间内挑最深(排除过深接飞刀)

口径与轻量模拟一致: T+1 开盘买, T+2 收盘卖, 同成本, 满仓复利。
这是**排序比较**, 不作最终绩效口径。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context     # noqa: E402
from src.config import load_config          # noqa: E402
from src.selector import select_stocks      # noqa: E402

COST = (2.5 * 2 + 5 + 10) / 1e4


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=1500)
    ap.add_argument("--out", default="output/exp_gap_select.json")
    a = ap.parse_args()

    cfg = load_config()
    ctx = prepare_context(cfg, a.days)
    by_date, dates = ctx["by_date"], ctx["main_dates"]
    pos = {d: i for i, d in enumerate(dates)}

    # 缓存: 每个信号日的候选 + 各自的 T+1 开盘缺口与 T+2 收盘收益
    days_data = []
    for d in dates:
        i = pos[d]
        if i + 2 >= len(dates):
            continue
        picks, _ = select_stocks(d, by_date[d], cfg, None, top_n=300, name_filter=False)
        picks = [p for p in picks if p.get("pool", "main") == "main"]
        if not picks:
            continue
        bd, sd = dates[i + 1], dates[i + 2]
        dfb, dfs = by_date[bd], by_date[sd]
        cands = []
        for k, p in enumerate(picks, 1):
            code = p["code"]
            mb = dfb[dfb["ts_code"] == code]
            ms = dfs[dfs["ts_code"] == code]
            if mb.empty or ms.empty:
                continue
            rb, rs = mb.iloc[0], ms.iloc[0]
            o, c, pc = float(rb["open_adj"]), float(rs["close_adj"]), float(rb["prev_close_adj"])
            if o <= 0 or c <= 0 or pc <= 0:
                continue
            o_raw = float(rb["open"])
            if o_raw >= float(rb["high_limit"]) - 1e-9 or o_raw <= float(rb["low_limit"]) + 1e-9:
                continue
            cands.append({"rank": k, "score": float(p["score"]), "code": code,
                          "gap": (o / pc - 1.0) * 100.0,
                          "ret": (c / o - 1.0 - COST) * 100.0})
        if cands:
            days_data.append({"d": d, "cands": cands})
    print(f"有效候选日 {len(days_data)}")

    def run(tag: str, picker) -> dict:
        nav, navs, rets = 1.0, [1.0], []
        for dd in days_data:
            sel = picker(dd["cands"])
            if sel is None:
                navs.append(nav)
                continue
            nav *= (1 + sel["ret"] / 100.0)
            rets.append(sel["ret"])
            navs.append(nav)
        years = len(navs) / 252.0
        cagr = (nav ** (1 / years) - 1) * 100 if nav > 0 else -100.0
        s = pd.Series(navs)
        mdd = float((s / s.cummax() - 1).min()) * 100
        arr = np.array(rets)
        t = arr.mean() / (arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 2 else np.nan
        r = {"年化%": round(cagr, 2), "回撤%": round(mdd, 2),
             "卡玛": round(cagr / abs(mdd), 2) if mdd else None,
             "笔数": len(arr), "单笔%": round(float(arr.mean()), 3) if len(arr) else None,
             "胜率%": round(float((arr > 0).mean() * 100), 1) if len(arr) else None,
             "t": round(float(t), 2) if not np.isnan(t) else None}
        print(f"{tag:<16} 年化 {r['年化%']:>8.2f}%  回撤 {mdd:>7.2f}%  卡玛 {r['卡玛']:>6.2f}  "
              f"笔 {r['笔数']:>4}  单笔 {r['单笔%']:>+6.3f}%  胜率 {r['胜率%']:>4}%  t={r['t']}")
        return r

    res = {}
    # 现行: 评分第1, 缺口<=0
    res["score_top1"] = run("score_top1",
                            lambda cs: next((c for c in cs if c["rank"] == 1 and c["gap"] <= 0), None))

    for n in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
        res[f"gap_deep_{n}"] = run(
            f"gap_deep_{n:g}",
            lambda cs, n=n: min([c for c in cs if c["gap"] <= -n], key=lambda c: c["gap"], default=None))

    res["gap_deep_any"] = run("gap_deep_any",
                              lambda cs: min(cs, key=lambda c: c["gap"], default=None))

    for lo, hi in [(3.0, 6.0), (2.5, 5.0), (2.0, 4.0), (4.0, 99.0)]:
        res[f"band_{lo:g}_{hi:g}"] = run(
            f"band_{lo:g}_{hi:g}",
            lambda cs, lo=lo, hi=hi: min([c for c in cs if -hi < c["gap"] <= -lo],
                                         key=lambda c: c["gap"], default=None))

    p = ROOT / a.out
    p.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"写出 {p}")


if __name__ == "__main__":
    main()
