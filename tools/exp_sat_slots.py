# -*- coding: utf-8 -*-
"""第185轮: 副策略"百倍"的第三条路 —— 并发槽位 × 持有期矩阵

为什么测这个(前几轮没测过的空白)
--------------------------------
第183轮把"百倍"拆成三条路: ①日 alpha×3.5 ②在场天数×2.5 ③杠杆×2.8。
①② 已被证伪, ③ 的代价(2.8 倍杠杆 = -55% 回撤)用户不接受。但还有第四条被漏掉了:

    **④ 资金占用率**

现行副策略是单吊 1 只、100% 仓位、持 10 日。但它只在 53% 的交易日"有资格"开仓
(MA20+0.3% 闸门)。剩下 47% 的日子, 100% 现金躺着 —— 这不是风控, 是闲置。

若改成 max_pos=2/3/4 且 buys_per_day 同步放开, 每一笔只用 1/max_pos 的钱,
但同时在场的槽位变多 -> 资金占用率从 ~50% 抬到 80%+。
这在数学上等于"无息杠杆": 收益按占用率同比放大, 但因为是**分散在 N 只不同股票**上,
个股特异性风险被摊掉, 回撤放大倍数 < 收益放大倍数 -> 卡玛可能反而上升。
第 176 轮曾测到"4 只并发 卡玛 2.25 > 单吊 1.95", 但当时是旧口径(18.99% 仓位),
从未在现行口径(回踩限价 + 双向闸门 + 单日 1 只 -> N 只)下重测。

同时扫持有期: 单吊 H10 的笔数上限是 1500/11=136, 已用 106, structural 天花板。
H5 上限 250, H7 上限 187 —— 若单笔赔率掉得没那么快, 乘积可能更大。

产物: output/exp_sat_slots.json
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context  # noqa: E402
from src.config import load_config  # noqa: E402
from src.satellite import entry_ok_series  # noqa: E402
import tools.satellite_backtest as sb  # noqa: E402

BASE = {"mode": "lowvol", "max_pos": 1, "buys_per_day": 1, "hold_days": 10,
        "ivol_pct_max": 0.1, "sort_by": "mom20_ivol", "min_score": 0,
        "pullback_pct": 1.0, "pullback_days": 3, "pullback_mode": "low",
        "market_gate": True, "gate_entry_buffer": 0.3,
        "gate_exit": True, "gate_exit_mode": "all",
        "trail_pct": 8.0, "stop_pct": -12.0, "min_pool": 20,
        "exit_limit_pct": 0.0, "leverage": 1.0, "margin_rate": 0.055}


def run_one(ctx, cfg, sat: dict):
    c = copy.deepcopy(cfg)
    c.setdefault("satellite", {}).update(sat)
    e = entry_ok_series(ctx.get("reg"), c)
    res = sb.run(dict(ctx), c, entry_ok=e)
    m = res["metrics"]
    return {"总收益%": m["总收益率%"], "倍数": round(1.0 + m["总收益率%"] / 100.0, 2),
            "年化%": m["年化收益率%"], "回撤%": m["最大回撤%"], "卡玛": m["卡玛比率"],
            "笔数": m["交易次数"], "胜率%": m["胜率%"], "单笔%": m["平均单笔收益%"],
            "空仓天": m["空仓天数"]}


def main():
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    tag = sys.argv[2] if len(sys.argv) > 2 else "a"

    if tag == "a":       # 矩阵: 持有期 × 槽位
        plan = []
        for h in (5, 7, 10):
            for mp in (1, 2, 3, 4):
                nm = f"h{h}_p{mp}"
                plan.append((nm, {**BASE, "hold_days": h, "max_pos": mp,
                                  "buys_per_day": mp}))
    elif tag == "b":     # 池宽度 × 最优槽位
        plan = []
        for iv in (0.03, 0.05, 0.1, 0.2, 0.3):
            for mp in (4,):
                plan.append((f"iv{int(iv*100)}_p{mp}",
                             {**BASE, "ivol_pct_max": iv, "max_pos": mp,
                              "buys_per_day": mp}))
            plan.append((f"iv{int(iv*100)}_h5p4",
                         {**BASE, "ivol_pct_max": iv, "hold_days": 5,
                          "max_pos": 4, "buys_per_day": 4}))
    elif tag == "c":     # 池宽度 × 单吊 H10（单笔赔率能不能再抬）
        plan = []
        for iv in (0.02, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5):
            plan.append((f"iv{int(iv*100):02d}_h10p1",
                         {**BASE, "ivol_pct_max": iv}))
    else:                # 杠杆叠加
        plan = []
        for lv in (1.0, 1.25, 1.5, 2.0):
            plan.append((f"h5p4_lev{lv}",
                         {**BASE, "hold_days": 5, "max_pos": 4, "buys_per_day": 4,
                          "leverage": lv}))

    cfg = load_config()
    ctx = prepare_context(cfg, days)
    print(f"===== {len(ctx['main_dates'])} 日 {ctx['main_dates'][0]} ~ "
          f"{ctx['main_dates'][-1]} | tag={tag} =====", flush=True)
    out = {"days": len(ctx["main_dates"]), "tag": tag, "variants": {}}
    for nm, sat in plan:
        try:
            r = run_one(ctx, cfg, sat)
        except Exception as ex:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            out["variants"][nm] = {"error": str(ex)}
            print(f"{nm:14s} ERROR {ex}", flush=True)
            continue
        out["variants"][nm] = r
        print(f"{nm:14s} {r['倍数']:>9.2f}x 年化{r['年化%']:7.1f}% "
              f"回撤{r['回撤%']:7.1f}% 卡玛{r['卡玛']:6.2f} 笔{r['笔数']:4d} "
              f"胜{r['胜率%']:5.1f}% 单笔{r['单笔%']:+.2f}% 空仓{r['空仓天']:4d}",
              flush=True)
    p = ROOT / f"output/exp_sat_slots_{tag}_{days}.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float),
                 encoding="utf-8")
    print(f"写出 {p}", flush=True)


if __name__ == "__main__":
    main()
