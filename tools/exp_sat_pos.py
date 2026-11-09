# -*- coding: utf-8 -*-
"""第182轮·D: 入场缓冲 × 仓位只数 —— 两个已验证的改动从没一起测过

为什么必须补这一格
------------------
- 第一百七十九轮: 买入端限价(信号日收盘 -1%, 3 日有效) -> 1500 日 19.0%->32.4%
- 第一百八十轮: 单吊(max_pos 4->1) -> 全史回撤 -36.1%->-42.6%, 用户拍板选单吊
- 第一百八十一轮: 入场缓冲(指数须站上 MA20+0.3%) -> 1500 日 28.1%->42.2%, 卡玛 1.95

但 181 轮的缓冲**只在单吊下测过**。缓冲收紧后"每天能选的票变少", 多只并发可能拿不满仓;
反过来, 多只并发在缓冲下能分散单票风险, 回撤也许不再需要用"单吊"来换。
这一格没测过, 而它直接决定 180 轮那个选择还成不成立。

变体全部在当前口径下(限价 1%/3日 + 闸门 + 缓冲 0.3), 只改仓位结构:
  base   max_pos=1 buys=1  (现行单吊)
  pos2   max_pos=2 buys=1  (两份钱, 每日最多建 1 仓 -> 逐步建满)
  pos2b  max_pos=2 buys=2  (两份钱, 一次买满)
  pos3   max_pos=3 buys=1
  pos3b  max_pos=3 buys=3
  pos4b  max_pos=4 buys=4  (第 179 轮口径, 无缓冲时的 32.4%/-15.5%/2.09)

产物: output/exp_sat_pos.json
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


def yearly(res: dict) -> dict:
    out = {}
    for pt in res["equity"]:
        y = pt["d"][:4]
        out.setdefault(y, [pt["nav"], pt["nav"]])
        out[y][1] = pt["nav"]
    return {y: round((v[1] / v[0] - 1) * 100, 1) for y, v in out.items()}


def run_one(ctx, cfg, entry, sat=None):
    c = copy.deepcopy(cfg)
    if sat:
        c["satellite"].update(sat)
    res = sb.run(dict(ctx), c, entry_ok=entry)
    m = res["metrics"]
    return {"总收益%": m["总收益率%"], "年化%": m["年化收益率%"], "回撤%": m["最大回撤%"],
            "卡玛": m["卡玛比率"], "笔数": m["交易次数"], "胜率%": m["胜率%"],
            "单笔%": m["平均单笔收益%"], "空仓天": m["空仓天数"], "年度": yearly(res)}


def main():
    cfg = load_config()
    out = {"windows": {}}
    items = [
        ("base", {}),
        ("pos2", {"max_pos": 2, "buys_per_day": 1}),
        ("pos2b", {"max_pos": 2, "buys_per_day": 2}),
        ("pos3", {"max_pos": 3, "buys_per_day": 1}),
        ("pos3b", {"max_pos": 3, "buys_per_day": 3}),
        ("pos4b", {"max_pos": 4, "buys_per_day": 4}),
    ]
    for days in (1500, 3053):
        print(f"\n===== 窗口 {days} 日 =====", flush=True)
        ctx = prepare_context(cfg, days)
        entry = entry_ok_series(ctx.get("reg"), cfg)
        w = {"days": len(ctx["main_dates"]), "start": ctx["main_dates"][0],
             "end": ctx["main_dates"][-1], "variants": {}}
        for nm, sat in items:
            try:
                r = run_one(ctx, cfg, entry, sat)
            except Exception as ex:
                print(f"{nm:8s} ERROR {type(ex).__name__}: {ex}", flush=True)
                w["variants"][nm] = {"error": str(ex)}
                continue
            w["variants"][nm] = r
            print(f"{nm:8s} 总{r['总收益%']:+9.1f}% 年化{r['年化%']:+6.1f}% "
                  f"回撤{r['回撤%']:7.1f}% 卡玛{r['卡玛']:5.2f} 笔{r['笔数']:4d} "
                  f"胜{r['胜率%']:5.1f}% 单笔{r['单笔%']:+.2f}%", flush=True)
        out["windows"][str(days)] = w

    p = ROOT / "output/exp_sat_pos.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"\n写出 {p}", flush=True)


if __name__ == "__main__":
    main()
