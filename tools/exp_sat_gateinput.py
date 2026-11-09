# -*- coding: utf-8 -*-
"""第182轮·C: 最后一层没动过的输入 —— 大盘指数怎么算出来的

为什么要动它
------------
副策略的大盘闸门只有一个真正生效的判据: `mkt_idx >= MA20`(第 181 轮已证明宽度条件
从不先触发)。而 mkt_idx 的构造是 `(1 + 全市场 pct 均值).cumprod()` —— 等权均值。
均值对小票暴涨极度敏感: 某天 4000 只票里 100 只涨停, 均值被拉高 0.3%, 指数虚高,
闸门会在"多数票还在跌"的时候开着; 反过来某天少数大票暴跌也会压低指数。

本轮只换指数的**输入统计量**, 闸门规则(MA20 + 入场 0.3% 缓冲)一字不动:
  mean    —— 现行: 全市场 pct 均值(等权)
  median  —— 全市场 pct 中位数(更抗极端值)
  trim    —— 10%~90% 分位截尾均值
  br45/55 —— 不用指数, 直接把宽度(breadth>=0.45/0.55)当主判据

同时给一个对照: 只放宽/收紧宽度阈值, 确认"宽度条件是否真的从不生效"。

产物: output/exp_sat_gateinput.json
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context  # noqa: E402
from src.config import load_config  # noqa: E402
from src.selector import apply_regime  # noqa: E402
from src.satellite import entry_ok_series  # noqa: E402
import tools.satellite_backtest as sb  # noqa: E402


def yearly(res: dict) -> dict:
    out = {}
    for pt in res["equity"]:
        y = pt["d"][:4]
        out.setdefault(y, [pt["nav"], pt["nav"]])
        out[y][1] = pt["nav"]
    return {y: round((v[1] / v[0] - 1) * 100, 1) for y, v in out.items()}


def build_reg(ctx, cfg, kind: str):
    """按不同统计量重建 mkt_idx, 再套用同一套风控规则。"""
    rows = {}
    for d in ctx["reg"].index:
        day = ctx["by_date"].get(str(d))
        if day is None or not len(day):
            rows[d] = (float("nan"), float("nan"))
            continue
        m = ((day["is_paused"] == 0) & (day["is_st"] == 0) & (day["bars"] >= 21))
        p = day.loc[m, "pct"].astype(float)
        br = float(day.loc[m, "above_ma20"].mean())
        if kind == "median":
            v = float(p.median())
        elif kind == "trim":
            if len(p) > 20:
                lo, hi = p.quantile(0.10), p.quantile(0.90)
                v = float(p[(p >= lo) & (p <= hi)].mean())
            else:
                v = float(p.mean())
        else:
            v = float(p.mean())
        rows[d] = (br, v)
    df = pd.DataFrame.from_dict(rows, orient="index", columns=["breadth", "mkt_pct"])
    df.index = [str(i) for i in df.index]
    df["mkt_idx"] = (1 + df["mkt_pct"] / 100).cumprod()
    return apply_regime(df, cfg)


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
    for days, kinds in ((1500, ["mean", "median", "trim", "br45", "br55", "br35"]),
                        (3053, ["mean", "median", "trim", "br45"])):
        print(f"\n===== 窗口 {days} 日 =====", flush=True)
        ctx = prepare_context(cfg, days)
        w = {"days": len(ctx["main_dates"]), "start": ctx["main_dates"][0],
             "end": ctx["main_dates"][-1], "variants": {}}
        for kind in kinds:
            try:
                c2 = copy.deepcopy(cfg)
                if kind.startswith("br"):
                    c2["regime"]["breadth_min"] = float(kind[2:]) / 100.0
                    c2["regime"]["mode"] = "and"
                    reg = apply_regime(ctx["reg_base"], c2)
                elif kind == "mean":
                    reg = ctx["reg"]
                else:
                    reg = build_reg(ctx, cfg, kind)
                ctx2 = dict(ctx)
                ctx2["reg"] = reg
                entry = entry_ok_series(reg, c2)
                r = run_one(ctx2, c2, entry)
                _r = reg.copy()
                _r.index = [str(i) for i in _r.index]
                r["闸门开%"] = round(float(_r.loc[_r.index.isin(ctx["main_dates"]),
                                                 "trade_ok"].mean()) * 100, 1)
                r["入场开%"] = round(float(entry.mean()) * 100, 1) if entry is not None else None
            except Exception as ex:
                print(f"{kind:8s} ERROR {type(ex).__name__}: {ex}", flush=True)
                w["variants"][kind] = {"error": str(ex)}
                continue
            w["variants"][kind] = r
            print(f"{kind:8s} 总{r['总收益%']:+9.1f}% 年化{r['年化%']:+6.1f}% "
                  f"回撤{r['回撤%']:7.1f}% 卡玛{r['卡玛']:5.2f} 笔{r['笔数']:4d} "
                  f"胜{r['胜率%']:5.1f}% 闸门开{r['闸门开%']:5.1f}% 入场开{r['入场开%']}", flush=True)
        out["windows"][str(days)] = w

    p = ROOT / "output/exp_sat_gateinput.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"\n写出 {p}", flush=True)


if __name__ == "__main__":
    main()
