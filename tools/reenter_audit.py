# -*- coding: utf-8 -*-
"""rf_ramp渐进复参与机制十年审计(tools/reenter_audit.py, 第五十二轮)。

问题: 第45轮防线审计了no_regime/no_panic/no_breadth, 但"连续风控超过
max_block_days=5日后按95%仓位渐进复参与(rf_ramp: +0.05/日直至1.0)"这一
设计从未单独审计 —— 当前实盘正处于连续空仓10日的超限参与状态, 直接相关。

两个诊断(事前注册, 纯诊断, 零参数变更 —— 无论结果方向均不修改regime参数,
第44轮已证参数不可调, 本轮目的是补全防线画像):
  A. 画像: BASE十年成交按买入日day_frac分桶(frac==1正常日 / 0<frac<1超限
     渐进参与日), 超限桶的均值/胜率/复利/分年 —— "渐进复参与放行的票质量如何";
  B. 反证: NO_REENTER(风控触发后永不复参与, 即max_block_days=10^7) vs BASE,
     复利口径对比 —— "渐进复参与机制的净贡献"。与第50轮blocked_shadow互补:
     那是单日独立口径(被拦日若放行期望), 本轮是复利路径口径(复参与设计整体)。

口径说明: 买入日frac==0的成交理论上不存在(frac==0日pending_buy为空);
信号日frac决定次日是否买入, 买入日frac决定仓位系数与承接线(渐进递增,
买入日frac>=信号日frac且>0)。超限参与日承接线用block_gap_below=-2.75。

用法: python3 tools/reenter_audit.py   (依赖本地panel缓存, 约4~6分钟)
"""
from __future__ import annotations

import copy
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("BT_FULL_TRADES", "1")   # 需要BASE的trades_all做分桶画像

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from src.config import load_config                     # noqa: E402
from src.backtest import prepare_context, run_backtest  # noqa: E402
from src.selector import apply_regime                  # noqa: E402
from blocked_shadow import build_day_frac              # noqa: E402

ERA = "20240901"   # era切分锚点, 与第42~51轮一致(按buy_date/equity日期)

VERDICT = ("事前注册: 本审计为第45轮防线审计补全(no_regime/no_panic/no_breadth已审, "
           "rf_ramp复参与未审); 无论结果方向均不修改regime参数(第44轮已证参数不可调)。")


def _stats(trades: list[dict]) -> dict:
    """成交桶统计: 单笔均值/胜率/复利和极值。"""
    if not trades:
        return {"n": 0}
    rets = [float(t["ret_pct"]) for t in trades]
    comp = 1.0
    for r in rets:
        comp *= (1 + r / 100)
    return {"n": len(rets), "avg_pct": round(sum(rets) / len(rets), 3),
            "win_pct": round(100 * sum(r > 0 for r in rets) / len(rets), 1),
            "comp_pct": round((comp - 1) * 100, 2),
            "worst_pct": round(min(rets), 2), "best_pct": round(max(rets), 2)}


def _seg(eq: list[dict], start: str, end: str) -> float | None:
    """equity([{d,nav}])在[start,end]段(YYYYMMDD)的复利收益%。"""
    pts = [x["nav"] for x in eq if start <= x["d"].replace("-", "") <= end]
    if len(pts) < 2:
        return None
    return round((pts[-1] / pts[0] - 1) * 100, 2)


def _yearly(eq: list[dict]) -> dict[str, float]:
    """按自然年收益(跨年衔接用上年末nav; 首年基=窗口首日nav)。"""
    years: dict[str, list[float]] = {}
    for x in eq:
        years.setdefault(x["d"][:4], []).append(x["nav"])
    out, prev_last = {}, None
    for y in sorted(years):
        arr = years[y]
        base = prev_last if prev_last else arr[0]
        out[y] = round((arr[-1] / base - 1) * 100, 2)
        prev_last = arr[-1]
    return out


def _metrics(bt: dict) -> dict:
    m = bt["metrics"]
    keep = ("总收益率%", "年化收益率%", "最大回撤%", "卡玛比率", "夏普比率",
            "交易次数", "胜率%", "平均单笔收益%", "空仓天数")
    return {k: m.get(k) for k in keep}


def main():
    cfg = load_config()
    days = int(os.environ.get("REENTER_AUDIT_DAYS", "2500"))
    t0 = time.time()
    print(f"[1/3] prepare_context({days}) ...")
    ctx = prepare_context(cfg, days)
    reg = apply_regime(ctx["reg_base"], cfg) if "reg_base" in ctx else ctx["reg"]
    day_frac = build_day_frac(reg, cfg)
    n_reen_days = sum(1 for v in day_frac.values() if 0 < v < 1)
    n_zero = sum(1 for v in day_frac.values() if v <= 0)
    print(f"  正常日{sum(1 for v in day_frac.values() if v >= 1)} / "
          f"超限渐进参与日{n_reen_days} / 全额空仓日{n_zero}")

    print("[2/3] BASE 回测(rf_ramp=true, 复参与) ...")
    base = run_backtest(cfg, ctx=ctx)
    cfg_no = copy.deepcopy(cfg)
    cfg_no["regime"]["max_block_days"] = 10**7   # 超限日永不出现 -> 风控期恒空仓
    print("[3/3] NO_REENTER 回测(风控后永不复参与) ...")
    nore = run_backtest(cfg_no, ctx=ctx)

    # ---- 诊断A: BASE成交按买入日day_frac分桶 ----
    # 第五十二轮发现: rf_ramp公式min(1, 0.95+0.05*(consec-5))首步即封顶1.0,
    # 超限日frac恒=1.0 —— 分桶不能按frac<1, 须按买入日reg.trade_ok判定。
    trades = [t for t in base.get("trades_all", []) if t.get("ret_pct") is not None]
    reen, norm, ghost = [], [], []
    for t in trades:
        f = float(day_frac.get(t["buy_date"], 1.0))
        t["_frac"] = round(f, 3)
        if f <= 0:
            ghost.append(t)   # frac==0日的成交: 修复前为units=0幽灵成交, 修复后应为0笔
        elif f < 1 or (t["buy_date"] in reg.index and not bool(reg.loc[t["buy_date"], "trade_ok"])):
            reen.append(t)    # 买入日仍处风控期(超限复参与, 承接线block_gap_below)
        else:
            norm.append(t)
    if ghost:
        print(f"  [警告] 仍有{len(ghost)}笔frac==0成交(应为0, 幽灵成交未清干净)")
    years_r = sorted({t["buy_date"][:4] for t in reen})
    by_year_r = {y: _stats([t for t in reen if t["buy_date"].startswith(y)]) for y in years_r}
    era_r = {"before_20240901": _stats([t for t in reen if t["buy_date"] < ERA]),
             "after_20240901": _stats([t for t in reen if t["buy_date"] >= ERA])}

    # ---- 诊断B: BASE vs NO_REENTER ----
    eq_b, eq_n = base["equity"], nore["equity"]
    first, last = eq_b[0]["d"].replace("-", ""), eq_b[-1]["d"].replace("-", "")
    compare = {
        "base": _metrics(base), "no_reenter": _metrics(nore),
        "era": {"base": {"before_20240901": _seg(eq_b, first, "20240831"),
                         "after_20240901": _seg(eq_b, ERA, last)},
                "no_reenter": {"before_20240901": _seg(eq_n, first, "20240831"),
                               "after_20240901": _seg(eq_n, ERA, last)}},
        "by_year": {"base": _yearly(eq_b), "no_reenter": _yearly(eq_n)},
    }

    out = {
        "meta": {"run_time": time.strftime("%Y-%m-%d %H:%M:%S"), "window_days": days,
                 "days": len(eq_b), "reenter_days": n_reen_days, "zero_days": n_zero,
                 "base_cfg": "rf_ramp=true, max_block_days=5, reenter_frac=0.95 (+0.05/日)",
                 "no_reenter_cfg": "max_block_days=10^7 (风控期恒空仓, 永不复参与)",
                 "判据": VERDICT,
                 "口径": ("诊断A: BASE十年成交按买入日day_frac分桶; 诊断B: 同ctx复用"
                          "prepare_context仅改regime.max_block_days重跑, 复利口径。")},
        "profile": {"normal_days": _stats(norm), "reenter_days": _stats(reen),
                    "ghost_n": len(ghost),
                    "reenter_era": era_r, "reenter_by_year": by_year_r,
                    "reenter_trades": reen},
        "compare": compare,
    }
    out_path = ROOT / "output" / "reenter_audit.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float),
                        encoding="utf-8")

    print("==== 诊断A: 超限渐进参与日成交画像(BASE) ====")
    for tag, st in (("正常日   ", _stats(norm)), ("超限参与日", _stats(reen)),
                    ("  era前  ", era_r["before_20240901"]), ("  era后  ", era_r["after_20240901"])):
        if st.get("n"):
            print(f"  {tag}: n={st['n']} 均值{st['avg_pct']:+.2f}%/笔 胜率{st['win_pct']}% 复利{st['comp_pct']:+.1f}%")
    for y in years_r:
        st = by_year_r[y]
        print(f"    {y}: n={st['n']} 均值{st['avg_pct']:+.2f}%/笔 复利{st['comp_pct']:+.1f}%")
    print("==== 诊断B: BASE vs NO_REENTER(复利口径) ====")
    print(f"  BASE      : {compare['base']}")
    print(f"  NO_REENTER: {compare['no_reenter']}")
    print(f"  era后: BASE {compare['era']['base']['after_20240901']}% vs "
          f"NO_REENTER {compare['era']['no_reenter']['after_20240901']}%")
    print(f"耗时 {time.time() - t0:.0f}秒 | 输出: {out_path}")


if __name__ == "__main__":
    main()
