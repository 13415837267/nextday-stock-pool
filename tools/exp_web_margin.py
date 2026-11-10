# -*- coding: utf-8 -*-
"""第184轮·E: 两融(融资余额)当市场级闸门 —— 与个股无关的第五个维度

为什么测这个
------------
前面四个因子(max/imax/on_abs/ctc)都是**个股横截面**因子, 全否。两融是唯一的
**市场级总量**数据(沪深两市合计, data/margin 每日一行), 与现有 MA20 闸门的信息源
完全正交: MA20 看价格, 两融看杠杆资金的存量与流量。

公开口径(第184轮检索):
  · 融资余额变化率 = (当日余额 - 前N日均值)/前N日均值, 反映杠杆资金进出(券商金工常用);
  · 超额融资因子 = 融资余额增长率 - 指数同期收益, 剔除"跟涨"后的净加杠杆意愿
    (某券商金工: 周频 10% 对冲组合年化 20.06%、夏普 2.26);
  · 反证: 日度融资净买入对 T+1 涨跌相关性仅 0.01~0.03, 且方向在 2020 年前后翻转
    → 故本脚本**不做次日择时**, 只做"是否允许开新仓"的慢变量闸门。

四个 mask 与现役 entry_ok 相与(不替换), 检验"加一层杠杆闸门"是否降回撤:
  rz5_up    近5日融资余额上升
  rz20_up   近20日融资余额上升
  rz_ex15  超额融资(15日余额增长 - 15日指数收益) > 0
  rzcalm    融资买入占余额比 < 其250日滚动中位(杠杆不过热)

所有滚动量只用当日及之前数据(rolling/quantile 均不含未来)。

用法: python tools/exp_web_margin.py 1500
"""
from __future__ import annotations

import copy
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context  # noqa: E402
from src.config import load_config  # noqa: E402
from src.satellite import entry_ok_series  # noqa: E402
import tools.satellite_backtest as sb  # noqa: E402


def load_margin() -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(str(ROOT / "data/margin/*.csv.gz"))):
        try:
            d = pd.read_csv(f)
        except Exception:  # noqa: BLE001
            continue
        if len(d):
            rows.append(d.iloc[[0]])
    m = pd.concat(rows, ignore_index=True)
    # 注意: 截取年份必须用 .str[:8]; 写成 .str.replace("-", "")[:8] 是对 Series 切片(只留前 8 行),
    # 会让整列变成 8 行 + 其余 NaN, 后面 drop_duplicates 只剩 9 个日期(本轮踩过)。
    m["trade_date"] = m["trade_date"].astype(str).str.replace("-", "").str[:8]
    return m.drop_duplicates("trade_date").set_index("trade_date").sort_index()


def masks(m: pd.DataFrame, mkt: pd.Series) -> dict:
    rz = m["rzye"].astype(float)
    buy = m["rzbuy"].astype(float)
    out = {}
    out["rz5_up"] = (rz.pct_change(5) > 0)
    out["rz20_up"] = (rz.pct_change(20) > 0)
    ex15 = rz.pct_change(15) - mkt.reindex(rz.index).rolling(15).sum() / 100.0
    out["rz_ex15"] = ex15 > 0
    ratio = buy / rz.replace(0, np.nan)
    out["rzcalm"] = ratio < ratio.rolling(250, min_periods=120).median()
    return out


def run_one(ctx, cfg, sat: dict, extra: pd.Series | None = None):
    c = copy.deepcopy(cfg)
    c.setdefault("satellite", {}).update(sat)
    e = entry_ok_series(ctx.get("reg"), c)
    if extra is not None:
        # index 对齐纪律: entry_ok_series 的索引跟着 reg 走(本地 csv 装载时可能是 int),
        # 而 margin 序列是自己按文件名拼的字符串日期。不统一成 str 会 reindex 全 NaN,
        # 表现为"放行率 0%、0 笔"(本轮踩过)。
        ex = extra.copy()
        ex.index = ex.index.astype(str)
        e = e.copy()
        idx = e.index.astype(str)
        e.index = idx
        e = e & ex.reindex(idx).fillna(False)
    res = sb.run(dict(ctx), c, entry_ok=e)
    m = res["metrics"]
    return {"总收益%": m["总收益率%"], "年化%": m["年化收益率%"], "回撤%": m["最大回撤%"],
            "卡玛": m["卡玛比率"], "笔数": m["交易次数"], "胜率%": m["胜率%"],
            "单笔%": m["平均单笔收益%"], "空仓天": m["空仓天数"],
            "闸门放行率%": round(100.0 * float(e.mean()), 1)}


FULL = {"mode": "lowvol", "max_pos": 1, "buys_per_day": 1, "hold_days": 10,
        "ivol_pct_max": 0.1, "sort_by": "mom20_ivol", "min_score": 0,
        "pullback_pct": 1.0, "pullback_days": 3, "pullback_mode": "low",
        "market_gate": True, "gate_entry_buffer": 0.3,
        "gate_exit": True, "gate_exit_mode": "all",
        "trail_pct": 8.0, "stop_pct": -12.0, "min_pool": 20,
        "exit_limit_pct": 0.0}


def main():
    cfg = load_config()
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    ctx = prepare_context(cfg, days)
    print(f"实际 {len(ctx['main_dates'])} 日 {ctx['main_dates'][0]} ~ {ctx['main_dates'][-1]}",
          flush=True)
    m = load_margin()
    # 指数(全市场等权)日收益: 直接用面板 mkt_pct 的逐日均值
    by_date = ctx["by_date"]
    mkt = pd.Series({str(d): float(v["pct"].mean()) for d, v in by_date.items()}).sort_index()
    mk = masks(m, mkt)
    out = {"days": len(ctx["main_dates"]), "variants": {},
           "coverage": {"mask_days": {k: int(v.reindex(ctx["main_dates"]).fillna(False).sum())
                                      for k, v in mk.items()},
                        "window": len(ctx["main_dates"])}}
    r0 = run_one(ctx, cfg, FULL)
    out["variants"]["base"] = r0
    print(f"{'base':12s} 总{r0['总收益%']:>10.1f}% 年化{r0['年化%']:7.1f}% 回撤{r0['回撤%']:6.1f}% "
          f"卡玛{r0['卡玛']:5.2f} 笔{r0['笔数']:4d} 单笔{r0['单笔%']:+.2f}%", flush=True)
    for nm, s in mk.items():
        r = run_one(ctx, cfg, FULL, extra=s)
        if "error" in r:
            print(f"{nm:12s} ERROR {r['error']}", flush=True)
            continue
        out["variants"][nm] = r
        print(f"{nm:12s} 总{r['总收益%'] or 0:>10.1f}% 年化{r['年化%'] or 0:7.1f}% "
              f"回撤{r['回撤%'] or 0:6.1f}% 卡玛{r['卡玛'] or 0:5.2f} 笔{r['笔数'] or 0:4d} "
              f"单笔{r['单笔%'] or 0:+.2f}% 放行{r['闸门放行率%'] or 0:.0f}%", flush=True)
    p = ROOT / f"output/exp_web_margin_{days}.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"写出 {p}", flush=True)


if __name__ == "__main__":
    main()
