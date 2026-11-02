# -*- coding: utf-8 -*-
"""周内日历效应诊断 —— 第一百二十八轮。

研究问题
--------
策略的每一笔都是"信号日 T 收盘出票 → T+1 开盘买 → T+2 收盘卖"。T 落在星期几,
决定了买入日与卖出日落在星期几, 也决定了持仓要跨几个非交易日(周五信号要跨周末)。
**这是一个纯外生的、从没进过模型的维度**: 它不进评分、不进硬筛选、也没被任何一轮扫过。

先验上它多半是噪声 —— 所以本工具的第一职责不是"找到规则", 而是**先证伪**:
  1) 按信号日星期分组, 看均值/胜率/复利贡献是否单调或分组分明;
  2) 控制缺口: 星期效应如果只是"某天缺口更深"的影子, 分缺口桶后应消失;
  3) 分年/分时代: 真效应应在推导窗口(2012~2019)与近端(2020~)同向;
  4) 买入日 vs 卖出日: 效应落在哪条腿上(买入日日内 t1 / 卖出日日内 intra)。

口径声明
--------
· 全史一次回测(BT_FULL_TRADES=1)拿到全部成交, 只做**已实现**成交的归因, 不改任何参数;
· "假设剔除某星期"的复利是**对数贡献求和**的近似(Σln(1+r)), 不含路径重构
  (真实路径会因现金/名额变化而不同), 只能当筛选, 不能当结论 —— 要结论必须走真实回测;
· 本工具不写 config、不写 latest.json, 仅产出 output/exp_dow.json。

用法
----
  python3 tools/exp_dow.py --days 3378
  python3 tools/exp_dow.py --days 1500 --out output/exp_dow_1500.json
"""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ["BT_FULL_TRADES"] = "1"   # 让 run_backtest 把全部成交写进 result["trades_all"]

from src.backtest import prepare_context, run_backtest  # noqa: E402
from src.config import load_config  # noqa: E402

WD = ["周一", "周二", "周三", "周四", "周五"]


def _wd(s):
    try:
        return pd.Timestamp(str(s)).dayofweek
    except Exception:  # noqa: BLE001
        return None


def _stats(g: pd.DataFrame) -> dict:
    if not len(g):
        return {"n": 0}
    r = g["ret_pct"].astype(float)
    logc = np.log1p(r / 100.0)
    return {
        "n": int(len(g)),
        "avg_pct": round(float(r.mean()), 3),
        "med_pct": round(float(r.median()), 3),
        "win_pct": round(float((r > 0).mean() * 100), 2),
        "sumlog": round(float(logc.sum()), 4),
        "avglog_bp": round(float(logc.mean() * 1e4), 1),   # 平均每笔对数贡献(基点)
        "gap_avg": round(float(g["gap_pct"].mean()), 3) if "gap_pct" in g else None,
        "pos_avg": round(float(g["pos_frac"].mean()), 3) if "pos_frac" in g else None,
    }


def _by(df: pd.DataFrame, col: str, label_map=None) -> dict:
    out = {}
    for k, g in df.groupby(col):
        key = label_map(int(k)) if (label_map and k == k) else str(k)
        out[key] = _stats(g)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=3378, help="窗口交易日数(默认全史)")
    ap.add_argument("--out", default="output/exp_dow.json")
    args = ap.parse_args()

    cfg = load_config()
    t0 = time.time()
    print(f"[1/3] 构建上下文 {args.days} 日 ...", flush=True)
    ctx = prepare_context(cfg, args.days)
    print(f"      ctx ready {time.time() - t0:.0f}s, main_dates={len(ctx['main_dates'])}", flush=True)

    print("[2/3] 全史回测(不改任何参数) ...", flush=True)
    t1 = time.time()
    res = run_backtest(cfg, args.days, ctx)
    print(f"      done {time.time() - t1:.0f}s", flush=True)
    m = res.get("metrics", {})

    trades = pd.DataFrame(res.get("trades_all") or [])
    if not len(trades):
        raise RuntimeError("trades_all 为空 —— BT_FULL_TRADES 未生效?")
    ex = trades[trades["ret_pct"].notna()].copy()
    ex["ret_pct"] = ex["ret_pct"].astype(float)
    ex["swd"] = ex["signal_date"].map(_wd)
    ex["bwd"] = ex["buy_date"].map(_wd)
    ex["spd"] = ex["sell_date"].map(_wd)
    ex["year"] = ex["signal_date"].astype(str).str[:4].astype(int)
    ex["era"] = np.where(ex["year"] >= 2020, "2020+", "2012-2019")
    ex["gapb"] = pd.cut(ex["gap_pct"].astype(float),
                        [-99, -2, -1, 0, 0.5, 2, 99],
                        labels=["<-2", "-2~-1", "-1~0", "0~0.5", "0.5~2", ">=2"])

    out = {
        "meta": {
            "days": args.days,
            "start": res.get("meta", {}).get("start"),
            "end": res.get("meta", {}).get("end"),
            "metrics": m,
            "run_sec": round(time.time() - t0),
            "note": "只读诊断; sumlog 为对数贡献求和近似, 非路径重构",
        },
        "by_signal_weekday": _by(ex, "swd", lambda i: WD[i]),
        "by_buy_weekday": _by(ex, "bwd", lambda i: WD[i]),
        "by_sell_weekday": _by(ex, "spd", lambda i: WD[i]),
        "by_era_weekday": {},
        "by_year_weekday": {},
        "by_gap_weekday": {},
    }

    # 分时代 × 星期: 真效应应两段同向
    for era, g in ex.groupby("era"):
        out["by_era_weekday"][era] = {
            WD[int(k)]: _stats(gg) for k, gg in g.groupby("swd") if k == k
        }
    # 分年 × 星期(均值), 供 G3 式一致性检查
    piv_mean = ex.pivot_table(index="year", columns="swd", values="ret_pct", aggfunc="mean")
    piv_n = ex.pivot_table(index="year", columns="swd", values="ret_pct", aggfunc="count")
    out["by_year_weekday"] = {
        str(y): {WD[int(c)]: (round(float(piv_mean.loc[y, c]), 3) if pd.notna(piv_mean.loc[y, c]) else None,
                             int(piv_n.loc[y, c]) if pd.notna(piv_n.loc[y, c]) else 0)
                 for c in piv_mean.columns}
        for y in piv_mean.index
    }
    # 缺口桶 × 星期(控制缺口后效应是否仍在)
    for gb, g in ex.groupby("gapb", observed=True):
        out["by_gap_weekday"][str(gb)] = {
            WD[int(k)]: _stats(gg) for k, gg in g.groupby("swd") if k == k
        }

    # 近似反事实: 剔除某一星期信号后的对数增长
    total_log = float(np.log1p(ex["ret_pct"] / 100.0).sum())
    out["drop_weekday_approx"] = {
        WD[int(wd)]: {
            "removed_n": int((ex["swd"] == wd).sum()),
            "removed_sumlog": round(float(np.log1p(ex.loc[ex["swd"] == wd, "ret_pct"] / 100.0).sum()), 4),
            "approx_total_x": round(float(np.exp(total_log - np.log1p(
                ex.loc[ex["swd"] == wd, "ret_pct"] / 100.0).sum())), 1),
        }
        for wd in sorted(ex["swd"].dropna().unique())
    }
    out["baseline_approx_x"] = round(float(np.exp(total_log)), 1)

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")

    # 控制台摘要(进 GitHub Step Summary 也靠这段)
    print("\n===== 全史基线 =====")
    for k in ["总收益率%", "最大回撤%", "卡玛比率", "交易次数", "胜率%", "平均单笔收益%"]:
        print(f"  {k}: {m.get(k)}")
    print("\n===== 按信号日星期 =====")
    print(f"  {'星期':6s} {'n':>5s} {'均值%':>8s} {'胜率%':>7s} {'对数和':>9s} {'均价差%':>8s}")
    for wd in WD:
        s = out["by_signal_weekday"].get(wd) or {}
        if not s.get("n"):
            continue
        print(f"  {wd:6s} {s['n']:5d} {s['avg_pct']:8.3f} {s['win_pct']:7.2f} "
              f"{s['sumlog']:9.3f} {s.get('gap_avg') or 0:8.3f}")
    print(f"\n  近似全史倍数 {out['baseline_approx_x']}x; 剔除单星期的近似:")
    for k, v in out["drop_weekday_approx"].items():
        print(f"    -{k}: 少 {v['removed_n']} 笔, 对数 {v['removed_sumlog']:+.3f} -> {v['approx_total_x']}x")
    print(f"\n结果已写 {out_path}  (总耗时 {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        sys.exit(1)
