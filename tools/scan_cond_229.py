"""第229轮 · 全史（4201日）选股条件全量挖掘

用户要求：**基于全历史**、**只挖条件**。

与 226/228 轮的区别：
  · 226/228 的窗口是最近 1500 日（2020-09 起），其中 2021(+190%)/2022(+243%) 两个大年
    占权重极大；227 轮已经证明"1500 日上的改进在 2500 日上系统性翻转"。
  · 本轮直接用全部 4201 个交易日（2009-07 起），并以**三段切分**代替两段：
    只靠某一段撑起来的改善一律不算通过。

判据（条件挖掘专用，四条）：
  1) 全史年化 >= 基准；2) 全史卡玛 >= 基准；
  3) 三段（前/中/后 1/3）年化全部 >= 基准对应段；
  4) 单笔收益 >= 基准（防止"靠笔数堆出来的年化"）。

工程：ctx 只建一次（约 6 分钟），每个配置约 40s；每完成一个立刻落盘，
      进程被杀可幂等续跑（跳过 CSV 里已有的 (参数,取值)）。
"""
from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context, run_backtest  # noqa: E402
from src.config import load_config  # noqa: E402

OUT = ROOT / "output" / "w229_cond.csv"

# 只动"选股条件"：回调形态 / 安全边际 / 股票池门槛。不动权重、不动卖出、不动风控。
GRID = [
    ("dip_pct_min", "selection", [-8.0, -7.0, -5.0, -4.0]),
    ("dip_pct_max", "selection", [-4.5, -3.5, -2.0]),
    ("dip_vol_ratio_max", "selection", [2.0, 3.0, 5.0, 6.0]),
    ("dip_mom20_min", "selection", [0.0, 3.0, 8.0, 10.0]),
    ("dip_mom20_max", "selection", [15.0, 20.0, 35.0, 45.0]),
    ("dip_dist_low20_min", "selection", [2.0, 4.0, 9.0, 12.0]),
    ("dip_close_pos_max", "selection", [0.20, 0.42, 0.55]),
    ("bias20_max", "selection", [9.0, 10.5, 12.5, 14.0]),
    ("amount_min_yi", "universe", [0.5, 1.2, 2.0]),
    ("turnover_min", "universe", [1.0, 3.0]),
    ("turnover_max", "universe", [8.0, 18.0]),
    ("price_min", "universe", [5.0, 8.0]),
    ("listed_min_bars", "universe", [250, 400]),
]


def tri(res: dict) -> dict:
    """三段等分年化/卡玛（按净值曲线三等分），用于识别"只靠某一段"的改善。"""
    eq = pd.DataFrame(res["equity"]).rename(columns={"d": "date"})
    nav = eq["nav"].astype(float).values
    out = {}
    n = len(nav)
    for i, tag in enumerate(["s1", "s2", "s3"]):
        seg = nav[int(n * i / 3): int(n * (i + 1) / 3)]
        if len(seg) < 40:
            out[f"{tag}_年化"], out[f"{tag}_卡玛"] = 0.0, 0.0
            continue
        yrs = max(0.3, len(seg) / 244.0)
        ann = (seg[-1] / seg[0]) ** (1 / yrs) - 1
        mdd = float((seg / pd.Series(seg).cummax().values - 1).min())
        out[f"{tag}_年化"] = round(ann * 100, 2)
        out[f"{tag}_卡玛"] = round(ann / abs(mdd) if mdd else 0.0, 2)
    return out


def run_one(cfg, ctx) -> dict:
    res = run_backtest(cfg, None, ctx)
    m = res["metrics"]
    d = {"年化": round(float(m["年化收益率%"]), 2),
         "回撤": round(float(m["最大回撤%"]), 2),
         "卡玛": round(float(m["卡玛比率"] or 0), 2),
         "笔数": int(m["交易次数"]),
         "胜率": round(float(m["胜率%"]), 1),
         "单笔": round(float(m["平均单笔收益%"]), 3)}
    d.update(tri(res))
    return d


def main() -> None:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 4201
    cfg = load_config()

    t0 = time.time()
    ctx = prepare_context(cfg, days)
    print(f"ctx {time.time()-t0:.0f}s, 窗口 {len(ctx['main_dates'])} 日 "
          f"({ctx['main_dates'][0]}~{ctx['main_dates'][-1]})\n", flush=True)

    if OUT.exists():
        done = pd.read_csv(OUT, encoding="utf-8-sig")
        base_row = done[done["参数"] == "基准"].head(1).to_dict("records")
    else:
        done = pd.DataFrame()
        base_row = []

    if base_row:
        base = base_row[0]
        print(f"基准(存档)  年化{base['年化']:7.2f}% 卡玛{base['卡玛']:5.2f} "
              f"单笔{base['单笔']:.3f}% {int(base['笔数'])}笔\n", flush=True)
    else:
        base = {"参数": "基准", "取值": "-", **run_one(cfg, ctx)}
        pd.DataFrame([base]).to_csv(OUT, index=False, encoding="utf-8-sig")
        print(f"基准        年化{base['年化']:7.2f}% 回撤{base['回撤']:7.2f}% 卡玛{base['卡玛']:5.2f} "
              f"{base['笔数']}笔 单笔{base['单笔']:.3f}%  "
              f"三段{base['s1_年化']:.0f}/{base['s2_年化']:.0f}/{base['s3_年化']:.0f}\n", flush=True)

    have = set()
    if len(done):
        for _, r in done[done["参数"] != "基准"].iterrows():
            have.add((r["参数"], round(float(r["取值"]), 4)))

    for name, sect, vals in GRID:
        cur = cfg[sect][name]
        print(f"--- {name} (当前 {cur}) ---", flush=True)
        for v in vals:
            if (name, round(float(v), 4)) in have:
                print(f"  {v} 已有, 跳过", flush=True)
                continue
            c = copy.deepcopy(cfg)
            c[sect][name] = type(cur)(v) if not isinstance(cur, bool) else v
            t = time.time()
            r = run_one(c, ctx)
            row = {"参数": name, "取值": v, "当前": cur, **r,
                   "Δ年化": round(r["年化"] - base["年化"], 2),
                   "Δ卡玛": round(r["卡玛"] - base["卡玛"], 2),
                   "Δ单笔": round(r["单笔"] - base["单笔"], 3),
                   "Δ笔数": r["笔数"] - int(base["笔数"])}
            pd.concat([pd.read_csv(OUT, encoding="utf-8-sig") if OUT.exists() else pd.DataFrame(),
                       pd.DataFrame([row])]).to_csv(OUT, index=False, encoding="utf-8-sig")
            ok = (row["Δ年化"] > 0 and row["Δ卡玛"] >= 0 and row["Δ单笔"] >= 0
                  and r["s1_年化"] >= base["s1_年化"] and r["s2_年化"] >= base["s2_年化"]
                  and r["s3_年化"] >= base["s3_年化"])
            print(f"  {v:<8} 年化{r['年化']:7.2f}%({row['Δ年化']:+6.2f}) 卡玛{r['卡玛']:5.2f}"
                  f"({row['Δ卡玛']:+.2f}) 单笔{r['单笔']:.3f}%({row['Δ单笔']:+.3f}) "
                  f"{r['笔数']:4d}笔 三段{r['s1_年化']:.0f}/{r['s2_年化']:.0f}/{r['s3_年化']:.0f}"
                  f" {'★四过' if ok else ''} {time.time()-t:.0f}s", flush=True)

    d = pd.read_csv(OUT, encoding="utf-8-sig")
    d = d[d["参数"] != "基准"]
    w = d[(d["Δ年化"] > 0) & (d["Δ卡玛"] >= 0) & (d["Δ单笔"] >= 0)].sort_values("Δ年化", ascending=False)
    print("\n=== 年化↑ 且 卡玛↑ 且 单笔↑ ===")
    cols = ["参数", "取值", "年化", "卡玛", "单笔", "笔数", "Δ年化", "Δ卡玛", "Δ单笔",
            "s1_年化", "s2_年化", "s3_年化"]
    print(w[cols].to_string(index=False) if len(w) else "(无)")


if __name__ == "__main__":
    main()
