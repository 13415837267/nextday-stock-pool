"""第229轮 B · 全史（4201日）条件核查：连续性 + 叠加

起因：A 批 41 档全史单维扫描里，只有 dip_close_pos_max 0.55 出现大幅正向
（+7.52pp 年化 / 单笔 +0.040），但曲线是锯齿的：
  0.32 → 85.08 / 0.42 → 80.20 / 0.55 → 92.60
  放宽到 0.42 反而变差、再放宽到 0.55 又变好 —— 这是噪声的典型形状，不是台阶。
本批只做三件事：
  1) 连续性：0.45/0.48/0.50/0.60/0.65/0.70，看是否平滑；锯齿即判噪声。
  2) listed_min_bars（A 批唯一"三改善"）连续性：200/300/350。
  3) 叠加：两个三改善项（250 + vol_ratio 6.0）是否仍改善。

判据不变：年化↑ 且 卡玛↑ 且 单笔↑ 且 三段都不低于基准（s3 允许 3pp 内噪声）。
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context, run_backtest  # noqa: E402
from src.config import load_config  # noqa: E402

OUT = ROOT / "output" / "w229_patch.csv"

CASES = [
    ("close_pos 0.45", {"selection__dip_close_pos_max": 0.45}),
    ("close_pos 0.48", {"selection__dip_close_pos_max": 0.48}),
    ("close_pos 0.50", {"selection__dip_close_pos_max": 0.50}),
    ("close_pos 0.60", {"selection__dip_close_pos_max": 0.60}),
    ("close_pos 0.65", {"selection__dip_close_pos_max": 0.65}),
    ("close_pos 0.70", {"selection__dip_close_pos_max": 0.70}),
    ("bars 200", {"universe__listed_min_bars": 200}),
    ("bars 300", {"universe__listed_min_bars": 300}),
    ("bars 350", {"universe__listed_min_bars": 350}),
    ("bars250+vr6", {"universe__listed_min_bars": 250, "selection__dip_vol_ratio_max": 6.0}),
    ("bars250+cp0.55", {"universe__listed_min_bars": 250, "selection__dip_close_pos_max": 0.55}),
]


def tri(res: dict) -> dict:
    eq = pd.DataFrame(res["equity"]).rename(columns={"d": "date"})
    nav = eq["nav"].astype(float).values
    out, n = {}, len(nav)
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
    base_cfg = load_config()
    t0 = time.time()
    ctx = prepare_context(base_cfg, days)
    print(f"ctx {time.time()-t0:.0f}s, {len(ctx['main_dates'])} 日\n", flush=True)

    have = set()
    if OUT.exists():
        prev = pd.read_csv(OUT, encoding="utf-8-sig")
        have = set(prev["方案"])
    else:
        prev = pd.DataFrame()

    def mk(patch: dict) -> dict:
        c = copy.deepcopy(base_cfg)
        for k, v in patch.items():
            node, parts = c, k.split("__")
            for p in parts[:-1]:
                node = node.setdefault(p, {})
            node[parts[-1]] = v
        return c

    if "基准" not in have:
        b = {"方案": "基准", **run_one(base_cfg, ctx)}
        prev = pd.concat([prev, pd.DataFrame([b])])
        prev.to_csv(OUT, index=False, encoding="utf-8-sig")
        print(f"基准  年化{b['年化']:.2f}% 卡玛{b['卡玛']:.2f} 单笔{b['单笔']:.3f}% "
              f"{b['笔数']}笔 三段{b['s1_年化']:.0f}/{b['s2_年化']:.0f}/{b['s3_年化']:.0f}\n", flush=True)
    base = pd.read_csv(OUT, encoding="utf-8-sig").query("方案=='基准'").iloc[0].to_dict()

    for name, patch in CASES:
        if name in have:
            print(f"{name} 已有, 跳过", flush=True)
            continue
        t = time.time()
        r = run_one(mk(patch), ctx)
        row = {"方案": name, **r,
               "Δ年化": round(r["年化"] - base["年化"], 2),
               "Δ卡玛": round(r["卡玛"] - base["卡玛"], 2),
               "Δ单笔": round(r["单笔"] - base["单笔"], 3),
               "cfg": json.dumps(patch, ensure_ascii=False)}
        prev = pd.concat([pd.read_csv(OUT, encoding="utf-8-sig"), pd.DataFrame([row])])
        prev.to_csv(OUT, index=False, encoding="utf-8-sig")
        print(f"{name:<16} 年化{r['年化']:7.2f}%({row['Δ年化']:+6.2f}) 卡玛{r['卡玛']:5.2f}"
              f"({row['Δ卡玛']:+.2f}) 单笔{r['单笔']:.3f}%({row['Δ单笔']:+.3f}) {r['笔数']:4d}笔 "
              f"三段{r['s1_年化']:.0f}/{r['s2_年化']:.0f}/{r['s3_年化']:.0f} {time.time()-t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
