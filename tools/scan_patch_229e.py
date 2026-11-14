"""第229轮 E · 全史（4201日）新条件实测：流通市值 / 振幅 / VWAP偏离 / 下影线

为什么是这四个：
  D 批把 952 笔按观测值分箱后，只有这四个变量呈现非平坦的收益形状：
    · circ_mv  <27 亿 0.48%、>156 亿 2.59%  —— 单调且 5 倍差
    · amp      <4.1% 0.97%、4.95~7.42% 1.63~1.71%、>7.42% 0.97%  —— 倒 U
    · vwap_dev 收盘低于均价 vs 高于均价 —— 第172轮曾测过、近期转正
    · low_tail 下影线 —— 盘中是否已有承接
  gap（低开幅度）分箱几乎全平（1.14~1.35），判死，不测。

但 D 批的分箱是"在同一批 952 笔里切开"，而**改条件后会由第 2 名替补**，
分箱看到的差异未必能在重选中存活 —— 本批用真实回测裁决，不用分箱结论。
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



CASES = [
    ("circ_mv_min 20", {"universe__circ_mv_min": 20}),
    ("circ_mv_min 27", {"universe__circ_mv_min": 27}),
    ("circ_mv_min 40", {"universe__circ_mv_min": 40}),
    ("circ_mv_min 60", {"universe__circ_mv_min": 60}),
    ("amp_max 8.5", {"selection__dip_amp_max": 8.5}),
    ("amp_max 7.0", {"selection__dip_amp_max": 7.0}),
    ("amp_max 6.0", {"selection__dip_amp_max": 6.0}),
    ("vwap_dev_max -1.0", {"selection__dip_vwap_dev_max": -1.0}),
    ("vwap_dev_max 0.0", {"selection__dip_vwap_dev_max": 0.0}),
    ("low_tail_min 0.3", {"selection__dip_low_tail_min": 0.3}),
    ("low_tail_min 0.6", {"selection__dip_low_tail_min": 0.6}),
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
    out = ROOT / "output" / f"w229_newcond_{days}.csv"
    base_cfg = load_config()
    t0 = time.time()
    ctx = prepare_context(base_cfg, days)
    print(f"ctx {time.time()-t0:.0f}s, {len(ctx['main_dates'])} 日\n", flush=True)

    prev = pd.read_csv(out, encoding="utf-8-sig") if out.exists() else pd.DataFrame()
    have = set(prev["方案"]) if len(prev) else set()

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
        prev.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"基准  年化{b['年化']:.2f}% 卡玛{b['卡玛']:.2f} 单笔{b['单笔']:.3f}% "
              f"{b['笔数']}笔 三段{b['s1_年化']:.0f}/{b['s2_年化']:.0f}/{b['s3_年化']:.0f}\n", flush=True)
    base = pd.read_csv(out, encoding="utf-8-sig").query("方案=='基准'").iloc[0].to_dict()

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
        prev = pd.concat([pd.read_csv(out, encoding="utf-8-sig"), pd.DataFrame([row])])
        prev.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"{name:<18} 年化{r['年化']:7.2f}%({row['Δ年化']:+6.2f}) 卡玛{r['卡玛']:5.2f}"
              f"({row['Δ卡玛']:+.2f}) 单笔{r['单笔']:.3f}%({row['Δ单笔']:+.3f}) {r['笔数']:4d}笔 "
              f"三段{r['s1_年化']:.0f}/{r['s2_年化']:.0f}/{r['s3_年化']:.0f} {time.time()-t:.0f}s",
              flush=True)


if __name__ == "__main__":
    main()
