"""第二百三十四轮: 改排序权重(不改笔数) —— 唯一不触发替补效应的改法

前面的结论: 加任何硬条件都亏, 因为策略是"每天买评分第 1 名", 筛掉第 1 名就换成第 2 名,
而排序本身值 +0.51pp(条件子集 0.741% -> 实际成交 1.252%)。

所以唯一可行的改进是改排序, 不改入围。而全样本归纳指出:
  在策略条件内(跌3-6% + 收盘低 + 动量5-25%), 20 日乖离越大收益越高(0.636 -> 0.845),
  但评分给的是 neg_bias20 = 0.7 —— 乖离越小得分越高, 方向相反。

本轮直接翻转这个符号, 并测几个替代组合。注意权重绝对值改变会移动 score 分布,
进而影响 min_top1_score=1.0 门槛, 所以必须同时看笔数是否漂移。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context, run_backtest      # noqa: E402
from src.config import load_config                          # noqa: E402

OUT = ROOT / "output" / "w234_weight.csv"


def brief(res: dict) -> dict:
    m = res["metrics"]
    nav = pd.DataFrame(res["equity"])["nav"].astype(float).values
    n = len(nav)
    seg = {}
    for i, tag in enumerate(["s1", "s2", "s3"]):
        x = nav[int(n * i / 3): int(n * (i + 1) / 3)]
        yrs = max(0.3, len(x) / 244.0)
        seg[tag] = round(((x[-1] / x[0]) ** (1 / yrs) - 1) * 100, 1) if len(x) > 40 else None
    return {"年化": round(float(m["年化收益率%"]), 2),
            "回撤": round(float(m["最大回撤%"]), 2),
            "卡玛": round(float(m["卡玛比率"] or 0), 2),
            "笔数": int(m["交易次数"]),
            "胜率": round(float(m["胜率%"]), 1),
            "单笔": round(float(m["平均单笔收益%"]), 3), **seg}


def mk(base: dict, w: dict, gate: float | None = None) -> dict:
    c = dict(base)
    c["selection"] = dict(base["selection"])
    c["selection"]["dip_weights"] = dict(base["selection"]["dip_weights"], **w)
    if gate is not None:
        c["selection"]["min_top1_score"] = gate
    return c


def main() -> None:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 2500
    base = load_config()
    t0 = time.time()
    ctx = prepare_context(base, days)
    print(f"ctx {time.time()-t0:.0f}s, {len(ctx['main_dates'])} 日\n", flush=True)

    plan = [
        ("E0 基准(bias -0.7)", {}, None),
        ("E1 bias 翻转(+0.7)", {"neg_bias20": -0.7}, None),
        ("E2 bias 归零", {"neg_bias20": 0.0}, None),
        ("E3 bias 减半(0.35)", {"neg_bias20": 0.35}, None),
        ("E4 纯动量(mom.6/bias0)", {"neg_bias20": 0.0, "mom20": 0.6}, None),
        ("E5 mom.3 + bias翻(-.5)", {"neg_bias20": -0.5, "mom20": 0.3}, None),
    ]
    rows = []
    for name, w, gate in plan:
        t = time.time()
        r = brief(run_backtest(mk(base, w, gate), None, ctx))
        r["方案"] = name
        if rows:
            r["Δ年化"] = round(r["年化"] - rows[0]["年化"], 2)
            r["Δ笔数"] = r["笔数"] - rows[0]["笔数"]
        else:
            r["Δ年化"], r["Δ笔数"] = 0.0, 0
        rows.append(r)
        print(f"{name:26s} 年化 {r['年化']:7.2f}% ({r['Δ年化']:+6.2f})  "
              f"回撤 {r['回撤']:7.2f}%  卡玛 {r['卡玛']:5.2f}  "
              f"笔数 {r['笔数']:4d} ({r['Δ笔数']:+4d})  单笔 {r['单笔']:+.3f}%  "
              f"[{time.time()-t:.0f}s]", flush=True)
        pd.DataFrame(rows)[["方案", "年化", "回撤", "卡玛", "笔数", "胜率", "单笔",
                            "s1", "s2", "s3", "Δ年化", "Δ笔数"]].to_csv(
            OUT, index=False, encoding="utf-8-sig")

    print("\n" + pd.DataFrame(rows)[["方案", "年化", "回撤", "卡玛", "笔数", "胜率",
                                     "单笔", "s1", "s2", "s3"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
