"""第228轮B · 单因子排序 vs 加权打分（选股条件的核心）

现在的 score 是 5 因子加权: mom20 0.20 / neg_bias20 0.70 / neg_amount 0.05
/ neg_vol_ratio 0.02 / low_tail 0.03。权重是历史调出来的, 第 225 轮用 412 万样本
训练权重全部失败(离线好、回测崩), 因为策略是右尾驱动, 优化平均收益会毁掉右尾。

那么换一个方向: **不做加权, 只按单一维度排序**。理由是
  (1) 加权需要权重估计, 而权重在 6 年样本上估不准 → 过拟合;
  (2) 单因子不需要估计, 参数个数为 0, 泛化能力最强;
  (3) 若某个单因子能打平或超过加权, 说明加权里 0.7 的 neg_bias20 已经在主导,
      其余 4 个因子只是噪声。

单因子定义(均按"越优越靠前"排序):
  dip_only      只按回调深度(dip_pct 越负越好, 即越深越好)
  bias_only     只按 bias20 越低(离均线越远越好)
  amt_only      只按成交额越小(盘子小弹性大)
  vr_only       只按量比越小(缩量越极致)
  cpos_only     只按收盘位置越低(恐慌释放越充分)
  mom_only      只按 mom20 越高(前期趋势越强)

判据: 1500 日 + 2500 日双窗口, 年化与卡玛都不低于现行加权, 才考虑替换。
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

OUT = ROOT / "output" / "w228b_single.csv"

CUR = {"mom20": 0.20, "neg_bias20": 0.70, "neg_amount": 0.05,
       "neg_vol_ratio": 0.02, "low_tail": 0.03}

FACTORS = {
    "dip_only":  {"mom20": 0.0, "neg_bias20": 0.0, "neg_amount": 0.0,
                  "neg_vol_ratio": 0.0, "low_tail": 0.0},
    "bias_only": {"mom20": 0.0, "neg_bias20": 1.0, "neg_amount": 0.0,
                  "neg_vol_ratio": 0.0, "low_tail": 0.0},
    "amt_only":  {"mom20": 0.0, "neg_bias20": 0.0, "neg_amount": 1.0,
                  "neg_vol_ratio": 0.0, "low_tail": 0.0},
    "vr_only":   {"mom20": 0.0, "neg_bias20": 0.0, "neg_amount": 0.0,
                  "neg_vol_ratio": 1.0, "low_tail": 0.0},
    "cpos_only": {"mom20": 0.0, "neg_bias20": 0.0, "neg_amount": 0.0,
                  "neg_vol_ratio": 0.0, "low_tail": 1.0},
    "mom_only":  {"mom20": 1.0, "neg_bias20": 0.0, "neg_amount": 0.0,
                  "neg_vol_ratio": 0.0, "low_tail": 0.0},
}


def main() -> None:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    cfg = load_config()
    t0 = time.time()
    ctx = prepare_context(cfg, days)
    print(f"ctx {time.time()-t0:.0f}s, {days} 日\n", flush=True)

    def run(c) -> dict:
        m = run_backtest(c, days, ctx)["metrics"]
        return {"年化": round(float(m["年化收益率%"]), 2),
                "回撤": round(float(m["最大回撤%"]), 2),
                "卡玛": round(float(m["卡玛比率"] or 0), 2),
                "笔数": int(m["交易次数"]),
                "胜率": round(float(m["胜率%"]), 1),
                "单笔": round(float(m["平均单笔收益%"]), 3)}

    base = run(cfg)
    print(f"基准·加权(现行) 年化{base['年化']:7.2f}% 回撤{base['回撤']:7.2f}% 卡玛{base['卡玛']:5.2f} "
          f"{base['笔数']}笔 单笔{base['单笔']:.3f}%\n", flush=True)

    rows = [{"因子": "加权(现行)", **base, "Δ年化": 0.0, "Δ卡玛": 0.0}]
    for name, w in FACTORS.items():
        c = copy.deepcopy(cfg)
        c["selection"]["dip_weights"] = dict(w)
        # 全部为 0 时 score 恒为 0, 排序退化为保持原顺序 → 改用 dip_pct 单因子:
        # selector 内 dip 模式无独立 dip 权重, 因此 dip_only 用"仅 mom20 置 0 其余保持"
        # 无法表达; 这里显式置极小权重避免全 0
        if not any(w.values()):
            c["selection"]["dip_weights"] = {**w, "neg_bias20": 1e-9}
        t = time.time()
        r = run(c)
        row = {"因子": name, **r,
               "Δ年化": round(r["年化"] - base["年化"], 2),
               "Δ卡玛": round(r["卡玛"] - base["卡玛"], 2)}
        rows.append(row)
        pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8-sig")
        flag = "★" if r["年化"] >= base["年化"] and r["卡玛"] >= base["卡玛"] else " "
        print(f"{name:<12} 年化{row['年化']:7.2f}% ({row['Δ年化']:+7.2f}) 回撤{row['回撤']:7.2f}% "
              f"卡玛{row['卡玛']:5.2f} ({row['Δ卡玛']:+.2f}) {row['笔数']:4d}笔 "
              f"胜率{row['胜率']:4.1f}% 单笔{row['单笔']:.3f}% {flag} {time.time()-t:.0f}s", flush=True)

    d = pd.DataFrame(rows)
    print("\n=== 排序 ===")
    print(d.sort_values("年化", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
