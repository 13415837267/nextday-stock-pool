"""第228轮C · bias20 占比前沿（选股条件的连续优化）

228B 发现: 只用 neg_bias20 单因子 → 124.36% / 卡玛 8.56 / 回撤 -14.52%;
现行 5 因子加权(0.70 bias + 0.20 mom + 0.15 其余) → 132.63% / 7.71 / -17.21%。
说明 bias20 是真因子, 其余 4 个在拖累质量(换来了部分收益, 但也换来了更多回撤)。

这里沿"bias 占比"做连续扫描, 看是否存在一个点同时拿到高年化与高卡玛:
  ratio = bias / (bias + mom), 其余三因子固定为 0
  1.00(=228B) / 0.95 / 0.90 / 0.80 / 0.70(=现行 0.70:0.20) / 0.60

判据: 存在 ratio 使 年化 >= 132.63 且 卡玛 >= 7.71 → 纯改进, 直接落地;
      否则取"卡玛最大且年化不明显劣化"的点作为风险优先方案, 交用户拍板。
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

OUT = ROOT / "output" / "w228c_ratio.csv"


def main() -> None:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    tag = sys.argv[2] if len(sys.argv) > 2 else ""
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

    rows = []
    for ratio in (1.00, 0.95, 0.90, 0.80, 0.70, 0.60):
        c = copy.deepcopy(cfg)
        w = 1.0
        c["selection"]["dip_weights"] = {
            "mom20": round((1 - ratio) * w, 4),
            "neg_bias20": round(ratio * w, 4),
            "neg_amount": 0.0, "neg_vol_ratio": 0.0, "low_tail": 0.0,
        }
        t = time.time()
        r = run(c)
        rows.append({"bias占比": ratio, "bias权": round(ratio, 4),
                     "mom权": round(1 - ratio, 4), **r})
        pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8-sig")
        print(f"bias {ratio:.2f} / mom {1-ratio:.2f}  年化{r['年化']:7.2f}% 回撤{r['回撤']:7.2f}% "
              f"卡玛{r['卡玛']:5.2f} {r['笔数']:4d}笔 胜率{r['胜率']:4.1f}% "
              f"单笔{r['单笔']:.3f}% {time.time()-t:.0f}s", flush=True)

    d = pd.DataFrame(rows)
    b = d.iloc[4]   # 0.70 = 现行比例
    print(f"\n现行点(0.70/0.20): 年化{b['年化']}% 卡玛{b['卡玛']}")
    print("\n=== 相对现行 ===")
    for _, r in d.iterrows():
        print(f"bias {r['bias占比']:.2f}  Δ年化{r['年化']-b['年化']:+7.2f}pp  "
              f"Δ卡玛{r['卡玛']-b['卡玛']:+5.2f}  Δ回撤{r['回撤']-b['回撤']:+6.2f}pp")


if __name__ == "__main__":
    main()
