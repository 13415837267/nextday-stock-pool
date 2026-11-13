"""第225轮 · 把训练出的权重接进真实回测验证（双窗口 + 帕累托判定）

只替换 selection.dip_weights，其余(regime/执行/成本/仓位)全部沿用 config，
因此差异 100% 来自"同类候选里选谁"。

校验口径:
  - 1500 日（与第219/224轮同源）与全史两段
  - 帕累托定义: 年化更高 且 卡玛不低于现行, 且 两段都不劣化
输出: output/w225_verify.csv
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context, run_backtest  # noqa: E402
from src.config import load_config  # noqa: E402

# 训练集可精确对齐的 6 个因子（low_tail / ivol 在训练里是代理变量，置0不用）
ALIGNED = ["mom20", "neg_bias20", "neg_amount", "neg_vol_ratio",
           "neg_turnover_cool", "neg_dist_low20"]


def clean_weights(w: dict) -> dict:
    # low_tail 在训练集里用 vol20 代理（非下影线），必须置 0 以免错配
    out = {k: (round(float(w.get(k, 0.0)), 3) if k in ALIGNED else 0.0) for k in
           ["mom20", "neg_bias20", "neg_amount", "neg_vol_ratio",
            "low_tail", "neg_turnover_cool", "ivol", "neg_dist_low20"]}
    # ivol 训练时用 vol20 代理, 与真实 ivol20 方向一致且高度相关, 保留其权重
    out["ivol"] = round(float(w.get("ivol", 0.0)), 3)
    # neg_close_pos 不在训练空间内, 保持 0
    return out


def brief(m: dict) -> dict:
    return {"年化": m["年化收益率%"], "回撤": m["最大回撤%"],
            "卡玛": m["卡玛比率"], "笔数": m["交易次数"], "胜率": m["胜率%"]}


def main() -> None:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    cfg = load_config()
    src_json = sys.argv[2] if len(sys.argv) > 2 else "w225_best.json"
    cand = json.load(open(ROOT / "output" / src_json))

    ctx = prepare_context(cfg, days)
    print(f"上下文 {len(ctx['main_dates'])} 日", flush=True)

    rows = []
    base = run_backtest(cfg, days, ctx)
    b = brief(base["metrics"])
    print(f"基准(现行): {b}", flush=True)
    rows.append({"名称": "现行", **b, "权重": json.dumps(cfg['selection']['dip_weights'], ensure_ascii=False)})

    for i, c in enumerate(cand, 1):
        w = clean_weights(c["weights"])
        cfg2 = copy.deepcopy(cfg)
        cfg2["selection"]["dip_weights"] = w
        try:
            r = run_backtest(cfg2, days, ctx)
        except Exception as e:  # noqa: BLE001
            print(f"  [FAIL] #{i}: {e}", flush=True)
            continue
        br = brief(r["metrics"])
        d_ann = br["年化"] - b["年化"]
        d_cal = br["卡玛"] - b["卡玛"]
        print(f"#{i}: {br}  Δ年化 {d_ann:+.1f}pp Δ卡玛 {d_cal:+.2f}", flush=True)
        print(f"    w={ {k: v for k, v in w.items() if v} }", flush=True)
        rows.append({"名称": f"训练#{i}", **br, "Δ年化": round(d_ann, 2),
                     "Δ卡玛": round(d_cal, 2),
                     "权重": json.dumps({k: v for k, v in w.items() if v}, ensure_ascii=False)})

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "output" / "w225_verify.csv", index=False, encoding="utf-8-sig")
    print("\n" + df.to_string(index=False))


if __name__ == "__main__":
    main()
