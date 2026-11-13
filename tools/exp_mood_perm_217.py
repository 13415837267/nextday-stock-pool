"""第二百一十七轮(续四): 置换检验 —— 卡玛提升是"选对了时点"还是"降仓本身"

为什么必须做这一步
----------------
任何"降低平均仓位"的方案都会同时降低收益和回撤, 卡玛(年化/回撤)经常因此被动上升。
如果只报"卡玛 4.18→4.99", 无法区分两种解释:

  H1 质量分选对了降仓时点 → 真实卡玛应显著高于随机降仓
  H2 只是降仓的算术副作用 → 真实卡玛落在随机分布中间

做法: 固定倍率集合 {1.0, 0.8, 0.6}(各 1/3, 与真实方案完全一致), 随机打乱后分配给
368 笔交易, 重复 2000 次, 得到卡玛的经验分布; 再看真实方案的卡玛处于哪个百分位。

用法: python3 tools/exp_mood_perm_217.py [--n 2000]
产物: output/exp_mood_perm_217.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "exp_mood_perm_217.json"
SEED = 20261113


def stats(ret: np.ndarray, mult: np.ndarray, years: float) -> tuple[float, float, float]:
    nav = np.cumprod(1 + ret * mult)
    total = nav[-1]
    cagr = total ** (1 / years) - 1
    dd = float((nav / np.maximum.accumulate(nav) - 1).min())
    return float(cagr), dd, (cagr / abs(dd) if dd else np.nan)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    args = ap.parse_args()

    emo = pd.read_parquet(ROOT / "output" / "uplimit_emo.parquet")
    emo["date"] = emo["date"].astype(str)
    q = emo["quality"].astype(float)
    mu = q.shift(1).rolling(250, min_periods=60).mean()
    sd = q.shift(1).rolling(250, min_periods=60).std()
    emo["q_roll"] = (q - mu) / sd.replace(0, np.nan)
    emo["q_rank"] = emo["q_roll"].rolling(250, min_periods=60).apply(
        lambda s: float((s.iloc[-1] > s.iloc[:-1]).mean()), raw=False)

    bt = json.loads((ROOT / "output" / "backtest.json").read_text(encoding="utf-8"))
    tr = pd.DataFrame(bt["trades_all"])
    tr = tr[tr["ret_pct"].notna()].copy()
    tr["signal_date"] = tr["signal_date"].astype(str)
    tr = tr[tr["signal_date"] >= "20200102"].sort_values("signal_date")
    m = tr.merge(emo[["date", "q_rank"]], left_on="signal_date", right_on="date", how="left")
    m = m[m["q_rank"].notna()].reset_index(drop=True)

    ret = m["ret_pct"].to_numpy() / 100.0
    qr = m["q_rank"].to_numpy()
    d0, d1 = pd.to_datetime(m["signal_date"].min()), pd.to_datetime(m["signal_date"].max())
    years = max(0.5, (d1 - d0).days / 365.25)

    # 真实方案: 按分位给 1.0 / 0.8 / 0.6
    real = np.where(qr > 2/3, 1.0, np.where(qr > 1/3, 0.8, 0.6))
    c_real, d_real, k_real = stats(ret, real, years)

    base_c, base_d, base_k = stats(ret, np.ones_like(ret), years)

    # 置换: 保持倍率集合与各档数量不变, 只打乱"谁拿到哪一档"
    rng = np.random.default_rng(SEED)
    ks, cs, ds = [], [], []
    for _ in range(args.n):
        perm = rng.permutation(real)
        c, d, k = stats(ret, perm, years)
        if np.isfinite(k):
            ks.append(k); cs.append(c); ds.append(d)
    ks = np.array(ks); cs = np.array(cs); ds = np.array(ds)

    pct_k = float((ks < k_real).mean() * 100)
    pct_c = float((cs < c_real).mean() * 100)
    res = {
        "样本": {"n": int(len(m)), "年数": round(years, 2)},
        "基准": {"年化%": round(base_c * 100, 2), "回撤%": round(base_d * 100, 2),
                 "卡玛": round(base_k, 3)},
        "真实方案": {"年化%": round(c_real * 100, 2), "回撤%": round(d_real * 100, 2),
                     "卡玛": round(k_real, 3), "均仓": round(float(real.mean()), 3)},
        "随机降仓分布": {
            "次数": len(ks),
            "卡玛_中位": round(float(np.median(ks)), 3),
            "卡玛_P5": round(float(np.percentile(ks, 5)), 3),
            "卡玛_P95": round(float(np.percentile(ks, 95)), 3),
            "卡玛_max": round(float(ks.max()), 3),
            "年化_中位%": round(float(np.median(cs)) * 100, 2),
            "回撤_中位%": round(float(np.median(ds)) * 100, 2),
            "真实卡玛分位%": round(pct_k, 2),
            "真实年化分位%": round(pct_c, 2),
        },
        "判决": ("选对了时点" if pct_k >= 95 else
                 "方向对但不显著" if pct_k >= 80 else "与随机降仓无异"),
    }
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"\n结论: 真实卡玛 {k_real:.3f} 位于随机降仓分布的 {pct_k:.1f} 百分位 → {res['判决']}")
    print(f"写入 {OUT}")


if __name__ == "__main__":
    main()
