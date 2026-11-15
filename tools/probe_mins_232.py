"""第232轮 · 信号日日内结构 → 次日修复幅度

日线只知道"跌了多少"，分钟线知道"什么时候跌的"。本脚本对 2500 日窗口的 559 笔
逐笔算信号日的日内结构，看哪个结构与那笔的收益相关。

特征（全部来自 T 日，T 日收盘后即完全可知，无未来函数）:
  tail30   尾盘 30 分钟涨跌 %   (close / close@14:30 - 1)
  pm       午后涨跌 %           (close / close@11:30 - 1)
  am       早盘涨跌 %           (close@11:30 / open - 1)
  low_at   全天最低点位置 0~1   (0=开盘 1=收盘)
  tail_vol 尾盘 30 分钟成交量占全天
  am_vol   开盘 60 分钟成交量占全天（恐慌抛压集中度）
  vwap_dev 收盘相对当日 VWAP 的偏离 %
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MINS = ROOT / "output" / "mins_232"
OUT = ROOT / "output" / "w232_feat.csv"


def feats(code: str, d: str) -> dict | None:
    f = MINS / f"{code}_{d}.csv.gz"
    if not f.exists():
        return None
    m = pd.read_csv(f)
    if len(m) < 60:
        return None
    m["hm"] = (m["trade_time"].astype(str).str[-4:]).astype(int)
    m = m.sort_values("hm").reset_index(drop=True)
    o = float(m["open"].iloc[0])
    c = float(m["close"].iloc[-1])
    if o <= 0 or c <= 0:
        return None
    # 定位 11:30 与 14:30 之前最后一根的收盘
    def px_before(hm: int) -> float | None:
        s = m[m["hm"] <= hm]
        return float(s["close"].iloc[-1]) if len(s) else None

    c1130, c1430 = px_before(1130), px_before(1430)
    if not c1130 or not c1430:
        return None
    vol = m["vol"].astype(float)
    amt = m["amount"].astype(float)
    tot_v = float(vol.sum())
    if tot_v <= 0:
        return None
    tail_v = float(vol[m["hm"] > 1430].sum()) / tot_v
    am_v = float(vol[m["hm"] <= 1030].sum()) / tot_v
    vwap = float(amt.sum() / tot_v) if tot_v else np.nan
    lo_idx = int(np.argmin(m["low"].astype(float).values))
    return {
        "code": code, "signal": d,
        "tail30": round((c / c1430 - 1) * 100, 3),
        "pm": round((c / c1130 - 1) * 100, 3),
        "am": round((c1130 / o - 1) * 100, 3),
        "low_at": round(lo_idx / max(len(m) - 1, 1), 3),
        "tail_vol": round(tail_v, 3),
        "am_vol": round(am_v, 3),
        "vwap_dev": round((c / vwap - 1) * 100, 3) if vwap and vwap > 0 else np.nan,
        "n_bar": len(m),
    }


def main() -> None:
    t = pd.read_csv(ROOT / "output" / "w232_trades.csv", dtype={"signal": str})
    rows, miss = [], 0
    for _, r in t.iterrows():
        f = feats(r["code"], r["signal"])
        if f is None:
            miss += 1
            continue
        rows.append({**f, "ret": float(r["ret"]), "pool": r["pool"],
                     "score": r["score"], "pool_n": r["pool_n"]})
    d = pd.DataFrame(rows)
    d.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"特征 {len(d)} 笔, 缺 {miss}\n", flush=True)

    cols = ["tail30", "pm", "am", "low_at", "tail_vol", "am_vol", "vwap_dev"]
    print("=== 与单笔收益的相关 ===")
    for c in cols:
        s = d[[c, "ret"]].dropna()
        if len(s) < 50:
            continue
        pear = s[c].corr(s["ret"])
        spear = s[c].corr(s["ret"], method="spearman")
        n = len(s)
        tval = pear * np.sqrt((n - 2) / max(1e-9, 1 - pear ** 2))
        print(f"{c:9s} n={n:4d} pearson={pear:+.3f} spearman={spear:+.3f} t={tval:+.2f}")

    print("\n=== 单变量五档（按特征升序）===")
    for c in cols:
        s = d[[c, "ret"]].dropna().sort_values(c)
        if len(s) < 50:
            continue
        s["bin"] = pd.qcut(s[c], 5, labels=False, duplicates="drop")
        g = s.groupby("bin")["ret"].agg(["count", "mean"]).round(3)
        seq = " / ".join(f"{m:+.2f}" for m in g["mean"])
        print(f"{c:9s} 低→高: {seq}  (前50% {s[s.bin<2.5].ret.mean():+.2f} vs "
              f"后50% {s[s.bin>=2.5].ret.mean():+.2f})")

    print("\n=== 尾盘占比（tail30 占全天跌幅的比例）===")
    d["day"] = d["am"] + d["pm"]
    d["tail_share"] = np.where(d["day"].abs() > 1e-6, d["tail30"] / d["day"], np.nan)
    s = d[["tail_share", "ret"]].replace([np.inf, -np.inf], np.nan).dropna()
    print(f"n={len(s)} pearson={s['tail_share'].corr(s['ret']):+.3f} "
          f"spearman={s['tail_share'].corr(s['ret'], method='spearman'):+.3f}")


if __name__ == "__main__":
    main()
