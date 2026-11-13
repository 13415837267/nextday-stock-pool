"""第二百一十七轮(续五): 用行情面板自算 17 年涨停结构 —— 绕开"只有 7 年"的死限

为什么这一步最关键
----------------
zzshare 的涨停明细(带游资标签/题材/封单/涨停原因)只有 2020 年起, 只能对齐 411 笔交易。
但"情绪质量分"里最重的三个成分——**涨停家数、连板高度、炸板率**——根本不需要那份明细:
面板里 `close == high_limit` 就是涨停, 连续 N 日为涨停就是 N 连板, 盘中触及但收盘未封就是炸板。
这三样可以从 2009-12 算到现在, 覆盖全部 4108 个交易日。

于是检验样本从 411 笔扩到 800+ 笔, 从"近 6.3 年"扩到"近 17 年"。
如果全史也成立, 这个维度才算真正通过检验; 若只在 2020 后成立, 就是窗口产物。

口径
----
涨停: close >= high_limit - 0.005 (涨停价四舍五入到分, 留半分容差)
炸板: high >= high_limit - 0.005 且 close < high_limit - 0.005 (盘中触板未封)
一字: 涨停 且 low >= high_limit - 0.005 (全天未离开涨停价)
连板: 按 ts_code 维护"连续涨停天数", 当日非涨停则归零; 需剔除停牌(沿用前值)
排除: is_paused=1 与 is_st=1 不计入(与主策略的股票池口径一致)

产物: data/limit_hist.parquet (date, zt_cnt, zb_cnt, yz_cnt, max_board, avg_board, zb_ratio)
用法: python3 tools/build_limit_hist_217.py
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "limit_hist.parquet"
TOL = 0.005


def main() -> None:
    files = sorted(glob.glob(str(ROOT / "data" / "panel" / "*.csv.gz")))
    print(f"面板 {len(files)} 个交易日")

    prev_streak: dict[str, int] = {}
    rows = []
    for i, f in enumerate(files, 1):
        d = Path(f).name[:8]
        try:
            x = pd.read_csv(f, usecols=["ts_code", "open", "high", "low", "close",
                                        "high_limit", "is_paused", "is_st"])
        except Exception:  # noqa: BLE001
            continue
        if len(x) == 0:
            continue
        live = x[(x["is_paused"] == 0) & (x["is_st"] == 0)].copy()
        hl = live["high_limit"].astype(float)
        cl = live["close"].astype(float)
        hi = live["high"].astype(float)
        lo = live["low"].astype(float)

        zt = cl >= hl - TOL
        zb = (hi >= hl - TOL) & (cl < hl - TOL)
        yz = zt & (lo >= hl - TOL)

        # 连板: 逐股累计
        streak = {}
        for code, is_zt in zip(live["ts_code"].astype(str), zt.to_numpy()):
            streak[code] = prev_streak.get(code, 0) + 1 if is_zt else 0
        prev_streak.update(streak)

        zt_codes = live.loc[zt.to_numpy(), "ts_code"].astype(str).to_list()
        boards = np.array([streak[c] for c in zt_codes], dtype=float) if zt_codes else np.array([])
        rows.append({
            "date": d,
            "zt_cnt": int(zt.sum()),
            "zb_cnt": int(zb.sum()),
            "yz_cnt": int(yz.sum()),
            "max_board": float(boards.max()) if len(boards) else 0.0,
            "avg_board": float(boards.mean()) if len(boards) else 0.0,
            "zt_n2": int((boards >= 2).sum()) if len(boards) else 0,   # 2 板及以上家数
        })
        if i % 500 == 0:
            print(f"  {i}/{len(files)} {d}")

    df = pd.DataFrame(rows)
    df["zb_ratio"] = df["zb_cnt"] / (df["zt_cnt"] + df["zb_cnt"]).clip(lower=1)
    df["yz_ratio"] = df["yz_cnt"] / df["zt_cnt"].clip(lower=1)
    df.to_parquet(OUT, index=False)

    print(f"\n覆盖 {df['date'].min()} ~ {df['date'].max()} 共 {len(df)} 天")
    print(df[["zt_cnt", "zb_cnt", "max_board", "zb_ratio"]].describe().round(2).to_string())
    print(f"\n写入 {OUT}")


if __name__ == "__main__":
    main()
