"""第233轮 · 构建每日"市场状态"序列（全部为 T 日收盘可得，无未来函数）

前 232 轮全部在"个股层面"打磨（条件/排序/执行/日内），从未系统检验过
"这一笔发生时，市场处于什么状态"。本脚本把面板按日聚合成一条状态表：
情绪（涨停/跌停家数）、波动（振幅中位数）、资金（成交额环比）、
位置（等权指数距 60 日高点）、趋势（连跌天数），供后续分箱诊断使用。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context  # noqa: E402
from src.config import load_config  # noqa: E402
from src.selector import board_of  # noqa: E402

OUT = ROOT / "output" / "w233_state.csv"


def main() -> None:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 2500
    cfg = load_config()
    boards = set(cfg["universe"]["boards"])
    ctx = prepare_context(cfg, days)

    rows = []
    for d in ctx["main_dates"]:
        df = ctx["by_date"].get(d)
        if df is None or len(df) == 0:
            continue
        # 与 regime_base 同一口径：非停牌、非 ST、上市满 21 日、所属板块
        m = (df["is_paused"] == 0) & (df["is_st"] == 0) & (df["bars"] >= 21)
        m &= df["ts_code"].map(lambda c: board_of(c) in boards).values
        s = df.loc[m]
        if len(s) < 100:
            continue
        pct = s["pct"].astype("float64")
        amp = (s["high"].astype("float64") - s["low"].astype("float64")) \
            / s["pre_close"].astype("float64").replace(0, np.nan) * 100
        rows.append({
            "date": d,
            "n": len(s),
            "up_ratio": float((pct > 0).mean()),
            "limit_up": int((pct >= 9.8).sum()),
            "limit_dn": int((pct <= -9.8).sum()),
            "med_pct": float(pct.median()),
            "med_amp": float(amp.median()),
            "amt_yi": float(s["amount"].astype("float64").sum() / 1e5),   # 千元 -> 亿
            "breadth": float(s["above_ma20"].mean()),
            "mkt_pct": float(pct.mean()),
        })
    st = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    st["idx"] = (1 + st["mkt_pct"] / 100).cumprod()
    st["vol20"] = st["mkt_pct"].rolling(20).std()
    st["ret20"] = st["idx"] / st["idx"].shift(20) - 1
    st["dd60"] = st["idx"] / st["idx"].rolling(60, min_periods=20).max() - 1
    st["amt_ratio"] = st["amt_yi"] / st["amt_yi"].rolling(20).mean()
    # 连跌天数：等权指数当日收跌则 +1，收涨清零
    streak, out = 0, []
    for v in st["mkt_pct"].values:
        streak = streak + 1 if v < 0 else 0
        out.append(streak)
    st["dn_streak"] = out
    st["dow"] = pd.to_datetime(st["date"]).dt.dayofweek
    st["dom"] = pd.to_datetime(st["date"]).dt.day
    st.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"{len(st)} 日 -> {OUT}")
    print(st[["up_ratio", "limit_up", "limit_dn", "med_amp", "amt_yi", "breadth",
              "vol20", "ret20", "dd60", "dn_streak"]].describe().round(3).to_string())


if __name__ == "__main__":
    main()
