"""第229轮 D · 新条件探查：先看"变量与收益的关系"，再决定加不加条件

A/B 两批已经把 13 个**现有**条件的 52 个档位在全史跑完，结论是走廊很窄、无冗余。
再拧现有阈值已无信息量，本脚本转向**从未进入过模型的观测量**：

  1) circ_mv  流通市值(亿) = amount / (turnover_rate/100)   —— 由面板恒等式反推, 无需外部数据
  2) amp      当日振幅% = (high-low)/pre_close*100
  3) gap      当日开盘缺口% = (open/pre_close-1)*100
  4) ddown    截至信号日的连续收阴天数（用 per_code 历史算）

方法：跑一次基准（952 笔），对每笔回查信号日面板行，按变量五分位分箱，
      比较**平均单笔 / 胜率 / 笔数**。先看信号形状；形状平的东西不进下一轮扫描。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["BT_FULL_TRADES"] = "1"

from src.backtest import prepare_context, run_backtest  # noqa: E402
from src.config import load_config  # noqa: E402

OUT = ROOT / "output" / "w229_probe.csv"
PANEL = ROOT / "data" / "panel"


def load_day(d: str) -> pd.DataFrame:
    p = PANEL / f"{d}.csv.gz"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    df["circ_mv"] = df["amount"] / (df["turnover_rate"].replace(0, float("nan")) / 100) / 1e8
    df["amp"] = (df["high"] - df["low"]) / df["pre_close"].replace(0, float("nan")) * 100
    df["gap"] = (df["open"] / df["pre_close"].replace(0, float("nan")) - 1) * 100
    return df.set_index("ts_code")[["circ_mv", "amp", "gap", "close", "pct", "amount", "turnover_rate"]]


def ddown(per_code, code: str, date: str) -> int:
    """连续收阴天数(当日为第1天)。"""
    h = per_code.get(code)
    if h is None:
        return 0
    d = h[h.index <= date] if hasattr(h.index, "dtype") else h
    try:
        s = d.loc[:date, "pct"].tail(20).values
    except Exception:
        return 0
    n = 0
    for v in s[::-1]:
        if v < 0:
            n += 1
        else:
            break
    return n


def main() -> None:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 4201
    cfg = load_config()
    ctx = prepare_context(cfg, days)
    print(f"ctx ok, {len(ctx['main_dates'])} 日", flush=True)

    res = run_backtest(cfg, None, ctx)
    m = res["metrics"]
    print(f"基准 年化{m['年化收益率%']}% 卡玛{m['卡玛比率']} 单笔{m['平均单笔收益%']}% {m['交易次数']}笔\n",
          flush=True)

    rows = []
    cache: dict[str, pd.DataFrame] = {}
    for t in res["trades_all"]:
        if t.get("ret_pct") is None:
            continue
        d, code = str(t["signal_date"]), t["code"]
        if d not in cache:
            cache[d] = load_day(d)
        day = cache[d]
        if code not in day.index:
            continue
        r = day.loc[code]
        rows.append({"signal_date": d, "year": d[:4], "code": code, "ret": t["ret_pct"],
                     "circ_mv": float(r["circ_mv"]), "amp": float(r["amp"]),
                     "gap": float(r["gap"]), "close_pos": 0, "pct": float(r["pct"]),
                     "amount_yi": float(r["amount"]) / 1e8, "turnover": float(r["turnover_rate"]),
                     "ddown": ddown(ctx["per_code"], code, d)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"样本 {len(df)} 笔\n", flush=True)

    # 逐年
    y = df.groupby("year")["ret"].agg(["count", "mean"])
    print("=== 基准逐年 ===")
    for yr, r in y.iterrows():
        print(f"  {yr} {int(r['count']):4d}笔 {r['mean']:+.3f}%", flush=True)

    for col in ["circ_mv", "amp", "gap", "ddown", "amount_yi", "turnover"]:
        print(f"\n=== {col} 五分位 ===", flush=True)
        try:
            q = pd.qcut(df[col], 5, duplicates="drop")
        except Exception as e:
            print(f"  分箱失败 {e}")
            continue
        g = df.groupby(q, observed=True)["ret"].agg(["count", "mean", lambda s: (s > 0).mean() * 100])
        g.columns = ["笔数", "平均单笔", "胜率"]
        print(g.round(3).to_string(), flush=True)

    # 市值 × 收益 更细的十档
    print("\n=== circ_mv 十档 ===", flush=True)
    q = pd.qcut(df["circ_mv"], 10, duplicates="drop")
    g = df.groupby(q, observed=True)["ret"].agg(["count", "mean", lambda s: (s > 0).mean() * 100])
    g.columns = ["笔数", "平均单笔", "胜率"]
    print(g.round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
