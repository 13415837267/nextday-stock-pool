"""第227轮 · 全史复验（4201 日）：把 226 轮所有"1500 日上看起来有改进"的候选重跑。

为什么必须做：226 轮的扫描默认窗口是最近 1500 日（2020-09 ~ 2026-11），
其中 2021(+190.8%)、2022(+243.3%) 两个异常大年占权重大，窗口内搜索+窗口内
评估 = 二次过拟合。这里换成全史 17 年，并在前半段/后半段分别评估。

工程要点（上一版跑 4.5 小时无产出的教训）：
  - 每完成一个配置立刻落盘，进程被杀也不丢结果；
  - 配置数压到 6 个（实测 ctx 363s + 单次 99s ≈ 16 分钟可完成）。

判据（三条全过才算通过）：
  全史年化 >= 基准 且 全史卡玛 >= 基准 且 前半段与后半段都不低于基准
"""
from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest import prepare_context, run_backtest  # noqa: E402
from src.config import load_config  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "output" / "w227_fullhist.csv"
COLS = ["方案", "年化", "回撤", "卡玛", "笔数", "胜率", "单笔",
        "前半年化", "前半卡玛", "后半年化", "后半卡玛", "耗时"]


def brief(m: dict) -> dict:
    return {
        "年化": round(float(m.get("年化收益率%") or 0), 2),
        "回撤": round(float(m.get("最大回撤%") or 0), 2),
        "卡玛": round(float(m.get("卡玛比率") or 0), 2),
        "笔数": int(m.get("交易次数") or 0),
        "胜率": round(float(m.get("胜率%") or 0), 1),
        "单笔": round(float(m.get("平均单笔收益%") or 0), 3),
    }


def seg(res: dict, d0: str, d1: str) -> dict:
    """d0/d1 传 'HALF0'/'HALF1' 表示按曲线中点切前后半段。"""
    eq = pd.DataFrame(res["equity"]).rename(columns={"d": "date"})
    eq["date"] = eq["date"].astype(str).str.replace("-", "", regex=False)
    if d0 == "HALF0":
        mid = eq["date"].iloc[len(eq) // 2]; d0, d1 = eq["date"].iloc[0], mid
    elif d0 == "HALF1":
        mid = eq["date"].iloc[len(eq) // 2]; d0, d1 = mid, eq["date"].iloc[-1]
    s = eq[(eq.date >= d0) & (eq.date <= d1)].reset_index(drop=True)
    if len(s) < 30:
        return {"ann": 0.0, "cal": 0.0}
    nav = s["nav"].astype(float).values
    yrs = max(0.3, len(s) / 244.0)
    ann = (nav[-1] / nav[0]) ** (1 / yrs) - 1
    mdd = float((nav / pd.Series(nav).cummax().values - 1).min())
    return {"ann": round(ann * 100, 2), "cal": round(ann / abs(mdd) if mdd else 0.0, 2)}


def main() -> None:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 4201
    base = load_config()

    def mk(patch: dict) -> dict:
        c = copy.deepcopy(base)
        for k, v in patch.items():
            node = c
            parts = k.split("__")
            for p in parts[:-1]:
                node = node.setdefault(p, {})
            node[parts[-1]] = v
        return c

    cands = [
        ("基准·现行", {}),
        ("price_max→100", {"universe__price_max": 100}),
        ("turnover_max→18", {"universe__turnover_max": 18}),
        ("vol_min→0.6", {"regime__vol_min": 0.6}),
        ("blocked_frac→0.6", {"execution__blocked_frac": 0.6}),
        ("neg_bias20→0.85", {"selection__dip_weights__neg_bias20": 0.85}),
    ]

    t0 = time.time()
    ctx = prepare_context(base, days)
    print(f"ctx {time.time()-t0:.0f}s, 窗口 {days} 日\n", flush=True)

    done = []
    for name, p in cands:
        t = time.time()
        res = run_backtest(mk(p), days, ctx)
        m = brief(res["metrics"])
        a, b = seg(res, "HALF0", ""), seg(res, "HALF1", "")
        row = {"方案": name, **m,
               "前半年化": a["ann"], "前半卡玛": a["cal"],
               "后半年化": b["ann"], "后半卡玛": b["cal"],
               "耗时": round(time.time() - t)}
        done.append(row)
        pd.DataFrame(done)[COLS].to_csv(OUT, index=False, encoding="utf-8-sig")
        print(f"{name:20s} 全史 {m['年化']:7.2f}% {m['回撤']:7.2f}% 卡玛{m['卡玛']:5.2f}"
              f" | 前 {a['ann']:7.2f}%/{a['cal']:5.2f} 后 {b['ann']:7.2f}%/{b['cal']:5.2f}"
              f" | {m['笔数']}笔 {row['耗时']}s", flush=True)

    df = pd.DataFrame(done)
    b0 = df.iloc[0]
    print("\n=== 判决（全史 + 前后半段三重） ===")
    for _, r in df.iloc[1:].iterrows():
        okk = (r["年化"] >= b0["年化"] and r["卡玛"] >= b0["卡玛"]
               and r["前半年化"] >= b0["前半年化"] and r["后半年化"] >= b0["后半年化"])
        print(f"{r['方案']:20s} {'通过' if okk else '否'}  "
              f"Δ年化{r['年化']-b0['年化']:+6.2f}pp  Δ卡玛{r['卡玛']-b0['卡玛']:+5.2f}  "
              f"Δ前{r['前半年化']-b0['前半年化']:+6.2f}pp  Δ后{r['后半年化']-b0['后半年化']:+6.2f}pp")


if __name__ == "__main__":
    main()
