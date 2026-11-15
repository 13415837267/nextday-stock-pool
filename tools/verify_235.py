"""第二百三十五轮: 把"评分选股"换成随机 —— 当前策略是不是最好的?

问题来源: 用户质疑"你就这么肯定现在策略就是最好的?"
已有的三条自证说明"不确定":
  · 第121轮: 8 参数随机 30 组, 30/30 正收益, 中位年化 63.6% —— 结构贡献远大于调参;
  · 第120轮: WF 5 折选出的最优参数值折间全漂移, 参数不可辨识;
  · 第234轮: GBM 样本外 IC 只有 0.049, 32 特征里 bias20 排倒数第三。
但上面三条都没回答一件事: **84% 胜率里, 有多少是"评分选出来的", 有多少是"漏斗+每天只买一只"白送的?**

本脚本用最直接的口径回答: 对每一笔真实成交, 在**同一天的同一类票**里随机选一只,
看真实选中股比随机好多少。若超额不显著, 则"评分最优"是伪命题;
若显著但远低于上限, 则"最好"也只在当前机制内成立。

池的定义(逐笔锚定, 不依赖我猜参数):
  base  = 同日 & n1_ok(次日低开可成交) & pct ∈ [-6,-3]      <- 策略信号的核心定义
  strict= base & close_pos<=0.32 & price∈[3,150] & circ_mv<=600亿 & turnover∈[2,12]
                                                             <- 再叠现行硬约束, 更贴近真实候选池
三个量:
  ① 真实选中 y 在同日池内的分位(50% = 和瞎猜一样)
  ② 配对超额 d = y_sel - mean(池内其余), 做 t 检验 + bootstrap
  ③ oracle 上限 = 池内 max y 的均值(完美选股的物理天花板)

输出 output/w235_*.csv, 控制台打印摘要。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "output" / "feat234"
TRADES = ROOT / "output" / "w232_trades.csv"
OUT = ROOT / "output"

COLS = ["ts_code", "trade_date", "y", "n1_ok", "pct", "close_pos", "circ_mv",
        "close", "turnover_rate", "amt_yi", "amp"]


def load_pool(days: set[int]) -> pd.DataFrame:
    """只读 signal 日, 拼成一张小表。"""
    parts = []
    for p in sorted(FEAT.glob("f*.parquet")):
        d = pd.read_parquet(p, columns=COLS)
        d = d[d["trade_date"].isin(days)]
        if len(d):
            parts.append(d)
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    tr = pd.read_csv(TRADES)
    tr = tr.rename(columns={"signal": "trade_date", "code": "ts_code"})
    tr["trade_date"] = tr["trade_date"].astype(int)
    days = set(tr["trade_date"].unique())
    print(f"成交 {len(tr)} 笔 / {len(days)} 个信号日")

    d = load_pool(days)
    print(f"信号日全市场样本 {len(d):,} 行")

    base = d[(d["n1_ok"] == True) & d["pct"].between(-6.0, -3.0)].copy()  # noqa: E712
    strict = base[(base["close_pos"] <= 0.32)
                  & base["close"].between(3, 150)
                  & base["circ_mv"].between(0, 600)
                  & base["turnover_rate"].between(2, 12)].copy()
    print(f"base 池 {len(base):,} 行 / strict 池 {len(strict):,} 行")

    rows = []
    for name, pool in (("base", base), ("strict", strict)):
        g = pool.groupby("trade_date")
        agg = g["y"].agg(["mean", "max", "count"]).rename(
            columns={"mean": "pool_mean", "max": "pool_max", "count": "n_pool"})
        m = tr.merge(agg, left_on="trade_date", right_index=True, how="left")
        rows.append((name, m, pool))

    # 真实 y: 用 feat234 的 y(与 trades 的 ret 同口径, 但去掉执行细节差异)
    for name, m, pool in rows:
        yt = d.set_index(["trade_date", "ts_code"])["y"]
        m["y_true"] = [yt.get(k, np.nan) for k in zip(m["trade_date"], m["ts_code"])]
        den = (m["n_pool"] - 1).clip(lower=1)
        m["pool_mean_ex"] = (m["pool_mean"] * m["n_pool"] - m["y_true"]) / den
        m["d"] = m["y_true"] - m["pool_mean_ex"]

    for name, m, pool in rows:
        ok = m.dropna(subset=["y_true", "pool_mean_ex", "n_pool"])
        ok = ok[ok["n_pool"] >= 3]
        d_ = ok["d"].values
        n = len(d_)
        se = d_.std(ddof=1) / np.sqrt(n)
        t = d_.mean() / se
        boot = np.array([np.random.choice(d_, n, replace=True).mean() for _ in range(4000)])
        print(f"\n=== 池 {name} (n={n} 笔, 池内>=3 只)")
        print(f"  真实 y         {ok['y_true'].mean():+.3f}%   (全样本 {m['y_true'].mean():+.3f}%)")
        print(f"  同日池均值      {ok['pool_mean'].mean():+.3f}%")
        print(f"  配对超额 d      {d_.mean():+.3f}pp  t={t:.2f}  p≈{(1 - stats.norm.cdf(abs(t))) * 2:.4f}")
        print(f"  bootstrap 95%CI [{np.percentile(boot, 2.5):+.3f}, {np.percentile(boot, 97.5):+.3f}]  "
              f"正比例 {(boot > 0).mean() * 100:.1f}%")
        print(f"  oracle 上限     {ok['pool_max'].mean():+.3f}%  (完美选股; 还差 {ok['pool_max'].mean() - ok['y_true'].mean():+.3f}pp)")
        print(f"  池大小 中位 {ok['n_pool'].median():.0f} / P10 {ok['n_pool'].quantile(.1):.0f} / 只有1只的天数 "
              f"{(ok['n_pool'] <= 1).sum()}")
        # 分位
        ranks = []
        for _, r in ok.iterrows():
            sub = pool[(pool["trade_date"] == r["trade_date"])]["y"]
            ranks.append((sub < r["y_true"]).mean() * 100)
        ok = ok.assign(pctile=ranks)
        print(f"  真实选中股 y 的分位: 均值 {np.mean(ranks):.1f}% / 中位 {np.median(ranks):.1f}%  "
              f"(>50 才算会选)")
        ok.to_csv(OUT / f"w235_{name}.csv", index=False, encoding="utf-8-sig")

    # 评分单调性: score 五分位 x 配对超额
    m = rows[1][1]
    m = m[m["n_pool"] >= 3].dropna(subset=["d", "score", "y_true"])
    m["q"] = pd.qcut(m["score"], 5, labels=False, duplicates="drop")
    t = m.groupby("q").agg(n=("d", "size"), score=("score", "mean"),
                           d=("d", "mean"), y=("y_true", "mean")).round(3)
    print("\n=== 评分五分位 vs 配对超额(strict 池)")
    print(t.to_string())
    t.to_csv(OUT / "w235_score_q.csv", encoding="utf-8-sig")

    # === 置换检验: 每天从同日池里随机抽一只, 完整模拟 559 笔, 重复 5000 次 ===
    name, m, pool = rows[1]
    ok = m[m["n_pool"] >= 3].dropna(subset=["y_true"])
    g = {k: v["y"].dropna().values for k, v in pool.groupby("trade_date")}
    g = {k: v for k, v in g.items() if len(v)}
    days = [d for d in ok["trade_date"].values if d in g]
    real = ok[ok["trade_date"].isin(days)]["y_true"].mean()
    rng = np.random.default_rng(235)
    sims = np.array([np.mean([g[d][rng.integers(len(g[d]))] for d in days])
                     for _ in range(5000)])
    pct = (sims < real).mean() * 100
    print(f"\n=== 置换检验(strict 池, {len(days)} 天, 每天随机抽 1 只, 5000 次)")
    print(f"  真实单笔 {real:+.3f}%   随机分布: 均值 {sims.mean():+.3f}% / "
          f"P50 {np.percentile(sims, 50):+.3f}% / P95 {np.percentile(sims, 95):+.3f}% / "
          f"max {sims.max():+.3f}%")
    print(f"  真实值处于随机分布的 **{pct:.1f} 百分位**  (p≈{(5000 - (sims < real).sum()) / 5000:.4f})")
    print(f"  5000 次随机里最好的一次也只有 {sims.max():+.3f}%, 真实 {real:+.3f}% "
          f"{'高于全部随机' if real > sims.max() else '未超过随机极值'}")
    pd.DataFrame({"sim_mean": sims}).to_csv(OUT / "w235_perm.csv", index=False)


if __name__ == "__main__":
    main()
