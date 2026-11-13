"""第二百一十七轮(续): 涨停维度对主策略有没有用 —— 交易级对齐检验

问题
----
主策略买的是"距 250 日高点跌 65~80% + 次日深低开"的票。这类票本身极少涨停,
所以"个股昨日涨停"这种条件大概率没有样本。真正可能有用的是**市场情绪维度**:
涨停家数、最高连板、炸板率、一字板占比 —— 它们刻画的是"今天市场有多热/多冷",
而主策略赚的正是"冰点接货"的钱。

检验设计
-------
1) 从 output/backtest.json 取 trades_all(需 BT_FULL_TRADES=1), 只留真实成交。
2) 从 data/uplimit/*.csv.gz 聚合出**日频情绪序列**(每个交易日一行)。
3) 每笔交易按 signal_date(T 日收盘后决策日) 对齐情绪值 —— T 日盘后公布, 与决策同口径。
4) 按情绪指标三分位分组, 比较平均单笔/胜率/净值倍数。
5) 同时诊断"个股维度"的覆盖率: 主策略买的票, T 日在涨停名单里的比例有多少。
   若覆盖率极低, 个股维度直接判死, 不必再试。

样本边界(必须写在结论里)
-----------------------
涨停明细 2020-01-02 起, 因此只能对齐 2020 后的交易。全史 3664 日口径下约 1/3 的交易可对齐;
1500 日口径(≈2021 起)可全部对齐。**分组后每组样本 20~40 笔, 只能看方向, 不能谈显著性。**

用法: BT_FULL_TRADES=1 python3 -m src.backtest --days 3664
      python3 tools/exp_uplimit_217.py [--min-date 20200102]
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "exp_uplimit_217.json"


def load_emotion() -> pd.DataFrame:
    """把逐日涨停明细压成日频情绪序列。"""
    files = sorted(glob.glob(str(ROOT / "data" / "uplimit" / "*.csv.gz")))
    if not files:
        raise SystemExit("没有 data/uplimit 数据, 先跑 tools/fetch_uplimit_217.py")
    frames = []
    for f in files:
        try:
            x = pd.read_csv(f, dtype={"stock_code": str, "plate_code": str})
        except Exception:  # noqa: BLE001
            continue
        # 日期取文件名而不是 date1 列: 个别日 date1 全空会被 pandas 推成 float
        x["date"] = Path(f).name[:8]
        frames.append(x)
    df = pd.concat(frames, ignore_index=True)

    t = df["up_limit_type"].fillna("")
    df["is_yizi"] = t.str.startswith("一")
    df["is_lan"] = t.str.startswith("烂")
    df["board"] = pd.to_numeric(df["up_limit_keep_times"], errors="coerce").fillna(1)
    df["fengdan_rate"] = pd.to_numeric(df["fengdan_rate"], errors="coerce")
    df["plate_score"] = pd.to_numeric(df["plate_score"], errors="coerce")

    g = df.groupby("date")
    emo = pd.DataFrame({
        "limit_cnt": g.size(),                                   # 涨停家数
        "max_board": g["board"].max(),                           # 最高连板
        "avg_board": g["board"].mean(),                          # 平均连板
        "yizi_cnt": g["is_yizi"].sum(),                          # 一字板家数
        "lan_cnt": g["is_lan"].sum(),                            # 烂板家数
        "youzi_cnt": g["market_type"].apply(lambda s: (s == "游资").sum()),
        "jijin_cnt": g["market_type"].apply(lambda s: (s == "基金").sum()),
        "plate_top": g["plate_score"].max(),                     # 最热题材分
        "plate_n": g["plate_name"].nunique(),                    # 活跃题材数
        "fd_rate": g["fengdan_rate"].apply(lambda s: pd.to_numeric(s, errors="coerce").mean()),
    }).reset_index().sort_values("date").reset_index(drop=True)

    emo["lan_ratio"] = emo["lan_cnt"] / emo["limit_cnt"].clip(lower=1)   # 烂板率
    emo["yizi_ratio"] = emo["yizi_cnt"] / emo["limit_cnt"].clip(lower=1)  # 一字占比
    emo["youzi_ratio"] = emo["youzi_cnt"] / emo["limit_cnt"].clip(lower=1)

    # 情绪的水平 vs 变化: 5 日均值与 20 日分位
    for c in ["limit_cnt", "max_board"]:
        emo[f"{c}_ma5"] = emo[c].rolling(5, min_periods=3).mean()
        emo[f"{c}_ma20"] = emo[c].rolling(20, min_periods=10).mean()
        emo[f"{c}_q20"] = emo[c].rolling(60, min_periods=20).apply(
            lambda s: (s.iloc[-1] > s.iloc[:-1]).mean(), raw=False)
    emo["limit_cnt_chg"] = emo["limit_cnt"] / emo["limit_cnt_ma20"] - 1

    # 情绪质量分: 连板高度↑ / 烂板率↓ / 游资占比↑ / 题材集中度↑ / 封单比↓
    def z(c):
        v = emo[c].astype(float)
        sd = v.std()
        return (v - v.mean()) / (sd if sd and sd > 0 else 1)

    emo["quality"] = (z("max_board") - z("lan_ratio") + z("youzi_ratio")
                      + z("plate_top") - z("fd_rate"))
    emo.to_parquet(ROOT / "output" / "uplimit_emo.parquet", index=False)
    return emo, df


def load_trades(min_date: str) -> pd.DataFrame:
    p = ROOT / "output" / "backtest.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    tr = d.get("trades_all")
    if not tr:
        raise SystemExit("backtest.json 里没有 trades_all, 请用 BT_FULL_TRADES=1 重跑回测")
    df = pd.DataFrame(tr)
    df = df[df["ret_pct"].notna()].copy()
    df["signal_date"] = df["signal_date"].astype(str)
    df["code6"] = df["code"].astype(str).str[:6]   # 002235.SZ -> 002235
    df = df[df["signal_date"] >= min_date]
    return df


def bucket_stats(x: pd.Series, y: pd.Series, q: int = 3) -> list[dict]:
    """按 x 分位分组, 比较 y 的均值/胜率/累计。"""
    ok = x.notna() & y.notna()
    x, y = x[ok], y[ok]
    if len(x) < 12:
        return []
    try:
        lab = pd.qcut(x, q, labels=False, duplicates="drop")
    except ValueError:
        return []
    out = []
    for k in sorted(pd.Series(lab).dropna().unique()):
        m = lab == k
        yy = y[m]
        out.append({
            "组": int(k) + 1,
            "n": int(m.sum()),
            "x范围": f"{x[m].min():.3g}~{x[m].max():.3g}",
            "平均单笔%": round(float(yy.mean()), 3),
            "胜率%": round(float((yy > 0).mean() * 100), 1),
            "累计倍": round(float(np.prod(1 + yy / 100)), 3),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-date", default="20200102")
    args = ap.parse_args()

    emo, raw = load_emotion()
    tr = load_trades(args.min_date)
    print(f"涨停数据覆盖 {emo['date'].min()} ~ {emo['date'].max()} ({len(emo)} 天)")
    print(f"可对齐交易 {len(tr)} 笔 ({tr['signal_date'].min()} ~ {tr['signal_date'].max()})")

    # ---- 覆盖率诊断: 主策略买的票, T 日在不在涨停名单 ----
    lim = raw.drop_duplicates(["date", "stock_code"]).assign(in_limit=1)
    m = tr.merge(lim[["date", "stock_code", "in_limit", "board", "up_limit_time",
                      "market_type", "plate_name"]],
                 left_on=["signal_date", "code6"], right_on=["date", "stock_code"], how="left")
    cover = float(m["in_limit"].fillna(0).mean() * 100)

    # 近 20 日涨停次数(个股维度)
    by_code = raw.groupby("stock_code")["date"].apply(lambda s: sorted(s.tolist())).to_dict()
    days = sorted(emo["date"])
    dpos = {d: i for i, d in enumerate(days)}

    def cnt_20(code: str, d: str) -> float:
        ds = by_code.get(code)
        if not ds or d not in dpos:
            return np.nan
        i = dpos[d]
        lo = days[max(0, i - 19)]
        return float(sum(1 for x in ds if lo <= x <= d))

    m["limit_cnt_20d"] = [cnt_20(c, d) for c, d in zip(m["code6"], m["signal_date"])]
    m = m.merge(emo, left_on="signal_date", right_on="date", how="left", suffixes=("", "_emo"))

    # ---- 分组检验 ----
    dims = ["quality", "limit_cnt", "max_board", "lan_ratio", "yizi_ratio", "youzi_ratio",
            "plate_top", "plate_n", "limit_cnt_ma5", "limit_cnt_chg", "limit_cnt_q20",
            "fd_rate", "limit_cnt_20d"]
    labels = {
        "limit_cnt": "T日涨停家数", "max_board": "T日最高连板", "lan_ratio": "烂板率",
        "yizi_ratio": "一字板占比", "youzi_ratio": "游资标签占比", "plate_top": "最热题材分",
        "plate_n": "活跃题材数", "limit_cnt_ma5": "涨停家数5日均", "limit_cnt_chg": "涨停家数/20日均",
        "limit_cnt_q20": "涨停家数60日分位", "fd_rate": "平均封单/成交比",
        "limit_cnt_20d": "个股近20日涨停次数",
        "quality": "情绪质量分(合成)",
    }
    result = {
        "覆盖": {
            "涨停数据天数": len(emo),
            "可对齐交易": len(tr),
            "T日涨停覆盖率%": round(cover, 2),
            "个股涨停次数非空%": round(float(m["limit_cnt_20d"].notna().mean() * 100), 1),
        },
        "基准": {
            "平均单笔%": round(float(m["ret_pct"].mean()), 3),
            "胜率%": round(float((m["ret_pct"] > 0).mean() * 100), 1),
            "n": len(m),
        },
        "分组": {},
    }
    for c in dims:
        st = bucket_stats(m[c], m["ret_pct"])
        if st:
            result["分组"][labels[c]] = st

    # ---- 年度稳定性: 合成分的三档是否每年都站得住(188 轮"最优全靠 2024"的教训) ----
    if m["quality"].notna().sum() >= 60:
        m["q_lab"] = pd.qcut(m["quality"], 3, labels=False, duplicates="drop")
        m["year"] = m["signal_date"].str[:4]
        rows = []
        for y, gy in m.groupby("year"):
            r = {"年": y, "n": int(len(gy))}
            for k in (0, 1, 2):
                ss = gy.loc[gy["q_lab"] == k, "ret_pct"]
                r[f"低{k+1}" if k == 0 else (f"中{k+1}" if k == 1 else "高3")] = (
                    round(float(ss.mean()), 2) if len(ss) else None)
            rows.append(r)
        result["情绪分年度"] = rows

    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== 覆盖率 ===")
    for k, v in result["覆盖"].items():
        print(f"  {k}: {v}")
    print(f"\n=== 基准: 平均 {result['基准']['平均单笔%']}% / 胜率 {result['基准']['胜率%']}% / n={result['基准']['n']}")
    for name, st in result["分组"].items():
        print(f"\n{name}")
        for r in st:
            print(f"  组{r['组']} n={r['n']:>3} x={r['x范围']:<18} 单笔{r['平均单笔%']:>7.2f}% 胜率{r['胜率%']:>5.1f}% 累计{r['累计倍']:.2f}")
    print(f"\n写入 {OUT}")


if __name__ == "__main__":
    main()
