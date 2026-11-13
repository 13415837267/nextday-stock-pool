"""第二百一十七轮(续二): 情绪质量分是不是"大盘"的马甲 —— 决定性检验

上一版发现: 把 T 日的涨停结构压成一个"情绪质量分"(连板高度↑ 烂板率↓ 游资占比↑
题材集中度↑ 封单比↓), 主策略的单笔收益从 0.66% 单调升到 2.27%。但有两个致命疑问:

  Q1 会不会只是"大盘好"的代理? 情绪热的时候大盘本来就在涨, 深跌股反弹自然容易。
     若是, 它与现有的"大盘暴跌日次日"择时高度重合, 没有增量。
  Q2 分年度站得住吗? 188 轮的教训是"最优组合全靠 2024"。

本脚本回答这两问:

  A. Spearman 秩相关(对异常值稳健), 给每个情绪指标的单独解释力。
  B. 控制大盘: 按 T 日全市场中位涨跌幅分 3 档, 每档内再按质量分 2 组。
     若"高-低"差在各档内**都为正**, 说明是独立维度; 若只在某一档为正, 就是马甲。
  C. 滚动标准化(只用过去 250 日)重算质量分, 消除全样本 z 的前视。
  D. 年度 + 分档交叉表。

用法: python3 tools/exp_mood_217.py
产物: output/exp_mood_217.json, output/mkt_med_217.parquet
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "exp_mood_217.json"
CACHE = ROOT / "output" / "mkt_med_217.parquet"


def market_state() -> pd.DataFrame:
    """全市场每日中位/均值涨跌幅 —— 作为"大盘状态"的控制变量。"""
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    rows = []
    for f in sorted(glob.glob(str(ROOT / "data" / "panel" / "*.csv.gz"))):
        d = Path(f).name[:8]
        if d < "20200101":
            continue
        try:
            x = pd.read_csv(f, usecols=["pct", "is_paused"])
        except Exception:  # noqa: BLE001
            continue
        v = x.loc[x["is_paused"] == 0, "pct"].dropna()
        if len(v) < 100:
            continue
        rows.append({"date": d, "mkt_med": float(v.median()),
                     "mkt_mean": float(v.mean()), "breadth": float((v > 0).mean())})
    df = pd.DataFrame(rows)
    df.to_parquet(CACHE, index=False)
    return df


def spearman(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    ok = x.notna() & y.notna()
    if ok.sum() < 30:
        return np.nan, np.nan, int(ok.sum())
    r = x[ok].rank().corr(y[ok].rank())
    n = int(ok.sum())
    t = r * np.sqrt((n - 2) / max(1e-9, 1 - r * r))
    return float(r), float(t), n


def main() -> None:
    emo = pd.read_parquet(ROOT / "output" / "uplimit_emo.parquet")
    bt = json.loads((ROOT / "output" / "backtest.json").read_text(encoding="utf-8"))
    tr = pd.DataFrame(bt["trades_all"])
    tr = tr[tr["ret_pct"].notna()].copy()
    tr["signal_date"] = tr["signal_date"].astype(str)
    tr = tr[tr["signal_date"] >= "20200102"]

    mkt = market_state()
    mkt["date"] = mkt["date"].astype(str)

    # 滚动标准化后的质量分: 只用过去 250 日, 无前视
    def rz(c: str, win: int = 250) -> pd.Series:
        v = emo[c].astype(float)
        mu = v.shift(1).rolling(win, min_periods=60).mean()
        sd = v.shift(1).rolling(win, min_periods=60).std()
        return (v - mu) / sd.replace(0, np.nan)

    emo["quality_roll"] = (rz("max_board") - rz("lan_ratio") + rz("youzi_ratio")
                           + rz("plate_top") - rz("fd_rate"))

    m = tr.merge(emo, left_on="signal_date", right_on="date", how="left")
    m = m.merge(mkt, left_on="signal_date", right_on="date", how="left", suffixes=("", "_mkt"))
    m["mkt5"] = m["mkt_med"].rolling(5, min_periods=1).mean() if False else m["mkt_med"]

    res: dict = {
        "基准": {"n": int(len(m)), "平均单笔%": round(float(m["ret_pct"].mean()), 3),
                 "胜率%": round(float((m["ret_pct"] > 0).mean() * 100), 1)},
        "A_秩相关": {}, "B_控制大盘": {}, "C_滚动分": {}, "D_年度": [],
    }

    # ---- A. 单变量秩相关 ----
    dims = {
        "quality": "情绪质量分(全样本z)", "quality_roll": "情绪质量分(滚动z)",
        "max_board": "最高连板", "lan_ratio": "烂板率", "youzi_ratio": "游资占比",
        "plate_top": "最热题材分", "plate_n": "活跃题材数", "fd_rate": "封单/成交比",
        "limit_cnt": "涨停家数", "yizi_ratio": "一字占比", "mkt_med": "大盘中位涨幅",
    }
    for c, lb in dims.items():
        rho, t, n = spearman(m[c], m["ret_pct"])
        res["A_秩相关"][lb] = {"rho": round(rho, 3), "t": round(t, 2), "n": n}

    # ---- B. 控制大盘: 大盘 3 档 × 质量 2 档 ----
    m["mkt_lab"] = pd.qcut(m["mkt_med"], 3, labels=["大盘弱", "大盘中", "大盘强"], duplicates="drop")
    m["q2"] = pd.qcut(m["quality"], 2, labels=["质量低", "质量高"], duplicates="drop")
    tab = []
    for ml in ["大盘弱", "大盘中", "大盘强"]:
        row = {"大盘档": ml}
        for ql in ["质量低", "质量高"]:
            s = m.loc[(m["mkt_lab"] == ml) & (m["q2"] == ql), "ret_pct"]
            row[f"{ql}_n"] = int(len(s))
            row[f"{ql}_单笔"] = round(float(s.mean()), 2) if len(s) else None
        row["高-低"] = (round(row["质量高_单笔"] - row["质量低_单笔"], 2)
                        if row["质量高_单笔"] is not None and row["质量低_单笔"] is not None else None)
        tab.append(row)
    res["B_控制大盘"] = tab

    # ---- B2. 偏相关: 剔除大盘后质量分还剩多少 ----
    ok = m[["quality", "mkt_med", "ret_pct"]].dropna()
    if len(ok) > 40:
        rq = ok["quality"].rank()
        rm = ok["mkt_med"].rank()
        ry = ok["ret_pct"].rank()
        rqy, rmy, rqm = rq.corr(ry), rm.corr(ry), rq.corr(rm)
        partial = (rqy - rmy * rqm) / np.sqrt(max(1e-9, (1 - rmy ** 2) * (1 - rqm ** 2)))
        res["B2_偏相关"] = {"raw_rho": round(float(rqy), 3),
                            "mkt_rho": round(float(rmy), 3),
                            "q_vs_mkt_rho": round(float(rqm), 3),
                            "partial_rho": round(float(partial), 3),
                            "n": int(len(ok))}

    # ---- C. 滚动质量分(无前视) ----
    res["C_滚动分"] = {}
    for c, lb in [("quality", "全样本z"), ("quality_roll", "滚动z(无前视)")]:
        st = []
        sub = m[m[c].notna()]
        if len(sub) >= 60:
            lab = pd.qcut(sub[c], 3, labels=False, duplicates="drop")
            for k in sorted(pd.Series(lab).dropna().unique()):
                s = sub.loc[lab == k, "ret_pct"]
                st.append({"组": int(k) + 1, "n": int(len(s)),
                           "单笔": round(float(s.mean()), 2),
                           "胜率": round(float((s > 0).mean() * 100), 1)})
        res["C_滚动分"][lb] = st

    # ---- D. 年度 ----
    m["year"] = m["signal_date"].str[:4]
    m["q3"] = pd.qcut(m["quality"], 3, labels=False, duplicates="drop")
    for y, gy in m.groupby("year"):
        r = {"年": y, "n": int(len(gy))}
        for k in (0, 1, 2):
            s = gy.loc[gy["q3"] == k, "ret_pct"]
            r[f"G{k+1}"] = round(float(s.mean()), 2) if len(s) else None
        hi, lo = gy.loc[gy["q3"] == 2, "ret_pct"], gy.loc[gy["q3"] == 0, "ret_pct"]
        r["高-低"] = (round(float(hi.mean() - lo.mean()), 2)
                      if len(hi) and len(lo) else None)
        res["D_年度"].append(r)

    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"基准 n={res['基准']['n']} 单笔 {res['基准']['平均单笔%']}% 胜率 {res['基准']['胜率%']}%")
    print("\n=== A. 秩相关(与单笔收益) ===")
    for k, v in sorted(res["A_秩相关"].items(), key=lambda x: -abs(x[1]["rho"] or 0)):
        print(f"  {k:22s} rho={v['rho']:>6.3f}  t={v['t']:>6.2f}  n={v['n']}")
    print("\n=== B. 控制大盘 ===")
    for r in res["B_控制大盘"]:
        print(f"  {r['大盘档']}  低 n={r['质量低_n']:>3} {r['质量低_单笔']:>6}  |  高 n={r['质量高_n']:>3} {r['质量高_单笔']:>6}  |  差 {r['高-低']}")
    print("\n=== B2. 偏相关(剔除大盘) ===")
    print(" ", res.get("B2_偏相关"))
    print("\n=== C. 滚动 vs 全样本 ===")
    for k, v in res["C_滚动分"].items():
        print(f"  {k}: " + "  ".join(f"G{x['组']}(n={x['n']}) {x['单笔']}%/{x['胜率']}%" for x in v))
    print("\n=== D. 年度 ===")
    for r in res["D_年度"]:
        print(f"  {r['年']} n={r['n']:>3}  G1={r['G1']}  G2={r['G2']}  G3={r['G3']}  高低差={r['高-低']}")
    print(f"\n写入 {OUT}")


if __name__ == "__main__":
    main()
