"""第二百一十七轮(续六): 全史 17 年检验 —— 只用手算得出的三个成分

前一步已把涨停家数/连板高度/炸板率从 panel 自算出来(2009-12 起, 4109 个交易日),
于是同一个假设可以在**全部 843 笔**交易上重测, 不再受"涨停明细只有 2020 起"的限制。

成分(全部只依赖 panel, 无需任何付费数据):
  zt_cnt   涨停家数       —— 情绪广度
  max_board 最高连板      —— 情绪高度
  zb_ratio 炸板率         —— 情绪质量(封不住 = 分歧大)

合成: mood = z(max_board) - z(zb_ratio) + z(zt_cnt), z 用**滚动 250 日**(无前视)

检验顺序:
  A 单变量秩相关(全史)
  B 三分组 + 年度
  C 作为仓位倍率 1.0/0.8/0.6 的净值, 与"随机降仓 2000 次"对比
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "exp_mood_full_217.json"


def spearman(x: pd.Series, y: pd.Series):
    ok = x.notna() & y.notna()
    if ok.sum() < 30:
        return np.nan, np.nan, int(ok.sum())
    r = x[ok].rank().corr(y[ok].rank())
    n = int(ok.sum())
    t = r * np.sqrt((n - 2) / max(1e-9, 1 - r * r))
    return float(r), float(t), n


def stats(ret: np.ndarray, mult: np.ndarray, years: float):
    nav = np.cumprod(1 + ret * mult)
    cagr = nav[-1] ** (1 / years) - 1
    dd = float((nav / np.maximum.accumulate(nav) - 1).min())
    return float(cagr), dd, (cagr / abs(dd) if dd else np.nan), float(nav[-1])


def main() -> None:
    lh = pd.read_parquet(ROOT / "data" / "limit_hist.parquet").sort_values("date").reset_index(drop=True)
    lh["date"] = lh["date"].astype(str)

    def rz(c: str, win: int = 250) -> pd.Series:
        v = lh[c].astype(float)
        mu = v.shift(1).rolling(win, min_periods=60).mean()
        sd = v.shift(1).rolling(win, min_periods=60).std()
        return (v - mu) / sd.replace(0, np.nan)

    lh["mood"] = rz("max_board") - rz("zb_ratio") + rz("zt_cnt")
    lh["mood_q"] = lh["mood"].rolling(250, min_periods=60).apply(
        lambda s: float((s.iloc[-1] > s.iloc[:-1]).mean()), raw=False)

    bt = json.loads((ROOT / "output" / "backtest.json").read_text(encoding="utf-8"))
    tr = pd.DataFrame(bt["trades_all"])
    tr = tr[tr["ret_pct"].notna()].copy()
    tr["signal_date"] = tr["signal_date"].astype(str)
    m = tr.merge(lh, left_on="signal_date", right_on="date", how="left")

    res: dict = {"样本": {"笔数": int(len(m)), "有情绪值": int(m["mood"].notna().sum()),
                        "起": m["signal_date"].min(), "止": m["signal_date"].max()},
                 "基准": {"单笔%": round(float(m["ret_pct"].mean()), 3),
                        "胜率%": round(float((m["ret_pct"] > 0).mean() * 100), 1)},
                 "A_秩相关": {}, "B_分组": [], "B_年度": [], "C_仓位": {}}

    # ---- A ----
    for c, lb in [("mood", "情绪合成(全史)"), ("max_board", "最高连板"),
                  ("zb_ratio", "炸板率"), ("zt_cnt", "涨停家数"),
                  ("yz_cnt", "一字板家数")]:
        rho, t, n = spearman(m[c], m["ret_pct"])
        res["A_秩相关"][lb] = {"rho": round(rho, 3), "t": round(t, 2), "n": n}

    sub = m[m["mood_q"].notna()].copy()
    # ---- B 三档 ----
    sub["g"] = pd.cut(sub["mood_q"], [0, 1/3, 2/3, 1.01], labels=[1, 2, 3])
    for g, gy in sub.groupby("g", observed=True):
        res["B_分组"].append({"组": int(g), "n": int(len(gy)),
                            "单笔%": round(float(gy["ret_pct"].mean()), 3),
                            "胜率%": round(float((gy["ret_pct"] > 0).mean() * 100), 1)})

    # ---- B 年度 ----
    sub["year"] = sub["signal_date"].str[:4]
    for y, gy in sub.groupby("year"):
        r = {"年": y, "n": int(len(gy))}
        for g in (1, 2, 3):
            s = gy.loc[gy["g"] == g, "ret_pct"]
            r[f"G{g}"] = round(float(s.mean()), 2) if len(s) else None
        hi, lo = gy.loc[gy["g"] == 3, "ret_pct"], gy.loc[gy["g"] == 1, "ret_pct"]
        r["高-低"] = round(float(hi.mean() - lo.mean()), 2) if len(hi) and len(lo) else None
        res["B_年度"].append(r)

    # ---- C 仓位倍率 + 置换 ----
    ret = sub["ret_pct"].to_numpy() / 100.0
    q = sub["mood_q"].to_numpy()
    d0, d1 = pd.to_datetime(sub["signal_date"].min()), pd.to_datetime(sub["signal_date"].max())
    years = max(0.5, (d1 - d0).days / 365.25)
    real = np.where(q > 2/3, 1.0, np.where(q > 1/3, 0.8, 0.6))
    cb, db, kb, nb = stats(ret, np.ones_like(ret), years)
    cr, dr, kr, nr = stats(ret, real, years)

    rng = np.random.default_rng(20261113)
    ks, cs = [], []
    for _ in range(2000):
        c, d, k, _ = stats(ret, rng.permutation(real), years)
        if np.isfinite(k):
            ks.append(k); cs.append(c)
    ks, cs = np.array(ks), np.array(cs)

    res["C_仓位"] = {
        "年数": round(years, 2),
        "基准": {"年化%": round(cb * 100, 2), "回撤%": round(db * 100, 2), "卡玛": round(kb, 3),
                 "总倍": round(nb, 1)},
        "真实1.0/0.8/0.6": {"年化%": round(cr * 100, 2), "回撤%": round(dr * 100, 2),
                            "卡玛": round(kr, 3), "总倍": round(nr, 1),
                            "均仓": round(float(real.mean()), 3)},
        "随机降仓": {"卡玛中位": round(float(np.median(ks)), 3),
                     "卡玛P95": round(float(np.percentile(ks, 95)), 3),
                     "年化中位%": round(float(np.median(cs)) * 100, 2),
                     "真实卡玛分位%": round(float((ks < kr).mean() * 100), 2),
                     "真实年化分位%": round(float((cs < cr).mean() * 100), 2)},
    }

    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"样本 {res['样本']}")
    print(f"基准 单笔 {res['基准']['单笔%']}% 胜率 {res['基准']['胜率%']}%")
    print("\n=== A. 全史秩相关 ===")
    for k, v in sorted(res["A_秩相关"].items(), key=lambda x: -abs(x[1]["rho"] or 0)):
        print(f"  {k:16s} rho={v['rho']:>6.3f} t={v['t']:>6.2f} n={v['n']}")
    print("\n=== B. 三档 ===")
    for r in res["B_分组"]:
        print(f"  组{r['组']} n={r['n']:>3} 单笔 {r['单笔%']:>6.2f}% 胜率 {r['胜率%']:>5.1f}%")
    print("\n=== B. 年度 ===")
    for r in res["B_年度"]:
        print(f"  {r['年']} n={r['n']:>3} G1={r['G1']} G2={r['G2']} G3={r['G3']} 高低差={r['高-低']}")
    print("\n=== C. 仓位 ===")
    print(json.dumps(res["C_仓位"], ensure_ascii=False, indent=2))
    print(f"\n写入 {OUT}")


if __name__ == "__main__":
    main()
