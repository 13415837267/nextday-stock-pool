"""第262轮: 风格背离与资金集中度能否当仓位开关

样本: output/backtest_long.json (3000 日, 2014-07 ~ 2026-11, 714 笔已平仓交易)
变量(只用截至信号日收盘的数据):
  spread     = 成交额加权收益 - 等权收益
  spread_top = 成交额前100只收益均值 - 全市场收益中位数
  conc       = 前100只成交额占比 - 其60日均值
门控: 信号值进入历史高分位/低分位时, 下一笔买入按 0 / 0.5 系数执行。
     扩窗分位 = 阈值只用该笔之前的历史, 无前视。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.selector import board_of

PANEL = ROOT / "data/panel"
WINDOWS = (20, 60, 120)
SIGNALS = ["s20", "s60", "s120", "t20", "t60", "t120", "c60"]
COLS = ["ts_code", "pct", "amount", "is_paused", "is_st"]


def style_series(cfg: dict) -> pd.DataFrame:
    boards = cfg["universe"]["boards"]
    rec = []
    for p in sorted(PANEL.glob("*.csv.gz")):
        d = p.stem.replace(".csv", "")
        if d < "20140101" or d > "20261120":
            continue
        df = pd.read_csv(p, usecols=COLS)
        sub = df.loc[(df["is_paused"] == 0) & (df["is_st"] == 0),
                     ["ts_code", "pct", "amount"]].dropna()
        if len(sub) < 50:
            continue
        sub = sub[sub["ts_code"].map(lambda c: board_of(c) in boards)]
        sub = sub[(sub["pct"] > -21) & (sub["pct"] < 21)]
        if len(sub) < 50:
            continue
        w = sub["amount"].to_numpy(dtype="float64").clip(0)
        pct = sub["pct"].to_numpy(dtype="float64")
        if w.sum() <= 0:
            continue
        top = sub.nlargest(min(100, len(sub)), "amount")
        rec.append(dict(trade_date=d, aw_ret=float((w * pct).sum() / w.sum()),
                        ew_ret=float(pct.mean()), top100=float(top["pct"].mean()),
                        med=float(np.median(pct)),
                        conc=float(top["amount"].sum() / sub["amount"].sum())))
    s = pd.DataFrame(rec).set_index("trade_date").sort_index()
    s["spread"] = s["aw_ret"] - s["ew_ret"]
    s["spread_top"] = s["top100"] - s["med"]
    for w_ in WINDOWS:
        mp = max(5, w_ // 2)
        s[f"s{w_}"] = s["spread"].rolling(w_, min_periods=mp).sum()
        s[f"t{w_}"] = s["spread_top"].rolling(w_, min_periods=mp).sum()
    s["c60"] = s["conc"] - s["conc"].rolling(60, min_periods=20).mean()
    return s


def trade_stats(tr: pd.DataFrame, mult: np.ndarray) -> dict:
    keep = mult > 0
    pf = tr["pos_frac"].fillna(1.0).to_numpy()
    r = tr["ret_pct"].to_numpy()
    logeq = float(np.sum(np.log1p(mult * pf * r / 100)))
    return dict(n=int(keep.sum()), mean_ret=round(float(r[keep].mean()), 3),
                win=round(float((r[keep] > 0).mean() * 100), 1),
                log_sum_pct=round(logeq * 100, 1))


def expanding_mult(tr: pd.DataFrame, col: str, q: float, side: str, mult: float,
                   warm: int = 80) -> np.ndarray:
    hist, out = [], []
    for v in tr[col].to_numpy():
        m = 1.0
        if pd.notna(v) and len(hist) >= warm:
            th = float(np.quantile(hist, q))
            if (side == "high" and v >= th) or (side == "low" and v <= th):
                m = mult
        if pd.notna(v):
            hist.append(float(v))
        out.append(m)
    return np.array(out)


def oos_mult(tr: pd.DataFrame, col: str, q: float, side: str, mult: float,
             split: str = "20200101") -> np.ndarray:
    first = (tr["signal_date"] < split).to_numpy()
    th = float(np.quantile(tr.loc[first, col].dropna(), q))
    m = np.ones(len(tr))
    for i, v in enumerate(tr[col].to_numpy()):
        if (not first[i]) and pd.notna(v):
            if (side == "high" and v >= th) or (side == "low" and v <= th):
                m[i] = mult
    return m


def expanding_z_mult(tr: pd.DataFrame, col: str, z_thr: float, side: str, mult: float,
                     warm: int = 80) -> np.ndarray:
    hist, out = [], []
    for v in tr[col].to_numpy():
        m = 1.0
        if pd.notna(v) and len(hist) >= warm:
            mu, sd = float(np.mean(hist)), float(np.std(hist))
            if sd > 0:
                z = (float(v) - mu) / sd
                if (side == "high" and z >= z_thr) or (side == "low" and z <= -z_thr):
                    m = mult
        if pd.notna(v):
            hist.append(float(v))
        out.append(m)
    return np.array(out)


def quintiles(tr: pd.DataFrame, col: str) -> list[dict]:
    d = tr.dropna(subset=[col]).copy()
    d["q"] = pd.qcut(d[col], 5, labels=False, duplicates="drop")
    out = []
    for q, g in d.groupby("q"):
        out.append(dict(q=int(q) + 1, n=int(len(g)), lo=round(float(g[col].min()), 2),
                        hi=round(float(g[col].max()), 2),
                        mean_ret=round(float(g["ret_pct"].mean()), 3),
                        win=round(float((g["ret_pct"] > 0).mean() * 100), 1)))
    return out


def block_boot(tr: pd.DataFrame, mult: np.ndarray, b: int = 4000, l: int = 10,
               seed: int = 7) -> dict:
    pf = tr["pos_frac"].fillna(1.0).to_numpy()
    r = tr["ret_pct"].to_numpy() / 100
    diff = np.log1p(mult * pf * r) - np.log1p(pf * r)
    n, k = len(diff), int(np.ceil(len(diff) / l))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(b):
        idx = np.concatenate([np.arange(st, min(st + l, n))
                              for st in rng.integers(0, n - l + 1, k)])[:n]
        vals.append(diff[idx].sum() * 100)
    v = np.array(vals)
    return dict(mean_pp=round(float(v.mean()), 2), p_gt0=round(float((v > 0).mean()), 3),
                ci_lo=round(float(np.percentile(v, 2.5)), 2),
                ci_hi=round(float(np.percentile(v, 97.5)), 2))


def main() -> None:
    cfg = load_config()
    print("风格序列...", flush=True)
    sty = style_series(cfg)
    sty.to_csv(ROOT / "output/w262_style_daily.csv")
    print(f"  {len(sty)} 日 {sty.index[0]} ~ {sty.index[-1]}", flush=True)

    raw = json.load(open(ROOT / "output/backtest_long.json"))
    base = pd.DataFrame(raw["trades_all"])
    base = base[base["ret_pct"].notna()].copy()
    base["ret_pct"] = base["ret_pct"].astype(float)
    base = base.sort_values(["sell_date", "buy_date"])
    base = base.merge(sty[SIGNALS], left_on="signal_date", right_index=True, how="left")
    base.to_csv(ROOT / "output/w262_trades.csv", index=False)
    base_stats = trade_stats(base, np.ones(len(base)))
    print(f"  基线 {base_stats} 引擎 年化 {raw['metrics']['年化收益率%']}%", flush=True)

    overlap = int((pd.to_datetime(base["buy_date"]) <=
                   pd.to_datetime(base["sell_date"]).shift()).sum()) - 1
    print("  重叠笔", overlap, flush=True)

    d = base.copy()
    d["year"] = pd.to_datetime(d["sell_date"]).dt.year
    res = {"baseline": base_stats, "engine": raw["metrics"], "overlap": overlap,
           "trades": int(len(base)), "quintiles": {}, "variants": [], "oos": [],
           "yearly": {}, "signal_yearly": {}}
    for col in SIGNALS:
        res["quintiles"][col] = quintiles(base, col)
    res["yearly"] = {int(y): dict(n=int(len(g)), mean_ret=round(float(g["ret_pct"].mean()), 3),
                                  s60=round(float(g["s60"].mean()), 2)
                                  if g["s60"].notna().any() else None)
                     for y, g in d.groupby("year")}
    res["signal_yearly"] = {col: {int(y): round(float(g[col].mean()), 2)
                                  for y, g in d.groupby("year") if g[col].notna().any()}
                            for col in SIGNALS}
    res["corr"] = {col: dict(spearman=round(float(base[col].corr(base["ret_pct"], method="spearman")), 3),
                              pearson=round(float(base[col].corr(base["ret_pct"])), 3))
                   for col in SIGNALS}
    res["pool_stats"] = {str(k): dict(n=int(len(g)), mean_ret=round(float(g["ret_pct"].mean()), 3),
                                      win=round(float((g["ret_pct"] > 0).mean() * 100), 1))
                         for k, g in base.groupby("pool")}
    main_only = base[base["pool"] == "main"].reset_index(drop=True)
    res["main_quintiles"] = {col: quintiles(main_only, col) for col in ("s60", "t60", "c60")}

    specs = []
    for col in SIGNALS:
        for q in (0.2, 0.1):
            hi = int(round((1 - q) * 100))
            specs += [(f"{col}_high_q{hi}_block", col, 1 - q, "high", 0.0),
                      (f"{col}_high_q{hi}_half", col, 1 - q, "high", 0.5),
                      (f"{col}_low_q{int(q * 100)}_block", col, q, "low", 0.0),
                      (f"{col}_low_q{int(q * 100)}_half", col, q, "low", 0.5)]
    for name, col, q, side, mult in specs:
        m = expanding_mult(base, col, q, side, mult)
        res["variants"].append(dict(name=name, mult=mult, stats=trade_stats(base, m),
                                    boot=block_boot(base, m)))
        m2 = oos_mult(base, col, q, side, mult)
        res["oos"].append(dict(name=name + "_oos", mult=mult, stats=trade_stats(base, m2)))

    for col in ["s60", "t60", "c60"]:
        for z in (1.0, 1.5):
            for side in ("high", "low"):
                for mult in (0.0, 0.5):
                    name = f"{col}_z{z}_{side}_{'block' if mult == 0 else 'half'}"
                    m = expanding_z_mult(base, col, z, side, mult)
                    res["variants"].append(dict(name=name, mult=mult,
                                                stats=trade_stats(base, m),
                                                boot=block_boot(base, m)))

    with open(ROOT / "output/exp_style_262.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps(res["baseline"], ensure_ascii=False))
    for v in sorted(res["variants"], key=lambda x: -x["stats"]["mean_ret"])[:8]:
        print(f"{v['name']:<22} n {v['stats']['n']:>3}  mean {v['stats']['mean_ret']:>6.3f}%  "
              f"win {v['stats']['win']:>5.1f}%  log {v['stats']['log_sum_pct']:>6.0f}  "
              f"boot {v['boot']['mean_pp']:+.0f}pp p={v['boot']['p_gt0']:.2f}", flush=True)
    for v in sorted(res["oos"], key=lambda x: -x["stats"]["mean_ret"])[:6]:
        print(f"{v['name']:<26} n {v['stats']['n']:>3}  mean {v['stats']['mean_ret']:>6.3f}%", flush=True)


if __name__ == "__main__":
    main()
