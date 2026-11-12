"""第 204 轮: 用全史样本"训练"选股条件(而不是继续手调).

两个问题分开回答, 都用同一份 1,400 万股票日的面板:
  A. 评分权重 —— 十维因子(与 config.selection.dip_weights 的可选键一一对应)
     用**滚动前向**(walk-forward) 训练: 只用过去 3 年已结算样本拟合, 预测下一年,
     逐年滚动 16 年。样本外表现才是能不能改 config 的唯一依据。
  B. 硬门槛 —— pct / close_pos / mom20 / dist_low20 / vol_ratio 的现行切点
     是否最优。用同一份样本画边际曲线, 并在"前 8 年定切点、后 8 年验证"下复算。

口径: 目标 y = T+1 开盘买 -> T+2 收盘卖的净收益(%); 每日横截面 z 化(winsor 1%/99%);
      每天按评分取 top1(与 exec_top_n=1 一致), 记该笔收益与"当日池均值"的差。

输出 output/train_204.json / output/train_204_yearly.csv / output/gate_204.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"

# 与 src/selector.py 的 dip_weights 可选键严格对应(方向已内建: neg_* 表示越小越好)
FACTORS = [
    ("mom20", "mom20", 1.0),
    ("neg_bias20", "bias20", -1.0),
    ("neg_amount", "logamt", -1.0),
    ("neg_close_pos", "close_pos", -1.0),
    ("neg_vol_ratio", "vr", -1.0),
    ("low_tail", "low_tail", 1.0),
    ("neg_turnover_cool", "turnover_cool", -1.0),
    ("ivol", "ivol20", 1.0),
    ("neg_dist_ma5", "dist_ma5", -1.0),
    ("neg_dist_low20", "dist_low20", -1.0),
]
KEYS = [k for k, _, _ in FACTORS]
CUR_W = {"mom20": 0.20, "neg_bias20": 0.70, "neg_amount": 0.05,
         "neg_vol_ratio": 0.02, "low_tail": 0.03}
COST = 0.30          # 引擎 cost_pct(%): 双边佣金+印花+滑点


def load() -> pd.DataFrame:
    d = pd.read_parquet(OUT / "panel_train_204.parquet")
    # 复权异常清洗: 主板两天最多约 ±21%(连续涨跌停), 两天 |ret|>25 或缺口 >15%
    # 基本都来自 factor 口径错乱, 留着会污染 ridge 的系数
    bad = (d["ret"].abs() > 25) | (d["gap"].abs() > 15)
    print(f"剔除复权异常 {int(bad.sum())} 行")
    d = d[~bad].reset_index(drop=True)
    d["year"] = (d["d"].str[:4]).astype("int16")
    d["logamt"] = np.log1p(d["amt_yi"].astype("float64") * 1e8)
    for k, col, sign in FACTORS:
        d[f"raw_{k}"] = sign * d[col].astype("float64")
    return d


def zscore(d: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """按日横截面 z 化, 先 winsor 1%/99% 去极值。输入 raw_<key> 列, 输出 z_<key>。"""
    g = d.groupby("d", sort=False)
    out = {}
    for k in keys:
        c = "raw_" + k
        s = d[c]
        lo = g[c].transform(lambda x: x.quantile(0.01))
        hi = g[c].transform(lambda x: x.quantile(0.99))
        s = s.clip(lower=lo, upper=hi)
        mu = s.groupby(d["d"], sort=False).transform("mean")
        sd = s.groupby(d["d"], sort=False).transform("std")
        out["z_" + k] = ((s - mu) / sd.replace(0, np.nan)).fillna(0.0)
    return pd.DataFrame(out, index=d.index)


def ridge(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Xs = (X - mu) / sd
    A = Xs.T @ Xs + lam * np.eye(X.shape[1])
    return np.linalg.solve(A, Xs.T @ (y - y.mean()))


def norm_w(beta: np.ndarray) -> dict:
    """负系数截断为 0(config 只接受正权重), 再按 sum=1 归一化。"""
    b = np.clip(beta, 0, None)
    if b.sum() <= 0:
        b = np.ones_like(b)
    b = b / b.sum()
    return {k: round(float(v), 4) for k, v in zip(KEYS, b)}


def top1_stats(d: pd.DataFrame, w: dict, mask: pd.Series | None = None) -> dict:
    """每日按 w 打分取 top1, 返回逐笔收益与相对池均值的超额。"""
    dd = d if mask is None else d[mask]
    if dd.empty:
        return {}
    zv = dd[["z_" + k for k in KEYS]].values
    wv = np.array([w.get(k, 0.0) for k in KEYS])
    dd = dd.assign(_s=zv @ wv)
    idx = dd.groupby("d", sort=False)["_s"].idxmax()
    pick = dd.loc[idx]
    pool = dd.groupby("d", sort=False)["ret"].mean()
    r = pick["ret"] - COST
    return {"n": int(len(pick)), "mean": float(r.mean()),
            "median": float(r.median()), "win": float((r > 0).mean() * 100),
            "excess_vs_pool": float((pick["ret"] - pick["d"].map(pool)).mean()),
            "days": pick[["d", "ts_code", "ret", "year"]].assign(net=r.values)}


def main() -> None:
    d = load()
    print(f"样本 {len(d):,} 行 / {d.d.nunique()} 日 / hard 子集 {int(d.hard.sum()):,}", flush=True)
    z = zscore(d, KEYS)
    d = pd.concat([d, z], axis=1)

    ZC = ["z_" + k for k in KEYS]
    hard = d["hard"] == 1

    # ---------- A. 滚动前向训练 ----------
    years = sorted(d["year"].unique())
    rows, oos = [], []
    for y in years:
        tr = d[(d.year >= y - 3) & (d.year < y) & hard]      # 只在引擎真正会买的池里学
        te = d[(d.year == y) & hard]
        if len(tr) < 3000 or te.empty:
            continue
        beta = ridge(tr[ZC].values, tr["ret"].values, lam=30.0)
        w = norm_w(beta)
        s = top1_stats(d, w, (d.year == y) & hard)
        s_cur = top1_stats(d, CUR_W, (d.year == y) & hard)
        s_pool = {"mean": float((te["ret"] - COST).mean())}
        rows.append({"year": int(y), "train_n": len(tr),
                     "oos_mean": round(s["mean"], 3), "cur_mean": round(s_cur["mean"], 3),
                     "pool_mean": round(s_pool["mean"], 3),
                     "oos_excess": round(s["excess_vs_pool"], 3),
                     "cur_excess": round(s_cur["excess_vs_pool"], 3),
                     "n": s["n"], **{f"w_{k}": w[k] for k in KEYS}})
        oos.append(s["days"])
        print(f"  {y} 训练 {len(tr):>6,} 样本 | 样本外 top1 {s['mean']:+.3f}% "
              f"(超额 {s['excess_vs_pool']:+.3f}pp) | 现行 {s_cur['mean']:+.3f}% "
              f"({s_cur['excess_vs_pool']:+.3f}pp)", flush=True)

    yr = pd.DataFrame(rows)
    yr.to_csv(OUT / "train_204_yearly.csv", index=False)
    allq = pd.concat(oos, ignore_index=True)

    # ---------- 全史拟合(落地用) ----------
    beta_all = ridge(d[ZC].values, d["ret"].values, lam=30.0)
    w_all = norm_w(beta_all)
    # 只用 hard 子集拟合一版(引擎实际只在硬条件内选票)
    beta_hard = ridge(d.loc[hard, ZC].values, d.loc[hard, "ret"].values, lam=30.0)
    w_hard = norm_w(beta_hard)

    res = {
        "窗口": [int(allq["year"].min()), int(allq["year"].max())],
        "样本外笔数": int(len(allq)),
        "样本外 top1 均值%": round(float(allq["net"].mean()), 3),
        "现行权重 top1 均值%": round(float(yr["cur_mean"].mean()), 3),
        "池均值%": round(float(yr["pool_mean"].mean()), 3),
        "样本外胜率%": round(float((allq["net"] > 0).mean() * 100), 2),
        "样本外中位%": round(float(allq["net"].median()), 3),
        "训练权重(全史)": w_all,
        "训练权重(仅硬条件)": w_hard,
        "现行权重": CUR_W,
        "逐年": rows,
    }

    # 逐年胜负 + t 检验(逐笔配对: 同一交易日两套权重的选中票收益)
    cmp_rows = []
    for y in years:
        a = top1_stats(d, w_all, (d.year == y) & hard)
        b = top1_stats(d, CUR_W, (d.year == y) & hard)
        if not a or not b:
            continue
        m = a["days"][["d", "net"]].merge(b["days"][["d", "net"]], on="d", suffixes=("_new", "_cur"))
        cmp_rows.append({"year": int(y), "n": len(m),
                         "new": round(float(m["net_new"].mean()), 3),
                         "cur": round(float(m["net_cur"].mean()), 3),
                         "diff": round(float((m["net_new"] - m["net_cur"]).mean()), 3)})
    cmpdf = pd.DataFrame(cmp_rows)
    cmpdf.to_csv(OUT / "train_204_cmp.csv", index=False)
    diff = cmpdf["diff"].values
    res["全史对比_新vs现行"] = {"天数": int(cmpdf["n"].sum()),
                               "新": round(float(cmpdf["new"].mean()), 3),
                               "现行": round(float(cmpdf["cur"].mean()), 3),
                               "差pp": round(float(diff.mean()), 3),
                               "更好年数": int((cmpdf["diff"] > 0).sum()),
                               "总年数": int(len(cmpdf)),
                               "t": round(float(diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff)))), 2)}

    # ---------- B. 硬门槛边际曲线 ----------
    gates = []
    grid = [("pct", [-9, -8, -7, -6.5, -6, -5.5, -5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1]),
            ("close_pos", [0, .1, .2, .28, .32, .36, .4, .45, .5, .55]),
            ("mom20", [0, 5, 8, 10, 13, 16, 20, 25, 28, 32]),
            ("dist_low20", [1, 3, 6, 8, 10, 15, 20, 30, 45, 70]),
            ("vr", [0.2, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])]
    for col, edges in grid:
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            m = (d[col] >= lo) & (d[col] < hi) & hard
            if m.sum() < 300:
                continue
            r = d.loc[m, "ret"] - COST
            gates.append({"factor": col, "lo": lo, "hi": hi, "n": int(m.sum()),
                          "mean": round(float(r.mean()), 3),
                          "win": round(float((r > 0).mean() * 100), 2)})
    pd.DataFrame(gates).to_csv(OUT / "gate_204.csv", index=False)

    # 现行 vs 最优切点(前 8 年定, 后 8 年验证)
    split = 2018
    def gate_ret(lo, hi, df):
        m = df["pct"].between(lo, hi, inclusive="left")
        return float((df.loc[m, "ret"] - COST).mean()) if m.sum() > 200 else np.nan
    early, late = d[d.year < split], d[d.year >= split]
    best = None
    for lo in [-9, -8, -7, -6.5, -6, -5.5, -5, -4.5]:
        for hi in [-5, -4.5, -4, -3.5, -3, -2.5, -2]:
            if hi - lo < 1.0:
                continue
            v = gate_ret(lo, hi, early[hard.loc[early.index]])
            if v == v and (best is None or v > best[0]):
                best = (v, lo, hi)
    if best:
        v_late = gate_ret(best[1], best[2], late[hard.loc[late.index]])
        v_cur_late = gate_ret(-6.0, -3.0, late[hard.loc[late.index]])
        res["pct 门槛"] = {"前 8 年最优": [best[1], best[2]], "前 8 年收益": round(best[0], 3),
                           "后 8 年收益(新切点)": round(v_late, 3),
                           "后 8 年收益(现行 -6~-3)": round(v_cur_late, 3)}

    (OUT / "train_204.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n=== 全史拟合权重 ===")
    for k in KEYS:
        print(f"  {k:20s} 训练 {w_all[k]:.3f}  现行 {CUR_W.get(k, 0.0):.3f}")
    print("\n=== 样本外(逐年滚动) ===")
    print(f"  新权重 top1 均值 {res['样本外 top1 均值%']:.3f}% | "
          f"现行 {res['现行权重 top1 均值%']:.3f}% | 池均值 {res['池均值%']:.3f}%")
    print("  全史对比:", json.dumps(res["全史对比_新vs现行"], ensure_ascii=False))
    print("  pct 门槛:", json.dumps(res.get("pct 门槛", {}), ensure_ascii=False))


if __name__ == "__main__":
    main()
