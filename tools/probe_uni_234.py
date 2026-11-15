"""第二百三十四轮: 全样本单变量归纳

不问"加什么条件", 先问: 在 2400 万个(股票,日)样本里, 什么特征真的区分未来收益?

三个层次:
  L1 全样本(不限跌不跌): 找到的特征是"全市场普适的涨跌规律"
  L2 低开子集(n1_ok):     可执行子集, 因为不低开就不成交
  L3 跌幅分层内:          策略只在"跌"里选, 故最关心"在跌幅档内部, 什么还能再分"

每个特征按全样本十分位切 10 档, 然后按年统计每档平均 y。关键不是总体均值,
而是"分年符号一致率"—— 一个特征若 18 年里 15 年同向, 才叫规律; 若 9:9, 就是噪声。

输出 output/w234_uni_{scope}.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "output" / "feat234"

FEATS = ["pct", "amp", "close_pos", "low_tail", "up_shadow", "gap", "intraday",
         "vr", "turnover_rate", "amt_yi", "amt_ratio", "circ_mv",
         "bias5", "bias10", "bias20", "bias60",
         "dist_high60", "dist_low60", "dist_high20",
         "ret3", "ret5", "ret10", "ret20", "vol20", "maxdd20",
         "dn_streak", "up_streak", "rel_pct", "rs20",
         "breadth", "idx_dist", "listed"]

# 分箱边界需先在全样本上定, 否则各年不可比。主进程先算一次并存盘。
BINS = FEAT / "bins.json"


def clean(d: pd.DataFrame) -> pd.DataFrame:
    """剔除不可交易的样本: 停牌、ST、涨跌停买不进、极端除权。"""
    m = (d["is_paused"] == 0)
    m &= (d["is_st"] == 0)
    # T+1 一字涨停买不进: T+1 收盘价 >= T+1 涨停价 的情况无法判断, 故用 T 日不涨停做粗筛
    m &= (d["pct"] > -11.5) & (d["pct"] < 11.5)
    m &= d["y"].between(-40, 60)
    m &= d["listed"] >= 250           # 上市满一年
    return d[m]


def scope_mask(d: pd.DataFrame, scope: str) -> pd.Series:
    if scope == "all":
        return pd.Series(True, index=d.index)
    if scope == "lowopen":
        return d["n1_ok"] == True      # noqa: E712
    if scope == "dip":                 # 策略实际面对的: 低开 且 当日跌 3~6%
        return (d["n1_ok"] == True) & d["pct"].between(-6.5, -2.5)   # noqa: E712
    raise ValueError(scope)


def compute_bins(paths: list[Path]) -> dict:
    """全样本分位数作分箱边界(1%..99% 的 19 个切点 -> 20 档)。"""
    out = {}
    for f in FEATS:
        vals = []
        for p in paths:
            s = pd.read_parquet(p, columns=[f])[f].dropna()
            vals.append(s.sample(min(60000, len(s)), random_state=0).values)
        v = np.concatenate(vals)
        qs = np.nanquantile(v, np.linspace(0.02, 0.98, 25))
        out[f] = np.unique(np.round(qs, 6)).tolist()
    return out


def main() -> None:
    paths = sorted(FEAT.glob("f*.parquet"))
    print(f"{len(paths)} 年文件", flush=True)

    if not BINS.exists():
        import json
        b = compute_bins(paths)
        BINS.write_text(json.dumps(b))
        print("分箱边界已建", flush=True)
    import json
    bins = json.loads(BINS.read_text())

    for scope in ("all", "lowopen", "dip"):
        acc: dict[str, list[dict]] = {f: [] for f in FEATS}
        base = []
        for p in paths:
            year = int(p.stem[1:])
            cols = ["ts_code", "trade_date", "is_paused", "is_st", "high_limit",
                    "low_limit", "y", "y_1d", "y_3d", "n1_ok", "n1_gap", "pct", "listed"] + FEATS
            cols = list(dict.fromkeys(cols))
            d = pd.read_parquet(p, columns=cols)
            d = clean(d)
            d = d[scope_mask(d, scope)]
            if len(d) < 2000:
                continue
            base.append({"scope": scope, "year": year, "n": len(d),
                         "y": round(float(d["y"].mean()), 3),
                         "y_med": round(float(d["y"].median()), 3),
                         "win": round(float((d["y"] > 0).mean()) * 100, 1)})
            for f in FEATS:
                b = bins.get(f)
                if not b or len(b) < 3:
                    continue
                x = d[f].values
                y = d["y"].values
                ok = np.isfinite(x) & np.isfinite(y)
                if ok.sum() < 2000:
                    continue
                lab = np.digitize(x[ok], b)      # 0..len(b)
                yy = y[ok]
                for k in range(len(b) + 1):
                    m = lab == k
                    if m.sum() < 200:
                        continue
                    acc[f].append({"scope": scope, "year": year, "bin": k,
                                   "n": int(m.sum()), "y": float(yy[m].mean())})
            print(f"  {scope} {year} n={len(d)} y={base[-1]['y']:+.3f}", flush=True)

        pd.DataFrame(base).to_csv(FEAT / f"base_{scope}.csv", index=False)

        # 汇总: 每特征每档的加权均值 + 分年符号一致率
        rows = []
        for f in FEATS:
            if not acc[f]:
                continue
            t = pd.DataFrame(acc[f])
            agg = t.groupby("bin").apply(
                lambda g: pd.Series({
                    "n": g["n"].sum(),
                    "y": (g["y"] * g["n"]).sum() / g["n"].sum(),
                }), include_groups=False).reset_index()
            if len(agg) < 4:
                continue
            # 分年: 每档的相对高低(相对该年该特征全体均值)
            piv = t.pivot_table(index="year", columns="bin", values="y")
            # 极差方向: 最高档 - 最低档, 逐年
            if 0 not in piv.columns or (len(agg) - 1) not in piv.columns:
                continue
            spread = (piv[len(agg) - 1] - piv[0]).dropna()
            rows.append({
                "特征": f, "档数": len(agg),
                "低档": round(agg["y"].iloc[0], 3), "高档": round(agg["y"].iloc[-1], 3),
                "极差": round(agg["y"].iloc[-1] - agg["y"].iloc[0], 3),
                "单调": round(float(np.corrcoef(np.arange(len(agg)), agg["y"])[0, 1]), 3),
                "正年数": int((spread > 0).sum()), "年数": len(spread),
                "合计n": int(agg["n"].sum()),
            })
            # 也存明细
            t.to_csv(FEAT / f"detail_{scope}_{f}.csv", index=False)
        out = pd.DataFrame(rows).sort_values("单调", key=lambda s: s.abs(), ascending=False)
        out.to_csv(ROOT / "output" / f"w234_uni_{scope}.csv", index=False, encoding="utf-8-sig")
        print(f"\n===== {scope} =====", flush=True)
        print(out.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
