"""第244轮 H: 剔除"次日低开才买"的未来信息 —— 实盘真正能拿到多少

前面所有轮次的回测口径都带了一个隐藏的未来信息:
  build_features_234 里 n1_ok = (T+1 开盘 / T 收盘 - 1 <= +0.05%)，
  而候选池是 `d = d[d.n1_ok == True]` —— 在 T 日收盘的池子里, 事先剔除了"明天会高开"的票。
实盘做不到: T 日收盘只能选一只, T+1 开盘才知道它高开没有。高开就只能放弃(空仓)。

本轮用同一套 9 窗口集成, 池子不过滤 n1_ok, 选出 top1 后:
  n1_ok=True  -> 按 y 成交
  n1_ok=False -> 放弃, 该笔计 0 (空仓一天)
对照"池内预过滤"版本(即之前所有轮次的口径)。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"

WINS = (60, 90, 120, 150, 180, 200, 250, 300, 360)
COST = 0.10


def metrics(r: pd.Series) -> tuple[float, float, float]:
    nav = (1 + r / 100).cumprod()
    ann = nav.iloc[-1] ** (244 / len(r)) - 1
    dd = (nav / nav.cummax() - 1).min()
    return round(ann * 100, 2), round(dd * 100, 2), round(ann * 100 / abs(dd * 100), 2)


def build(pool: Path) -> tuple[pd.DataFrame, pd.Series]:
    sig = pd.read_parquet(pool / "signals.parquet")
    sig["trade_date"] = sig["trade_date"].astype("int64")
    wide = sig.copy()
    for W in WINS:
        px = pd.read_parquet(pool / f"W{W}.parquet").rename(columns={"dh": f"dh{W}"})
        wide = wide.merge(px, on=["ts_code", "trade_date"], how="left")
        del px
    return wide


def main() -> None:
    tr = pd.read_csv(OUT / "w232_trades.csv")
    all_dates = sorted(int(p.stem.replace(".csv", "")) for p in
                       (ROOT / "data" / "panel").glob("*.csv.gz"))
    cal = np.array([d for d in all_dates if d >= tr.signal.min()])
    base = pd.Series(0.0, index=cal)
    for dt, g in tr.groupby("signal"):
        base[int(dt)] = g["ret"].mean()
    dayrank = {int(d): i for i, d in enumerate(cal)}
    print(f"基准 年化{metrics(base)[0]:.2f} 回撤{metrics(base)[1]:.2f} 卡玛{metrics(base)[2]:.2f}")

    def run(wide: pd.DataFrame, cols: list[str], cd: int,
            skip_if_n1_bad: bool) -> tuple[pd.Series, dict]:
        dhcols = [c for c in cols]
        byday = {int(dt): g for dt, g in wide.groupby("trade_date")}
        last: dict[str, int] = {}
        v, skip, pk = {}, 0, {}
        for dt in sorted(byday):
            g = byday[dt].dropna(subset=dhcols).copy()
            if not len(g):
                continue
            denom = {c: max(int(g[c].notna().sum()), 1) for c in dhcols}
            rk = pd.DataFrame({c: g[c].rank(ascending=True) / denom[c] for c in dhcols})
            g["mr"] = rk.mean(axis=1, skipna=True)
            g = g.sort_values(["mr", "amt_yi"], ascending=[True, False])
            if cd > 0:
                ok = [(c not in last) or (dayrank[dt] - last[c] >= cd) for c in g["ts_code"]]
                cand = g[np.array(ok)]
                if len(cand):
                    g = cand
            r = g.iloc[0]
            if pd.isna(r["mr"]):
                continue
            ok_n1 = bool(r.get("n1_ok", True))
            if skip_if_n1_bad and not ok_n1:
                skip += 1
                v[dt] = 0.0                     # 高开, 放弃, 空仓
                continue
            y = r["y"]
            if pd.isna(y):
                v[dt] = 0.0
                skip += 1
                continue
            v[dt] = float(y) - COST
            pk[dt] = r["ts_code"]
            last[r["ts_code"]] = dayrank[dt]
        return pd.Series(v).sort_index(), {"n": len(v), "skip": skip, "picks": pd.Series(pk)}

    print("\n" + "=" * 86)
    print(f"{'口径':<28}{'仓位':>5}{'笔数':>5}{'放弃':>5}{'单笔':>8}{'年化':>9}{'回撤':>9}{'卡玛':>7}")
    rows = []
    for tag, pool, skip in (("池内预过滤(旧口径, 含未来信息)", OUT / "w244_pool", False),
                            ("T+1 开盘发现高开就放弃", OUT / "w244_pool2", True)):
        wide = build(pool)
        for cols, name in ((WINS, f"9窗口集成 C=10"), (["250"], "单窗口250 C=20")):
            cd = 10 if len(cols) > 1 else 20
            v, info = run(wide, [f"dh{W}" for W in cols] if len(cols) > 1 else ["dh250"],
                          cd, skip)
            traded = v[v != 0]
            for P in (0.50, 0.70):
                sim = base.copy()
                sim[v.index] += v.values * P
                a, dd, cm = metrics(sim)
                print(f"{tag + ' ' + name:<28}{P:>5.0%}{len(traded):>5}{info['skip']:>5}"
                      f"{traded.mean():>8.3f}{a:>9.2f}{dd:>9.2f}{cm:>7.2f}")
                rows.append({"口径": tag, "方案": name, "P": P, "成交": len(traded),
                             "放弃": info["skip"], "单笔": round(traded.mean(), 3),
                             "年化": a, "回撤": dd, "卡玛": cm})
        del wide

    print("\n分年(集成 C=10 P=50%, 高开放弃口径):")
    wide = build(OUT / "w244_pool2")
    v, info = run(wide, [f"dh{W}" for W in WINS], 10, True)
    sim = base.copy()
    sim[v.index] += v.values * 0.50
    p = pd.DataFrame({"yr": v.index // 10000, "y": v.values})
    g = p.groupby("yr").agg(n=("y", "size"), zero=("y", lambda x: (x == 0).sum()),
                            ret=("y", lambda x: ((1 + x / 100).prod() - 1) * 100))
    g["胜%"] = p.groupby("yr").y.apply(lambda x: (x[x != 0] > 0).mean() * 100).round(0)
    print(g.round(1).to_string())
    pd.DataFrame(rows).to_csv(OUT / "w244_n1ok.csv", index=False, encoding="utf-8-sig")
    v.to_frame("y").to_csv(OUT / "w244_picks_live.csv", encoding="utf-8-sig")


if __name__ == "__main__":
    main()
