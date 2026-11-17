"""第244轮 F: 定稿比对 + 分年 + 图表

定稿: 9 窗口排名平均(rank-mean ensemble) + 10 日冷却 + 50% 仓
对照: 243 轮定稿 = 单窗口 250 日 + 20 日冷却 + 70% 仓（同一 198 天口径）
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
POOL = OUT / "w244_pool"

WINS = (60, 90, 120, 150, 180, 200, 250, 300, 360)
COST = 0.10

# scan_244e.py 实测输出（冷却 10 日, 仓位 50%）
SUBSET = {2: (4.75, 3.80, 5.29), 3: (4.94, 4.23, 5.45), 5: (5.11, 4.09, 5.69),
          7: (5.37, 4.54, 5.70), 9: (5.55, 5.44, 5.56)}
SINGLE_WIN = {60: 3.70, 90: 5.04, 120: 5.02, 150: 4.48, 180: 5.13, 200: 4.82,
              250: 4.73, 300: 4.48, 360: 4.61}


def metrics(r: pd.Series) -> tuple[float, float, float]:
    nav = (1 + r / 100).cumprod()
    ann = nav.iloc[-1] ** (244 / len(r)) - 1
    dd = (nav / nav.cummax() - 1).min()
    return round(ann * 100, 2), round(dd * 100, 2), round(ann * 100 / abs(dd * 100), 2)


def main() -> None:
    sig = pd.read_parquet(POOL / "signals.parquet")
    sig["trade_date"] = sig["trade_date"].astype("int64")
    tr = pd.read_csv(OUT / "w232_trades.csv")
    all_dates = sorted(int(p.stem.replace(".csv", "")) for p in
                       (ROOT / "data" / "panel").glob("*.csv.gz"))
    cal = np.array([d for d in all_dates if d >= tr.signal.min()])
    base = pd.Series(0.0, index=cal)
    for dt, g in tr.groupby("signal"):
        base[int(dt)] = g["ret"].mean()
    dayrank = {int(d): i for i, d in enumerate(cal)}

    wide = sig[["ts_code", "trade_date", "y", "amt_yi"]].copy()
    for W in WINS:
        px = pd.read_parquet(POOL / f"W{W}.parquet").rename(columns={"dh": f"dh{W}"})
        wide = wide.merge(px, on=["ts_code", "trade_date"], how="left")
        del px
    dhcols = [f"dh{W}" for W in WINS]
    wide = wide.dropna(subset=dhcols, how="all")
    byday = {int(dt): g for dt, g in wide.groupby("trade_date")}
    order = sorted(byday)

    def run(cols: list[str], cd: int) -> tuple[pd.Series, pd.Series]:
        last: dict[str, int] = {}
        v, pk = {}, {}
        for dt in order:
            g = byday[dt].copy()
            denom = {c: max(int(g[c].notna().sum()), 1) for c in cols}
            rk = pd.DataFrame({c: g[c].rank(ascending=True) / denom[c] for c in cols})
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
            v[dt] = float(r["y"]) - COST
            pk[dt] = r["ts_code"]
            last[r["ts_code"]] = dayrank[dt]
        return pd.Series(v).sort_index(), pd.Series(pk).sort_index()

    v_ens, pk_ens = run(list(dhcols), 10)
    v_243, pk_243 = run(["dh250"], 20)

    nav_b = (1 + base / 100).cumprod()
    s_ens = base.copy()
    s_ens[v_ens.index] += v_ens.values * 0.50
    s_243 = base.copy()
    s_243[v_243.index] += v_243.values * 0.70
    nav_e = (1 + s_ens / 100).cumprod()
    nav_2 = (1 + s_243 / 100).cumprod()

    print(f"{'方案':<20}{'次数':>5}{'单笔':>8}{'年化':>9}{'回撤':>9}{'卡玛':>7}{'票数':>5}{'连买':>5}")
    res = []
    for name, v, ser, pos in (("基准(仅主策略)", None, base, 0),
                              ("243定稿 250/20/70%", v_243, s_243, 0.70),
                              ("244集成 9win/10/50%", v_ens, s_ens, 0.50)):
        a, dd, cm = metrics(ser)
        if v is None:
            print(f"{name:<20}{'':>5}{'':>8}{a:>9.2f}{dd:>9.2f}{cm:>7.2f}")
        else:
            pk = pk_ens if pos == 0.50 else pk_243
            run_len = mx = 0
            prev = None
            for x in pk.values:
                run_len = run_len + 1 if x == prev else 1
                prev = x
                mx = max(mx, run_len)
            print(f"{name:<20}{len(v):>5}{v.mean():>8.3f}{a:>9.2f}{dd:>9.2f}{cm:>7.2f}"
                  f"{pk.nunique():>5}{mx:>5}")
        res.append({"方案": name, "次数": len(v) if v is not None else 0, "年化": a,
                    "回撤": dd, "卡玛": cm})

    # 三段
    print("\n分段卡玛")
    segs = [(2016, 2020), (2021, 2023), (2024, 2026)]
    hdr = "".join(f"{f'{a}-{b}':>16}" for a, b in segs)
    print(f"{'方案':<20}{hdr}")
    for name, ser in (("基准", base), ("243定稿", s_243), ("244集成", s_ens)):
        line = f"{name:<20}"
        for a, b in segs:
            x = ser[(ser.index // 10000 >= a) & (ser.index // 10000 <= b)]
            line += f"{metrics(x)[2]:>16.2f}"
        print(line)

    # 分年
    print("\n分年收益(%)")
    yy = pd.DataFrame({"yr": s_ens.index // 10000, "ens": s_ens.values,
                       "o": s_243.values, "b": base.values}).groupby("yr").agg(
        {"ens": lambda x: ((1 + x / 100).prod() - 1) * 100,
         "o": lambda x: ((1 + x / 100).prod() - 1) * 100,
         "b": lambda x: ((1 + x / 100).prod() - 1) * 100}).round(1)
    yy.columns = ["集成", "243定稿", "仅主策略"]
    print(yy.to_string())
    yy.to_csv(OUT / "w244_yearly.csv", encoding="utf-8-sig")

    pd.DataFrame({"signal": v_ens.index, "code": pk_ens.values,
                  "y": v_ens.values}).to_csv(OUT / "w244_picks.csv", index=False,
                                             encoding="utf-8-sig")
    pd.DataFrame(res).to_csv(OUT / "w244_final.csv", index=False, encoding="utf-8-sig")

    # ---------------- 图 ----------------
    grid = pd.read_csv(OUT / "w244_d_grid.csv")
    ic = pd.read_csv(OUT / "w244_ic.csv")
    fig, ax = plt.subplots(2, 2, figsize=(15.5, 10.4), dpi=150)
    fig.suptitle("第244轮  参数选择的谎言, 与集成这个解法",
                 fontsize=17, fontweight="bold", y=0.975)

    # (1) 参数面
    pv = grid.pivot_table(index="C", columns="P", values="卡玛")
    im = ax[0, 0].imshow(pv.values, cmap="RdYlGn", aspect="auto", vmin=4.0, vmax=5.7)
    ax[0, 0].set_xticks(range(len(pv.columns)))
    ax[0, 0].set_xticklabels([f"{p:.0%}" for p in pv.columns])
    ax[0, 0].set_yticks(range(len(pv.index)))
    ax[0, 0].set_yticklabels(pv.index)
    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            ax[0, 0].text(j, i, f"{pv.values[i, j]:.2f}", ha="center", va="center",
                          fontsize=7.5, color="#222")
    ci = list(pv.index).index(10)
    cj = list(pv.columns).index(0.50)
    ax[0, 0].plot(cj, ci, "*", ms=22, color="#c0392b", mec="white", mew=1.6,
                  label="244集成 (冷却10, 仓位50%)")
    ax[0, 0].set_xlabel("仓位"); ax[0, 0].set_ylabel("冷却(交易日)")
    ax[0, 0].set_title("① 集成参数面: 冷却 5~30 × 仓位 40~50% 整块平台 (243 定稿同口径仅 4.33)",
                       fontsize=11.5, loc="left")
    ax[0, 0].legend(fontsize=8.2, loc="upper center", bbox_to_anchor=(0.5, -0.13),
                   ncol=1, frameon=False)
    cb = plt.colorbar(im, ax=ax[0, 0], fraction=0.045)
    cb.set_label("卡玛", rotation=270, labelpad=14)

    # (2) IC
    ax[0, 1].bar(ic.yr.astype(str), ic.rho, color=np.where(ic.rho > 0, "#c0392b", "#2e7d32"))
    ax[0, 1].axhline(0, color="#333", lw=1)
    ax[0, 1].axhline(ic.rho.mean(), color="#8e44ad", ls="--", lw=1.6,
                     label=f"均值 {ic.rho.mean():+.3f}")
    ax[0, 1].set_ylim(-0.7, 0.35)
    ax[0, 1].set_ylabel("Spearman 秩相关")
    ax[0, 1].set_title("② 用历史卡玛挑参数 -> 次年表现: 反相关, 9 年只有 3 年为正",
                       fontsize=11.5, loc="left")
    ax[0, 1].legend(fontsize=9, loc="lower left")

    # (3) 子集大小
    ks = sorted(SUBSET)
    med = [SUBSET[k][0] for k in ks]
    lo = [SUBSET[k][1] for k in ks]
    hi = [SUBSET[k][2] for k in ks]
    ax[1, 0].fill_between(ks, lo, hi, color="#3498db", alpha=0.22, label="5%~95% 区间")
    ax[1, 0].plot(ks, med, "o-", color="#2980b9", lw=2.2, ms=8, label="中位卡玛")
    ax[1, 0].axhline(np.median(list(SINGLE_WIN.values())), color="#e67e22", ls="--", lw=1.6,
                     label=f"单窗口中位 {np.median(list(SINGLE_WIN.values())):.2f}")
    ax[1, 0].axhline(max(SINGLE_WIN.values()), color="#c0392b", ls=":", lw=1.6,
                     label=f"最好单窗口 {max(SINGLE_WIN.values()):.2f}")
    ax[1, 0].set_xlim(1.3, 9.7)
    ax[1, 0].set_xticks(ks)
    for k in ks:
        ax[1, 0].annotate(f"{SUBSET[k][0]:.2f}", (k, SUBSET[k][0]), (0, 9),
                          textcoords="offset points", ha="center", fontsize=8.5,
                          color="#1a5276")
    ax[1, 0].set_xlabel("集成使用的窗口数"); ax[1, 0].set_ylabel("卡玛")
    ax[1, 0].set_title("③ 窗口从 2 个加到 9 个, 卡玛单调上升, 越过最好的单窗口 5.13",
                       fontsize=11.5, loc="left")
    ax[1, 0].legend(fontsize=8.5, loc="lower right")

    # (4) 净值
    x = pd.to_datetime(nav_e.index.astype(str))
    ax[1, 1].plot(x, nav_b.values, color="#7f8c8d", lw=1.5, label=f"仅主策略 卡玛{res[0]['卡玛']:.2f}")
    ax[1, 1].plot(x, nav_2.values, color="#2c3e50", lw=1.5, label=f"243定稿 卡玛{res[1]['卡玛']:.2f}")
    ax[1, 1].plot(x, nav_e.values, color="#c0392b", lw=2.0, label=f"244集成 卡玛{res[2]['卡玛']:.2f}")
    ax[1, 1].set_yscale("log")
    ax[1, 1].set_ylabel("净值(对数)"); ax[1, 1].legend(fontsize=9, loc="upper left")
    ax[1, 1].set_title("④ 11 年净值: 集成把年化从 117% 提到 136%, 回撤还小 2.5pp",
                       fontsize=11.5, loc="left")
    ax[1, 1].grid(alpha=0.25)

    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(OUT / "第244轮_参数选择的谎言与集成.png", bbox_inches="tight")
    print("\n图已保存")


if __name__ == "__main__":
    main()
