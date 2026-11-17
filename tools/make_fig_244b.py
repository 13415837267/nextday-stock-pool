"""第244轮 图: 集成有效, 但前几轮的卡玛含 0.3~0.4 的水分"""
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
POOL = OUT / "w244_pool2"
WINS = (60, 90, 120, 150, 180, 200, 250, 300, 360)
COST = 0.10

SUBSET = {2: (4.75, 3.80, 5.29), 3: (4.94, 4.23, 5.45), 5: (5.11, 4.09, 5.69),
          7: (5.37, 4.54, 5.70), 9: (5.55, 5.44, 5.56)}
SINGLE_MED = 4.73
SINGLE_BEST = 5.13


def metrics(r: pd.Series) -> tuple[float, float, float]:
    nav = (1 + r / 100).cumprod()
    ann = nav.iloc[-1] ** (244 / len(r)) - 1
    dd = (nav / nav.cummax() - 1).min()
    return round(ann * 100, 2), round(dd * 100, 2), round(ann * 100 / abs(dd * 100), 2)


def main() -> None:
    tr = pd.read_csv(OUT / "w232_trades.csv")
    all_dates = sorted(int(p.stem.replace(".csv", "")) for p in
                       (ROOT / "data" / "panel").glob("*.csv.gz"))
    cal = np.array([d for d in all_dates if d >= tr.signal.min()])
    base = pd.Series(0.0, index=cal)
    for dt, g in tr.groupby("signal"):
        base[int(dt)] = g["ret"].mean()
    dayrank = {int(d): i for i, d in enumerate(cal)}

    def load(pool: Path):
        sig = pd.read_parquet(pool / "signals.parquet")
        sig["trade_date"] = sig["trade_date"].astype("int64")
        w = sig.copy()
        for W in WINS:
            px = pd.read_parquet(pool / f"W{W}.parquet").rename(columns={"dh": f"dh{W}"})
            w = w.merge(px, on=["ts_code", "trade_date"], how="left")
        return w

    def run(w: pd.DataFrame, cols: list[str], cd: int, skip: bool):
        byday = {int(dt): g for dt, g in w.groupby("trade_date")}
        last, v = {}, {}
        for dt in sorted(byday):
            g = byday[dt].dropna(subset=cols).copy()
            if not len(g):
                continue
            den = {c: max(int(g[c].notna().sum()), 1) for c in cols}
            rk = pd.DataFrame({c: g[c].rank(ascending=True) / den[c] for c in cols})
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
            if skip and (not bool(r["n1_ok"]) or pd.isna(r["y"])):
                v[dt] = 0.0
                continue
            v[dt] = float(r["y"]) - COST
            last[r["ts_code"]] = dayrank[dt]
        return pd.Series(v).sort_index()

    w1, w2 = load(OUT / "w244_pool"), load(POOL)
    dhcols = [f"dh{W}" for W in WINS]
    v_ens_old = run(w1, dhcols, 10, False)
    v_243_old = run(w1, ["dh250"], 20, False)
    v_ens_new = run(w2, dhcols, 10, True)
    v_243_new = run(w2, ["dh250"], 20, True)

    def ser(v, p):
        s = base.copy()
        s[v.index] += v.values * p
        return s

    rows = [("243定稿 250日/冷却20", v_243_old, v_243_new, 0.70),
            ("244集成 9窗口/冷却10", v_ens_old, v_ens_new, 0.70)]
    stats = []
    for name, vo, vn, p in rows:
        so, sn = ser(vo, p), ser(vn, p)
        stats.append((name, metrics(so), metrics(sn), len(vn[vn != 0]), int((vn == 0).sum())))
    s_ens_new = ser(v_ens_new, 0.70)
    s_243_new = ser(v_243_new, 0.70)

    for name, mo, mn, n, sk in stats:
        print(f"{name}: 旧口径 {mo}  实盘口径 {mn}  成交{n} 放弃{sk}")

    # ---------------- 图 ----------------
    ic = pd.read_csv(OUT / "w244_ic.csv")
    fig, ax = plt.subplots(2, 2, figsize=(15.5, 10.6), dpi=150)
    fig.suptitle("第244轮  不用挑参数了 —— 但要把次日低开才买这种未来信息扣掉",
                 fontsize=16.5, fontweight="bold", y=0.975)

    # ① IC
    ax[0, 0].bar(ic.yr.astype(str), ic.rho, color=np.where(ic.rho > 0, "#c0392b", "#2e7d32"))
    ax[0, 0].axhline(0, color="#333", lw=1)
    ax[0, 0].axhline(ic.rho.mean(), color="#8e44ad", ls="--", lw=1.6,
                     label=f"均值 {ic.rho.mean():+.3f}")
    ax[0, 0].set_ylim(-0.7, 0.35)
    ax[0, 0].set_ylabel("Spearman 秩相关")
    ax[0, 0].set_title("① 用历史卡玛挑排序窗口 -> 次年的表现: 反相关, 9 年只 3 年为正",
                       fontsize=11.5, loc="left")
    ax[0, 0].legend(fontsize=9, loc="lower left")

    # ② 子集
    ks = sorted(SUBSET)
    med = [SUBSET[k][0] for k in ks]
    ax[0, 1].fill_between(ks, [SUBSET[k][1] for k in ks], [SUBSET[k][2] for k in ks],
                          color="#3498db", alpha=0.2, label="5%~95% 区间")
    ax[0, 1].plot(ks, med, "o-", color="#2980b9", lw=2.2, ms=8, label="中位卡玛")
    ax[0, 1].axhline(SINGLE_MED, color="#e67e22", ls="--", lw=1.6, label=f"单窗口中位 {SINGLE_MED}")
    ax[0, 1].axhline(SINGLE_BEST, color="#c0392b", ls=":", lw=1.6, label=f"最好单窗口 {SINGLE_BEST}")
    for k in ks:
        ax[0, 1].annotate(f"{SUBSET[k][0]:.2f}", (k, SUBSET[k][0]), (0, 9),
                          textcoords="offset points", ha="center", fontsize=8.5, color="#1a5276")
    ax[0, 1].set_xlim(1.3, 9.7)
    ax[0, 1].set_xticks(ks)
    ax[0, 1].set_ylim(3.5, 5.85)
    ax[0, 1].set_xlabel("集成使用的窗口数"); ax[0, 1].set_ylabel("卡玛")
    ax[0, 1].set_title("② 窗口 2→9 个, 卡玛单调上升并越过最好的单窗口 —— 平均本身在赚钱",
                       fontsize=11.5, loc="left")
    ax[0, 1].legend(fontsize=8.5, loc="lower right")

    # ③ 旧口径 vs 实盘口径
    labels = ["243定稿\n250日/冷却20/70%", "244集成\n9窗口/冷却10/70%"]
    old = [stats[0][1][2], stats[1][1][2]]
    new = [stats[0][2][2], stats[1][2][2]]
    xx = np.arange(2)
    ax[1, 0].bar(xx - 0.19, old, 0.36, color="#95a5a6", label="旧口径(池内预过滤 n1_ok)")
    ax[1, 0].bar(xx + 0.19, new, 0.36, color="#c0392b", label="实盘口径(T+1 高开就放弃)")
    for i in range(2):
        ax[1, 0].text(xx[i] - 0.19, old[i] + 0.06, f"{old[i]:.2f}", ha="center", fontsize=10)
        ax[1, 0].text(xx[i] + 0.19, new[i] + 0.06, f"{new[i]:.2f}", ha="center", fontsize=10,
                      color="#c0392b", fontweight="bold")
        dlt = new[i] - old[i]
        ax[1, 0].annotate(f"{dlt:+.2f}", (xx[i], max(old[i], new[i]) + 0.45),
                          ha="center", fontsize=11, color="#c0392b" if dlt > 0 else "#2c3e50",
                          fontweight="bold")
    ax[1, 0].axhline(metrics(base)[2], color="#2c3e50", ls="--", lw=1.4,
                     label=f"仅主策略 {metrics(base)[2]:.2f}")
    ax[1, 0].set_xticks(xx); ax[1, 0].set_xticklabels(labels, fontsize=9.5)
    ax[1, 0].set_ylim(0, 6.8); ax[1, 0].set_ylabel("卡玛")
    ax[1, 0].set_title("③ 扣掉未来信息(均 70% 仓): 243 定稿 -0.16, 集成 +0.12 —— 买不到就不买反而更好",
                       fontsize=11.5, loc="left")
    ax[1, 0].legend(fontsize=8.5, loc="upper left")

    # ④ 净值
    x = pd.to_datetime(s_ens_new.index.astype(str))
    ax[1, 1].plot(x, (1 + base / 100).cumprod().values, color="#7f8c8d", lw=1.4,
                  label=f"仅主策略 {metrics(base)[2]:.2f}")
    ax[1, 1].plot(x, (1 + s_243_new / 100).cumprod().values, color="#2c3e50", lw=1.5,
                  label=f"243定稿(实盘口径) {stats[0][2][2]:.2f}")
    ax[1, 1].plot(x, (1 + s_ens_new / 100).cumprod().values, color="#c0392b", lw=2.1,
                  label=f"244集成(实盘口径) {stats[1][2][2]:.2f}")
    ax[1, 1].set_yscale("log")
    ax[1, 1].set_ylabel("净值(对数)"); ax[1, 1].grid(alpha=0.25)
    ax[1, 1].legend(fontsize=9, loc="upper left")
    ax[1, 1].set_title("④ 11 年净值: 集成把年化从 101% 提到 123%, 回撤还小 0.6pp",
                       fontsize=11.5, loc="left")

    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(OUT / "第244轮_不用挑参数但要扣未来信息.png", bbox_inches="tight")
    print("图已保存")


if __name__ == "__main__":
    main()
