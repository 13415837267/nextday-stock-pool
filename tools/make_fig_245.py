"""第245轮 图: 把 T日选好再祈祷低开 换成 T+1 开盘重选"""
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

    sig = pd.read_parquet(POOL / "signals.parquet")
    sig["trade_date"] = sig["trade_date"].astype("int64")
    w = sig.copy()
    for W in WINS:
        px = pd.read_parquet(POOL / f"W{W}.parquet").rename(columns={"dh": f"dh{W}"})
        w = w.merge(px, on=["ts_code", "trade_date"], how="left")
        del px
    dhcols = [f"dh{W}" for W in WINS]
    w = w.dropna(subset=dhcols)
    w["n1_ok"] = w["n1_ok"].fillna(False)
    w["y_net"] = w["y"] - COST

    def prep(g: pd.DataFrame) -> pd.DataFrame:
        den = {c: max(int(g[c].notna().sum()), 1) for c in dhcols}
        rk = pd.DataFrame({c: g[c].rank(ascending=True) / den[c] for c in dhcols})
        g = g.copy()
        g["mr"] = rk.mean(axis=1, skipna=True)
        return g

    w = pd.concat([prep(g) for _, g in w.groupby("trade_date")], ignore_index=True)
    byday = {dt: g for dt, g in w.groupby("trade_date")}

    def run(cd: int, rule: str):
        last, v = {}, {}
        for dt in sorted(byday):
            g = byday[dt]
            if rule == "reallot":
                lo = g[g.n1_ok]
                g = lo if len(lo) else g
            cand = g.sort_values(["mr", "amt_yi"], ascending=[True, False])
            ok = [(c not in last) or (dayrank[dt] - last[c] >= cd) for c in cand["ts_code"]]
            sub = cand[np.array(ok)]
            if len(sub):
                cand = sub
            if not len(cand):
                continue
            r = cand.iloc[0]
            if pd.isna(r["mr"]) or pd.isna(r["y"]):
                continue
            if rule == "pick_then_skip" and not bool(r["n1_ok"]):
                v[dt] = 0.0
                continue
            v[dt] = float(r["y_net"])
            last[r["ts_code"]] = dayrank[dt]
        return pd.Series(v).sort_index()

    v244 = run(10, "pick_then_skip")
    v245 = run(10, "reallot")
    s244 = base.copy()
    s244[v244.index] += v244.values * 0.70
    s245 = base.copy()
    s245[v245.index] += v245.values * 0.50
    m244, m245 = metrics(s244), metrics(s245)
    print(f"244 {m244}  245 {m245}")

    fig, ax = plt.subplots(2, 2, figsize=(15.5, 10.4), dpi=150)
    fig.suptitle("第245轮  T 日选好再祈祷低开 -> 改成 T+1 开盘在低开票里重选",
                 fontsize=16.5, fontweight="bold", y=0.975)

    # ① 两种执行方式
    labels = ["244: T日选top1\n高开就空仓", "245: T+1开盘\n低开票里重选"]
    xx = np.arange(2)
    ann = [m244[0], m245[0]]
    ax[0, 0].bar(xx - 0.18, ann, 0.34, color="#95a5a6", label="年化 %")
    ax2 = ax[0, 0].twinx()
    ax2.bar(xx + 0.18, [m244[2], m245[2]], 0.34, color="#c0392b", label="卡玛")
    for i in range(2):
        ax[0, 0].text(xx[i] - 0.18, ann[i] + 2, f"{ann[i]:.1f}%", ha="center", fontsize=10.5)
        ax2.text(xx[i] + 0.18, [m244[2], m245[2]][i] + 0.1,
                 f"{[m244[2], m245[2]][i]:.2f}", ha="center", fontsize=10.5, color="#c0392b",
                 fontweight="bold")
    ax[0, 0].set_xticks(xx)
    ax[0, 0].set_xticklabels(labels, fontsize=10)
    ax[0, 0].set_ylabel("年化 %", color="#555")
    ax2.set_ylabel("卡玛", color="#c0392b")
    ax2.set_ylim(0, 6.6)
    ax[0, 0].set_ylim(0, 175)
    ax[0, 0].set_title("① 同回撤档(约 -24~-25%): 年化 123->138, 卡玛 5.21->5.49",
                       fontsize=11.5, loc="left")

    # ② 参数面
    grid = pd.read_csv(OUT / "w245_grid.csv")
    pv = grid.pivot_table(index="C", columns="P", values="卡玛")
    im = ax[0, 1].imshow(pv.values, cmap="RdYlGn", aspect="auto", vmin=3.8, vmax=5.6)
    ax[0, 1].set_xticks(range(len(pv.columns)))
    ax[0, 1].set_xticklabels([f"{p:.0%}" for p in pv.columns])
    ax[0, 1].set_yticks(range(len(pv.index)))
    ax[0, 1].set_yticklabels(pv.index)
    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            ax[0, 1].text(j, i, f"{pv.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ci, cj = list(pv.index).index(10), list(pv.columns).index(0.50)
    ax[0, 1].plot(cj, ci, "*", ms=20, color="#c0392b", mec="white", mew=1.5)
    ax[0, 1].set_xlabel("仓位"); ax[0, 1].set_ylabel("冷却(交易日)")
    ax[0, 1].set_title("② 参数面: 冷却 5~30 x 仓位 40~50% 是平台, 不是孤峰",
                       fontsize=11.5, loc="left")
    cb = plt.colorbar(im, ax=ax[0, 1], fraction=0.045)
    cb.set_label("卡玛", rotation=270, labelpad=13)

    # ③ 救回的 82 天
    only = v245.index.difference(v244[v244 != 0].index)
    x = v245.loc[only].values
    ax[1, 0].hist(x, bins=22, color="#2980b9", alpha=0.75, edgecolor="white")
    ax[1, 0].axvline(0, color="#333", lw=1.2)
    ax[1, 0].axvline(x.mean(), color="#c0392b", ls="--", lw=1.8,
                     label=f"均值 {x.mean():.2f}%")
    ax[1, 0].set_xlabel("单笔收益 %"); ax[1, 0].set_ylabel("次数")
    ax[1, 0].set_title(f"③ 244 放弃的 {len(x)} 天, 245 全做成了: 单笔 {x.mean():.2f}%, "
                       f"胜率 {(x > 0).mean() * 100:.0f}%, 合计 {x.sum():.0f}pp",
                       fontsize=11.5, loc="left")
    ax[1, 0].legend(fontsize=9)

    # ④ 净值
    xs = pd.to_datetime(s245.index.astype(str))
    for s, c, lw, lab in ((base, "#7f8c8d", 1.4, f"仅主策略 {metrics(base)[2]:.2f}"),
                          (s244, "#2c3e50", 1.5, f"244 T日选 {m244[2]:.2f}"),
                          (s245, "#c0392b", 2.1, f"245 T+1重选 {m245[2]:.2f}")):
        ax[1, 1].plot(xs, (1 + s / 100).cumprod().values, color=c, lw=lw, label=lab)
    ax[1, 1].set_yscale("log")
    ax[1, 1].set_ylabel("净值(对数)"); ax[1, 1].grid(alpha=0.25)
    ax[1, 1].legend(fontsize=9, loc="upper left")
    ax[1, 1].set_title(f"④ 年化 {m245[0]:.0f}% / 回撤 {m245[1]:.1f}% / 卡玛 {m245[2]:.2f}, "
                       f"11 年 10 正 1 弱",
                       fontsize=11.5, loc="left")

    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(OUT / "第245轮_T+1开盘重选.png", bbox_inches="tight")
    print("图已保存")


if __name__ == "__main__":
    main()
