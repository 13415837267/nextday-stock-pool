"""第246轮 图: 每年 1 月没有信号 / 并列触发 / 单票限次"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
COST, SLIP = 0.10, 0.20


def metrics(r: pd.Series) -> tuple[float, float, float]:
    nav = (1 + r / 100).cumprod()
    ann = nav.iloc[-1] ** (244 / len(r)) - 1
    dd = (nav / nav.cummax() - 1).min()
    return round(ann * 100, 2), round(dd * 100, 2), round(ann * 100 / abs(dd * 100), 2)


def main() -> None:
    tr = pd.read_csv(OUT / "w232_trades.csv")
    all_dates = sorted(int(p.stem.replace(".csv", "")) for p in
                       (ROOT / "data" / "panel").glob("*.csv.gz"))
    cal = np.array([x for x in all_dates if x >= int(tr.signal.min())])
    base = pd.Series(0.0, index=cal)
    for dt, g in tr.groupby("signal"):
        base[int(dt)] = g["ret"].mean()

    A = pd.read_csv(OUT / "w246k_picksA.csv")
    B1 = pd.read_csv(OUT / "w246k_picks.csv")
    p245 = pd.read_csv(OUT / "w245_picks.csv", header=0)
    p245.columns = ["signal", "y"]

    def nav_of(c: pd.DataFrame, P: float) -> pd.Series:
        s = pd.Series((c["y"].values - COST - SLIP) * P, index=c.trade_date.values)
        s = s.sort_index()
        sim = base.copy()
        sim[s.index] = sim[s.index] + s.values
        return (1 + sim / 100).cumprod()

    fig, ax = plt.subplots(2, 2, figsize=(15.5, 10.5))

    # ① 按月: 245 的 198 笔 vs 246 的 232 笔
    m245 = p245.signal // 100 % 100
    mA = A.trade_date // 100 % 100
    c245 = m245.value_counts().reindex(range(1, 13), fill_value=0)
    cA = mA.value_counts().reindex(range(1, 13), fill_value=0)
    x = np.arange(1, 13)
    w = 0.38
    b1 = ax[0, 0].bar(x - w / 2, c245.values, w, color="#bdc3c7", label="245 版本 198 笔")
    b2 = ax[0, 0].bar(x + w / 2, cA.values, w, color="#c0392b", label="246 修正后 232 笔")
    ax[0, 0].bar_label(b1, fontsize=7.5, padding=1, color="#555")
    ax[0, 0].bar_label(b2, fontsize=7.5, padding=1, color="#c0392b", fontweight="bold")
    ax[0, 0].set_xticks(x)
    ax[0, 0].annotate("1 月 2 → 31\n其余月份几乎不变", (1.55, 26), fontsize=10.5,
                      color="#c0392b", fontweight="bold",
                      bbox=dict(boxstyle="round,pad=0.3", fc="#fdf2f0", ec="#c0392b",
                                alpha=0.95))
    ax[0, 0].set_title("① build_features 按年 rolling —— 每年 1 月前 20 个交易日全是 NaN",
                       fontsize=12, loc="left")
    ax[0, 0].set_xlabel("月份"); ax[0, 0].set_ylabel("交易笔数")
    ax[0, 0].legend(fontsize=9, loc="upper right")
    ax[0, 0].set_ylim(0, 40)

    # ② 仓位曲线
    pos = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70]
    ka = [4.34, 4.52, 4.69, 4.83, 4.67, 4.53, 4.42, 4.23, 4.10]
    kb = [4.60, 4.84, 5.09, 5.29, 5.16, 5.05, 4.97, 4.84, 4.75]
    ax[0, 1].plot(np.array(pos) * 100, ka, "o-", color="#7f8c8d", lw=2, ms=6,
                  label="A  245 口径（只破位）")
    ax[0, 1].plot(np.array(pos) * 100, kb, "o-", color="#c0392b", lw=2.2, ms=6,
                  label="B' 并列（+ 个股温和崩）")
    ax[0, 1].axvline(40, color="#c0392b", ls=":", lw=1.2, alpha=0.7)
    ax[0, 1].annotate("取 40%\nB' 5.16 / A 4.67", (40, 4.15), (46, 4.28), fontsize=9.5,
                      color="#2c3e50",
                      arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=1.1),
                      bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none",
                                alpha=0.9))
    ax[0, 1].set_title("② 仓位: 30~40% 是一块平台, 再往上回撤吃掉收益", fontsize=12, loc="left")
    ax[0, 1].set_xlabel("单吊仓位 (%)"); ax[0, 1].set_ylabel("卡玛")
    ax[0, 1].legend(fontsize=9, loc="lower left")
    ax[0, 1].grid(alpha=0.25)

    # ③ 三条触发路径各自的贡献
    g = B1.y - COST - SLIP
    mild = (B1.n1_gap > -9.5) & (B1.n1_gap <= -5)
    lid = B1.n1_gap <= -9.5
    shallow = ~mild & ~lid
    lab = ["浅低开 > -5%", "温和深低开\n-9.5% ~ -5%", "跌停开\n< -9.5%（不做）"]
    yv = [g[shallow].mean(), g[mild].mean(), g[lid].mean()]
    ns = [int(shallow.sum()), int(mild.sum()), int(lid.sum())]
    cols = ["#2c3e50", "#c0392b", "#bdc3c7"]
    bars = ax[1, 0].bar(lab, yv, color=cols, width=0.55)
    for b, v, n in zip(bars, yv, ns):
        ax[1, 0].annotate(f"{v:.2f}%\n{n} 笔", (b.get_x() + b.get_width() / 2, v + 0.12),
                          ha="center", fontsize=10,
                          color="#c0392b" if v > 2 else "#2c3e50",
                          fontweight="bold" if v > 2 else "normal")
    ax[1, 0].axhline(0, color="#333", lw=0.8)
    ax[1, 0].set_ylabel("单笔收益 (%) 已扣成本+滑点")
    ax[1, 0].set_title("③ 按买入那刻的缺口分档: 温和深低开 8.1%, 跌停开反而只有 3.0%",
                       fontsize=12, loc="left")
    ax[1, 0].set_ylim(0, 10)

    # ④ 净值
    for c, colr, wd, lb in ((A, "#7f8c8d", 1.5, "A  40% 仓 · 232 笔 · 119.1% / 4.67"),
                            (B1, "#c0392b", 2.2, "B  40% 仓 · 256 笔 · 131.7% / 5.16")):
        nv = nav_of(c, 0.40)
        ax[1, 1].plot(range(len(nv)), nv.values, color=colr, lw=wd, label=lb)
    s245 = pd.Series((p245.y.values - COST - SLIP) * 0.40,
                     index=p245.signal.values).sort_index()
    sim245 = base.copy()
    sim245[s245.index] = sim245[s245.index] + s245.values
    ax[1, 1].plot(range(len(sim245)), (1 + sim245 / 100).cumprod().values,
                  color="#2980b9", lw=1.3, ls="--",
                  label="245 口径 40% 仓 · 198 笔（漏 1 月）· 122.0% / 5.17")
    ax[1, 1].set_yscale("log")
    ax[1, 1].set_title("④ 同为 40% 仓: B 补回 1 月后仍高于 245 的漏月样本", fontsize=12,
                       loc="left")
    ax[1, 1].set_xlabel(f"交易日  {cal[0]} → {cal[-1]}")
    ax[1, 1].set_ylabel("净值（对数）")
    ax[1, 1].legend(fontsize=9, loc="upper left")
    ax[1, 1].grid(alpha=0.25, which="both")

    fig.suptitle("第246轮  每年 1 月没信号的 bug, 与「个股自己崩」这条新触发",
                 fontsize=14.5, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    p = OUT / "第246轮_一月无信号与新触发.png"
    fig.savefig(p, dpi=155)
    print("saved", p)


if __name__ == "__main__":
    main()
