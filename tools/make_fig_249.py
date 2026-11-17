"""第249轮 图: 合入后的三条腿 / 满仓假象 / 逐年贡献"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
RED, DARK, GREY, BLUE, ORANGE, GREEN = "#c0392b", "#2c3e50", "#7f8c8d", "#2980b9", "#d68910", "#1e8449"


def to_ret(eq: list[dict]) -> pd.Series:
    s = pd.Series({int(x["d"].replace("-", "")): float(x["nav"]) for x in eq}).sort_index()
    return (s.pct_change().dropna() * 100).rename("r")


def m(r: pd.Series) -> tuple[float, float, float]:
    nav = (1 + r / 100).cumprod()
    ann = nav.iloc[-1] ** (252 / len(r)) - 1
    dd = (nav / nav.cummax() - 1).min()
    return round(ann * 100, 2), round(dd * 100, 2), round(ann * 100 / abs(dd * 100), 2)


def main() -> None:
    j = json.loads((OUT / "w249_combo_2500.json").read_text(encoding="utf-8"))
    rs = {k: to_ret(v["equity"]) for k, v in j.items()}
    ks = list(rs)
    fig, ax = plt.subplots(2, 2, figsize=(15.4, 10.4))

    # 1 净值
    a1 = ax[0, 0]
    for k, cl, lw in ((ks[0], GREY, 1.5), (ks[1], BLUE, 1.9), (ks[2], RED, 1.9)):
        s = rs[k].copy()
        s.index = pd.to_datetime(s.index.astype(str))
        a1.plot(s.index, (1 + s / 100).cumprod(), color=cl, lw=lw,
                label=f"{k}  ({m(rs[k])[2]:.2f})")
    a1.set_yscale("log")
    a1.set_title("一个账户三条腿（2500 日，对数轴）", fontsize=12, color=DARK)
    a1.legend(fontsize=8, loc="upper left")
    a1.grid(alpha=0.25)

    # 2 逐年贡献
    a2 = ax[0, 1]
    yr = pd.DataFrame({k: v.groupby(v.index // 10000).apply(
        lambda x: (1 + x / 100).prod() - 1) * 100 for k, v in rs.items()})
    cb = yr[ks[1]] - yr[ks[0]]
    cc = yr[ks[2]] - yr[ks[0]]
    xs = np.arange(len(yr))
    a2.bar(xs - 0.2, cb.values, 0.4, color=BLUE, label="补位填空档 − 主策略")
    a2.bar(xs + 0.2, cc.values, 0.4, color=RED, alpha=0.85, label="再加让位 − 主策略")
    a2.axhline(0, color=DARK, lw=1)
    a2.set_xticks(xs)
    a2.set_xticklabels([str(k)[2:] for k in yr.index], fontsize=9)
    a2.set_ylabel("年度收益差（pp）")
    a2.set_title("补位在 2016-2019 的低效期有用，高效期基本是加杠杆", fontsize=12, color=DARK)
    a2.legend(fontsize=8, loc="upper left")
    a2.grid(axis="y", alpha=0.25)

    # 3 口径对照
    a3 = ax[1, 0]
    labs = ["主策略\n拼接口径", "主策略\n真实账户", "补位贡献\n拼接口径", "补位贡献\n真实账户"]
    vals = [3.57, m(rs[ks[0]])[2], 5.76 - 3.57, m(rs[ks[1]])[2] - m(rs[ks[0]])[2]]
    cols = [GREY, GREY, ORANGE, GREEN]
    b = a3.bar(range(4), vals, 0.58, color=cols)
    a3.set_xticks(range(4))
    a3.set_xticklabels(labs, fontsize=9)
    a3.set_ylabel("卡玛")
    a3.set_title("248 的 5.76 建在「每笔满仓」上；真实引擎回撤只有 -18.6%", fontsize=12, color=DARK)
    for bb, v in zip(b, vals):
        a3.text(bb.get_x() + bb.get_width() / 2, v + 0.08, f"{v:+.2f}" if v < 3 else f"{v:.2f}",
                ha="center", fontsize=11, fontweight="bold", color=DARK)
    a3.axhline(0, color=DARK, lw=0.8)
    a3.annotate("几乎为零", xy=(3, 0.12), xytext=(2.9, 1.6), fontsize=9, color=GREEN,
                ha="center", arrowprops=dict(arrowstyle="->", color=GREEN, lw=1))
    a3.grid(axis="y", alpha=0.25)
    a3.set_ylim(0, 7)

    # 4 空仓率与逐腿
    a4 = ax[1, 1]
    rows = []
    for k in ks:
        r = rs[k]
        nz = (r.abs() > 0.01)
        rows.append({"腿": k, "有仓日": int(nz.sum()), "空仓日": int((~nz).sum()),
                     "日均(有仓)": r[nz].mean(), "年化": m(r)[0]})
    df = pd.DataFrame(rows)
    ypos = np.arange(len(df))[::-1]
    a4.barh(ypos, df["有仓日"], 0.55, color=RED, alpha=0.85, label="有仓日")
    a4.barh(ypos, df["空仓日"], 0.55, left=df["有仓日"], color=GREY, alpha=0.45, label="空仓日")
    a4.set_yticks(ypos)
    a4.set_yticklabels([f"{r['腿']}\n年化 {r['年化']:.0f}%  有仓日 {r['有仓日']}"
                        for _, r in df.iterrows()], fontsize=8.5)
    a4.set_xlabel("交易日")
    a4.set_title("补位把空仓率从 59% 压到 42% —— 这是它真正的贡献", fontsize=12, color=DARK)
    a4.legend(fontsize=8, loc="lower right")
    a4.grid(axis="x", alpha=0.25)

    fig.suptitle("第249轮  合入主策略：248 的 5.76 是满仓假象，真实 +0.12",
                 fontsize=15, fontweight="bold", color=DARK, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(ROOT / "第249轮_合入主策略.png", dpi=150, bbox_inches="tight")
    print("saved")
    for k in ks:
        print(k, m(rs[k]))


if __name__ == "__main__":
    main()
