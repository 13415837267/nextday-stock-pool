"""第225轮图: 参数层已到平台, 收益增量在参数之外

① 六个方向的优化尝试 —— 全部落在噪声内或为负, 说明"再调参数"没有空间
② 分年对比: 唯一"改进"(neg_bias20=0.85) 17 年仅胜 8 年, 2021 起 6 年仅胜 3 年 → 噪声
③ 3 万本金的真实杠杆: 成本纪律 / 持续投入 的三年结果
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP"]
plt.rcParams["axes.unicode_minus"] = False

RED, GREEN, BLUE, GRAY, ORANGE = "#c0392b", "#27ae60", "#2c6fbb", "#8a8a8a", "#e08a1e"


def panel1(ax: plt.Axes) -> None:
    """六方向优化尝试的最优 Δ年化。"""
    names = ["大盘风控\n(64 组)", "选股权重\n(全史验收)", "仓位折减\n0.4→0.6",
             "持仓只数\nK=2 ", "卖出顺延\n(高浮盈笔)", "情绪维度\n(第218轮)"]
    vals = [0.0, -2.27, 1.87, -42.09, -0.83, 0.0]
    notes = ["现行即最优", "17年8胜", "750日卡玛降", "收益腰斩", "日内已证伪", "相关性−0.02"]
    cols = [GRAY if abs(v) < 2.5 else (RED if v > 0 else GREEN) for v in vals]
    y = np.arange(len(names))[::-1]
    bars = ax.barh(y, vals, height=0.55, color=cols, alpha=0.85, edgecolor="k", lw=0.6)
    for b, v, n in zip(bars, vals, notes):
        # 一律向右排: 负值条很长, 文字放条内右侧才不会压到 y 轴标签
        ax.text(v + 1.8, b.get_y() + b.get_height() / 2, f"{v:+.1f}pp  {n}",
                va="center", ha="left", fontsize=8.2)
    ax.axvline(0, color="k", lw=0.9)
    ax.set_yticks(y, names, fontsize=8.6)
    ax.set_xlim(-58, 34)
    ax.set_xlabel("相对现行 111.76% 的年化变化 (pp)", fontsize=9)
    ax.set_title("① 参数层六个方向：没有一个能稳定跑赢现行", fontsize=10.5, pad=8)
    ax.grid(axis="x", alpha=0.25)
    ax.text(0.02, 0.03, "灰色带内(|Δ|<2.5pp)视为噪声；红线为正但不通过全史验收",
            transform=ax.transAxes, fontsize=7.6, color="#555")


def panel2(ax: plt.Axes) -> None:
    """分年 A vs B。"""
    d = pd.read_csv(ROOT / "output" / "w225g_yearly.csv")
    x = np.arange(len(d))
    w = 0.38
    ax.bar(x - w / 2, d["收益%_A"], w, color=GRAY, alpha=0.9, label="现行 neg_bias20=0.70")
    ax.bar(x + w / 2, d["收益%_B"], w, color=BLUE, alpha=0.9, label="训练值 0.85")
    ax.set_xticks(x, [str(int(v)) for v in d["年"]], rotation=45, fontsize=7.6)
    ax.set_ylabel("当年收益 (%)", fontsize=9)
    ax.set_yscale("symlog", linthresh=100)
    ax.set_ylim(0, 620)
    ax.set_title("② 唯一\"改进\"是窗口噪声：17 年仅胜 8 年", fontsize=10.5, pad=26)
    ax.legend(fontsize=8, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2,
              frameon=False)
    ax.grid(axis="y", alpha=0.25)
    n_win = int((d["差"] > 0).sum())
    ax.text(0.02, 0.96, f"B 胜 {n_win}/{len(d)} 年 · 2021起 3/6 年\n累计 739.7% vs 731.7%",
            transform=ax.transAxes, ha="left", va="top", fontsize=7.8, color="#333",
            bbox=dict(fc="white", ec="#bbb", alpha=0.92, boxstyle="round,pad=0.28"))


def panel3(ax: plt.Axes) -> None:
    """3 万本金三年后的真实差别。"""
    save = 2.4
    labels, vals, cols = [], [], []
    # 纯复利 60%
    v = 3.0 * 1.6 ** 3
    labels.append("纯复利\n60%"); vals.append(v); cols.append(GRAY)
    v2 = v + save * ((1.6 ** 2) + 1.6 + 1)
    labels.append("复利 + 每年存 2.4 万"); vals.append(v2); cols.append(BLUE)
    v3 = 3.0 * 2.307 ** 3
    labels.append("成本纪律\n130.7%"); vals.append(v3); cols.append(ORANGE)
    v4 = v3 + save * ((2.307 ** 2) + 2.307 + 1)
    labels.append("两者都做"); vals.append(v4); cols.append(RED)

    x = np.arange(len(vals))
    bars = ax.bar(x, vals, width=0.58, color=cols, alpha=0.88, edgecolor="k", lw=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.1f} 万",
                ha="center", fontsize=9.5, fontweight="bold")
    ax.axhline(12.0, color="#666", ls=":", lw=1.2)
    ax.legend(handles=[plt.Line2D([], [], color="#666", ls=":", lw=1.3, label="回本 12 万")],
              loc="upper left", fontsize=8, frameon=False)
    ax.set_xticks(x, labels, fontsize=8.2)
    ax.set_ylabel("3 万元 · 三年后 (万元)", fontsize=9)
    ax.set_ylim(0, 74)
    ax.set_title("③ 参数之外的两个杠杆，都比调参大", fontsize=10.5, pad=8)
    ax.grid(axis="y", alpha=0.25)


def main() -> None:
    fig = plt.figure(figsize=(14.4, 4.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.12, 1.16, 0.95], wspace=0.30)
    panel1(fig.add_subplot(gs[0, 0]))
    panel2(fig.add_subplot(gs[0, 1]))
    panel3(fig.add_subplot(gs[0, 2]))
    out = ROOT / "第225轮_参数到顶之后.png"
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    print("saved", out)


if __name__ == "__main__":
    main()
