"""第 201 轮配图：池深是唯一双窗口单调的调节变量 + 敞口前沿。"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]

# 池深分档（trades_201_*.csv）
PN = {
    1500: dict(lab=["≤10", "10~20", "20~40", "40~80", "80+"],
               mean=[0.264, 0.787, 1.386, 2.304, 2.202],
               win=[41.9, 52.5, 54.5, 64.7, 61.9],
               n=[31, 80, 112, 68, 42]),
    3000: dict(lab=["≤10", "10~20", "20~40", "40~80", "80+"],
               mean=[0.418, 0.707, 1.172, 1.921, 2.630],
               win=[43.2, 51.1, 54.5, 62.9, 63.5],
               n=[74, 139, 178, 116, 74]),
}

# 敞口前沿（exp_exposure_201_*.json）
def load(days):
    f = ROOT / "output" / f"exp_exposure_201_{days}.json"
    rows = {r["name"]: r for r in json.load(open(f))}
    order = ["base", "pool41_x1.2", "pool41_x1.3", "pool41_x1.5"]
    return [(rows[k]["cap"], rows[k]["年化收益率%"], rows[k]["最大回撤%"], rows[k]["卡玛比率"],
             rows[k]["avg_pos"]) for k in order]


def main():
    fig = plt.figure(figsize=(13.4, 9.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05], hspace=0.46, wspace=0.24)

    # (a) 池深分档
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(5)
    w = 0.38
    c1, c2 = "#c0392b", "#2c7fb8"
    for i, (days, off) in enumerate([(1500, -w / 2), (3000, w / 2)]):
        d = PN[days]
        col = c1 if days == 1500 else c2
        ax.bar(x + off, d["mean"], w, color=col, alpha=.88,
               label=f"{days} 日")
        for xx, yy in zip(x + off, d["mean"]):
            ax.text(xx, yy + .05, f"{yy:.2f}", ha="center", va="bottom", fontsize=7.6, color=col)
    ax.set_xticks(x)
    ax.set_xticklabels(PN[1500]["lab"], fontsize=9)
    ax.set_xlabel("信号日达标票数（池深）", fontsize=9, labelpad=2)
    ax.set_ylabel("单笔平均收益 %", fontsize=9)
    ax.set_ylim(0, 3.05)
    ax.axvspan(2.5, 4.5, color="#f2d7d5", alpha=.45, zorder=0)
    ax.text(3.5, 2.86, "现行加仓区（≥41）", ha="center", fontsize=8.2, color="#8a3b34")
    ax.legend(fontsize=8, loc="upper left", frameon=False)
    ax.set_title("(a) 池深：两个窗口、五个档位，单调", fontsize=10, loc="left")

    # (b) 年化 vs 平均敞口
    ax2 = fig.add_subplot(gs[0, 1])
    for days, col, mk in [(1500, c1, "o"), (3000, c2, "s")]:
        f = load(days)
        xs = [r[4] for r in f]
        ax2.plot(xs, [r[1] for r in f], "-" + mk, color=col, lw=1.8, ms=6, label=f"{days} 日")
        for r in f:
            ax2.annotate(f"{r[0]:g}×", (r[4], r[1]), textcoords="offset points",
                         xytext=(0, 9), ha="center", fontsize=7.8, color=col)
            ax2.annotate(f"回撤{r[2]:.1f}%", (r[4], r[1]), textcoords="offset points",
                         xytext=(0, -13), ha="center", fontsize=6.8, color=col, alpha=.8)
    ax2.set_xlabel("平均敞口（× 权益）", fontsize=9, labelpad=2)
    ax2.set_ylabel("年化收益 %", fontsize=9)
    ax2.set_ylim(70, 168)
    ax2.set_xlim(0.79, 1.04)
    ax2.grid(alpha=.22, lw=.6)
    ax2.legend(fontsize=8, frameon=False, loc="upper left")
    ax2.set_title("(b) 年化随敞口上行，回撤几乎不动", fontsize=10, loc="left")

    # (c) 卡玛
    ax3 = fig.add_subplot(gs[1, 0])
    caps = [r[0] for r in load(1500)]
    for days, col, mk in [(1500, c1, "o"), (3000, c2, "s")]:
        f = load(days)
        ax3.plot(caps, [r[3] for r in f], "-" + mk, color=col, lw=1.8, ms=6, label=f"{days} 日")
        for r in f:
            ax3.annotate(f"{r[3]:.2f}", (r[0], r[3]), textcoords="offset points",
                         xytext=(0, 9), ha="center", fontsize=7.8, color=col)
    ax3.axhline(load(1500)[0][3], color=c1, ls=":", lw=.9)
    ax3.axhline(load(3000)[0][3], color=c2, ls=":", lw=.9)
    ax3.set_xticks(caps)
    ax3.set_xticklabels([f"{c:g}×\n{'基准' if c == 1.0 else '≥41 池'}" for c in caps], fontsize=8.2)
    ax3.set_ylabel("卡玛比率", fontsize=9)
    ax3.set_ylim(3.6, 7.7)
    ax3.grid(alpha=.22, lw=.6)
    ax3.legend(fontsize=8, frameon=False, loc="center right")
    ax3.set_title("(c) 卡玛随敞口单调上行（虚线为基准）", fontsize=10, loc="left")

    # (d) 同一批交易的敞口分布
    ax4 = fig.add_subplot(gs[1, 1])
    labels = ["≤10", "10~20", "20~40", "40~80", "80+"]
    cur = [0.30, 0.50, 1.00, 1.00, 1.00]
    new = [0.30, 0.50, 1.00, 1.30, 1.30]
    xx = np.arange(5)
    ax4.bar(xx - .19, cur, .36, color="#95a5a6", label="现行 1.0× 上限")
    ax4.bar(xx + .19, new, .36, color="#c0392b", label="本轮 1.3× 上限")
    for xi, a, b in zip(xx, cur, new):
        ax4.text(xi - .19, a + .03, f"{a:.1f}", ha="center", fontsize=7.6, color="#555")
        ax4.text(xi + .19, b + .03, f"{b:.1f}", ha="center", fontsize=7.6, color="#c0392b")
    ax4.set_xticks(xx)
    ax4.set_xticklabels(labels, fontsize=9)
    ax4.set_xlabel("信号日达标票数（池深）", fontsize=9, labelpad=2)
    ax4.set_ylabel("目标仓位（× 权益）", fontsize=9)
    ax4.set_ylim(0, 1.55)
    ax4.axhline(1.0, color="#333", ls="--", lw=.9)
    ax4.text(0.02, 1.03, "1.0× = 不用融资", fontsize=7.8, color="#333")
    ax4.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(0, 1.0))
    ax4.set_title("(d) 改动只有一格：深池从 1.0 放到 1.3", fontsize=10, loc="left")

    fig.suptitle("第 201 轮：池深是唯一还要得起的边际，只改一格仓位", fontsize=13, x=.012, ha="left", y=.985)
    fig.text(.012, .008,
             "数据：主策略 1500 日 355 笔 / 3000 日 717 笔；敞口以 pos_frac_cap 表达，leverage 固定 1.0（避免叠乘），"
             "融资利息按 5.5%/年逐日计提。",
             fontsize=7.8, color="#555")
    out = ROOT.parent / "第201轮_只改一格仓位.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
