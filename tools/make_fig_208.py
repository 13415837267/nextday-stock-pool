"""第208轮配图: 左=四条净值曲线(对数轴), 右=条件年化(把"空仓"拆开看).

左图必须用对数轴: 3,488× 与 19.3× 放在线性轴上, 低的那三条会贴成一条平线。
右图是本轮的核心 —— 不是"两条腿相关低", 而是"每条腿只在**别人也在仓**的日子里赚钱"。
所以按"对方在仓/空仓"把日收益切开, 用正负柱直接显示符号翻转。
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

C1, C2, C3, C4 = "#2F6F9F", "#C0392B", "#7F8C8D", "#E67E22"


def xlab(keys: list[str]) -> list[str]:
    return [f"{k[:4]}-{k[4:]}" for k in keys]


def main() -> None:
    d = json.loads((OUT / "exp_pool_208_3645.json").read_text(encoding="utf-8"))
    cv = d["curves"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15.4, 6.4),
                                  gridspec_kw={"width_ratios": [1.55, 1]})

    # ---------------- 左: 净值曲线 ----------------
    n_main = d["legs"]["main"]["总倍数"]
    n_half = d["shared_half"]["总倍数"]
    n_sh = d["shared"]["总倍数"]
    n_g = d["grid"]["60/40"]["总倍数"]
    series = [("main", "纯主腿 100%", C1, 2.4, "-", n_main),
              ("shared_half", "共享池 · 副腿半仓", C4, 1.7, "-.", n_half),
              ("shared", "共享池 · 主腿优先", C2, 1.9, "-", n_sh),
              ("grid_60_40", "切分 60/40（197 轮最优）", C3, 1.7, "--", n_g)]
    keys = list(cv["main"].keys())
    xs = range(len(keys))
    for k, lab, col, lw, ls, nmul in series:
        # 两腿初始本金不同(主腿 100 万, 副腿 10 万), 先各自归一到"10 万起步"再画,
        # 否则主腿曲线会整体高一个数量级, 看着像"收益更高"。
        raw = [cv[k][x] for x in keys]
        ys = [v / raw[0] * 1e5 for v in raw]
        ax.plot(list(xs), ys, color=col, lw=lw, ls=ls,
                label=f"{lab}　{nmul:,.0f}×")

    ax.set_yscale("log")
    ax.set_ylim(8e4, 9e8)
    ticks = list(range(0, len(keys), 24))
    ax.set_xticks(ticks)
    ax.set_xticklabels([xlab(keys)[i] for i in ticks], rotation=0, fontsize=8.6)
    ax.set_yticks([1e5, 1e6, 1e7, 1e8])
    ax.set_yticklabels(["10 万", "100 万", "1,000 万", "1 亿"])
    ax.minorticks_off()
    ax.set_ylabel("账户净值（初始 10 万，对数轴）", fontsize=10)
    ax.set_title("全史 3,645 个交易日：把两腿的钱并成一个池子，反而更慢",
                 fontsize=12.5, fontweight="bold", pad=10)
    ax.grid(alpha=0.25, ls=":", which="both")
    ax.legend(loc="lower right", fontsize=9.4, framealpha=0.94)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # ---------------- 右: 条件年化 ----------------
    sp = d["split"]
    groups = [
        ("副腿\n（主腿在仓日）", sp["sat0_ann_when_main_in"], "+"),
        ("副腿\n（主腿空仓日）", sp["sat0_ann_when_main_out"], "−"),
        ("主腿\n（副腿在仓日）", sp["main_ann_when_sat_in"], "+"),
        ("主腿\n（副腿空仓日）", sp["main_ann_when_sat_out"], "−"),
    ]
    x = [0, 1, 3, 4]
    vals = [g[1] for g in groups]
    cols = [C1 if v > 0 else C2 for v in vals]
    ax2.bar(x, vals, width=0.62, color=cols, alpha=0.9, zorder=3)
    ax2.axhline(0, color="#333", lw=1.0, zorder=4)
    for xi, v in zip(x, vals):
        ax2.annotate(f"{v:+.1f}%", xy=(xi, v), xytext=(0, 5 if v > 0 else -14),
                     textcoords="offset points", ha="center", fontsize=10,
                     fontweight="bold", color=C1 if v > 0 else C2)
    ax2.set_xticks(x)
    ax2.set_xticklabels([g[0] for g in groups], fontsize=9.4)
    ax2.set_ylim(-18, 105)
    ax2.set_ylabel("折算年化收益（按日收益均值 × 250）", fontsize=10)
    ax2.set_title("钱不是没地方去：对方空仓的日子，自己也赚不到钱",
                  fontsize=11.6, fontweight="bold", pad=10)
    ax2.grid(axis="y", alpha=0.25, ls=":", zorder=0)
    ax2.text(0.5, 96, "并集满仓率仅 58.3%（不是 100%）", ha="center", fontsize=9.2,
             color="#555", style="italic")
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)

    fig.tight_layout()
    p = OUT / "第208轮_共享资金池.png"
    fig.savefig(p, dpi=155, facecolor="white")
    print("写出", p)


if __name__ == "__main__":
    main()
