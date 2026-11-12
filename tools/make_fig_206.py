"""第206轮配图: 左=百倍的生存边界(成本), 右=三个结构方向全部碰壁."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

C1, C2, C3, C4 = "#2F6F9F", "#C0392B", "#7F8C8D", "#E67E22"


def main() -> None:
    cost = json.loads((OUT / "exp_cost_206_3645.json").read_text(encoding="utf-8"))
    cost["s10"] = {"总收益率%": 348714.9, "年化收益率%": 76.25, "最大回撤%": -22.60,
                   "卡玛比率": 3.37, "交易次数": 839}
    pts = {"s05": 0.05, "s10": 0.10, "s20": 0.20, "s30": 0.30, "s50": 0.50}
    xs = [pts[k] for k in pts]
    ys = [1 + cost[k]["总收益率%"] / 100 for k in pts]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.4, 5.3))

    # ---- 左: 滑点 -> 总倍数(对数) ----
    ax.set_xscale("log")
    ax.plot(xs, ys, "o-", color=C1, lw=2.2, ms=8, zorder=3)
    ax.set_yscale("log")
    ax.axhline(100, color=C2, ls="--", lw=1.5, zorder=2)
    ax.text(0.66, 112, "百倍线", color=C2, fontsize=9.6, ha="right", va="bottom")
    # 生存边界: 0.30(256x) 与 0.50(18.6x) 之间对数插值到 100x
    lo, hi = np.log(256.34), np.log(18.557)
    t = (lo - np.log(100)) / (lo - hi)
    bx = 0.30 + t * 0.20
    ax.axvline(bx, color=C4, ls=":", lw=1.6, zorder=2)
    ax.text(bx * 1.05, 3.0e4, f"百倍生存边界\n≈ {bx:.2f}%/边", color=C4,
            fontsize=9.4, va="top", ha="left")
    ax.axvline(0.10, color=C3, ls=":", lw=1.4, zorder=2)
    ax.text(0.098, 22, "现行假设\n0.10%", color=C3, fontsize=9.0, ha="right",
            va="center")
    # 曲线单调递减, 点正上方(同一 x, 更大 y)必不经过曲线 —— 唯一安全方向。
    # 两行排版把宽度减半, 最左点(0.05)才放得下。
    for k, x, y in zip(pts, xs, ys):
        lab = f"{y:,.0f}×" if y >= 100 else f"{y:.1f}×"
        txt = f"{lab} / {cost[k]['年化收益率%']:.0f}%"
        # log 轴把 5 个点均匀拉开后, 点正上方(曲线单调递减, 该处不经过)全部可用
        # 曲线单调递减: 点的"左侧+同高"整段都在曲线下方, 标签放这里永不压线
        if k == "s05":        # 最左点左侧空间不足, 反向放右侧(那段曲线已降到标签之下)
            ax.annotate(txt, (x, y), textcoords="offset points", xytext=(8, 8),
                        ha="left", va="bottom", fontsize=8.6, color=C1)
        else:
            ax.annotate(txt, (x, y), textcoords="offset points", xytext=(-8, 0),
                        ha="right", va="center", fontsize=8.6, color=C1)
    ax.set_xticks([0.05, 0.10, 0.20, 0.30, 0.50])
    ax.set_xticklabels(["0.05", "0.10", "0.20", "0.30", "0.50"])
    ax.minorticks_off()
    ax.set_xlabel("单边滑点假设（%/边，对数轴）", fontsize=10.5, labelpad=9)
    ax.set_ylabel("全史总倍数（对数轴）", fontsize=10.5)
    ax.set_xlim(0.035, 0.72)
    ax.set_ylim(9, 6.0e4)
    ax.grid(ls=":", color="#ddd", zorder=0)
    ax.set_title("百倍不是策略属性，是成本假设的因变量", fontsize=11.5, loc="left")

    # ---- 右: 三个结构方向 vs 基线(收益-回撤) ----
    d = json.loads((OUT / "exp_board_206_3645.json").read_text(encoding="utf-8"))
    p = json.loads((OUT / "exp_disp_206_3645.json").read_text(encoding="utf-8"))
    m = json.loads((OUT / "exp_lmt_206_3645.json").read_text(encoding="utf-8"))
    items = [
        ("现行（纯主板单吊）", 348714.9, -22.60, C1, 169),
        ("只创业板", d.get("cyb_only", {}).get("总收益率%", -100), d.get("cyb_only", {}).get("最大回撤%", -99), C2, 150),
        ("主板+创业板", d["cyb"]["总收益率%"], d["cyb"]["最大回撤%"], C2, 120),
        ("分散 2 只", p["n2"]["总收益率%"], p["n2"]["最大回撤%"], C4, 120),
        ("分散 3 只", p["n3"]["总收益率%"], p["n3"]["最大回撤%"], C4, 100),
        ("分散 4 只", p["n4"]["总收益率%"], p["n4"]["最大回撤%"], C4, 80),
        ("限价承接(回落才买)", m["lf00"]["总收益率%"], m["lf00"]["最大回撤%"], C3, 110),
    ]
    for name, tot, dd, c, size in items:
        y = 1 + tot / 100
        ax2.scatter(dd, max(y, 0.008), s=size, color=c, alpha=.85, zorder=3,
                    edgecolor="white", linewidth=1.2)
    ax2.set_yscale("log")
    ax2.axhline(100, color=C2, ls="--", lw=1.4)
    ax2.text(-50.4, 112, "百倍线", color=C2, fontsize=9.4)
    _off2 = {"现行（纯主板单吊）": (8, 7), "主板+创业板": (8, 12),
             "分散 2 只": (9, 4), "分散 3 只": (10, 17), "分散 4 只": (10, -4),
             "限价承接(回落才买)": (8, 9), "只创业板": (9, -2)}
    for name, tot, dd, c, size in items:
        y = max(1 + tot / 100, 1.0)
        v = f"{y:,.0f}×" if y >= 100 else f"{y:.1f}×"
        ax2.annotate(f"{name}\n{v} / {dd:.1f}%", (dd, y), textcoords="offset points",
                     xytext=_off2.get(name, (9, 5)), fontsize=8.6, color=c, va="center")
    ax2.set_xlim(-51, -6)
    ax2.set_ylim(14, 1.6e4)
    ax2.set_xlabel("最大回撤（%）", fontsize=10.5)
    ax2.set_ylabel("全史总倍数（对数轴）", fontsize=10.5)
    ax2.grid(ls=":", color="#ddd", zorder=0)
    ax2.set_title("扩池、分散、限价：三个结构方向都往左下方走", fontsize=11.5, loc="left")

    fig.text(0.01, -0.02,
             "左：全史 3,645 个交易日、839 笔，只改 execution.slippage，其余参数与信号完全不变。"
             "右：同口径全史，括号内为总倍数 / 最大回撤；越靠右上越好。",
             fontsize=8.4, color=C3)
    plt.tight_layout()
    fig.savefig(OUT / "第206轮_百倍的生存边界.png", dpi=170, bbox_inches="tight")
    print("saved", OUT / "第206轮_百倍的生存边界.png")


if __name__ == "__main__":
    main()
