# -*- coding: utf-8 -*-
"""第187轮两张图：右侧的杠杆前沿 + 十二个方向的天花板。"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"

plt.rcParams["font.family"] = ["Noto Serif CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load(name):
    p = OUT / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def fig1():
    lev = json.loads((OUT / "exp_right_lev_1500.json").read_text())["variants"]
    keys = ["pb_lev10", "pb_lev125", "pb_lev175", "pb_lev20", "pb_lev25"]
    labs = ["不融资 1.0x", "1.25x", "1.75x", "2.0x", "2.5x"]
    xs = [abs(lev[k]["回撤%"]) for k in keys]
    ys = [1 + lev[k]["总收益%"] / 100 for k in keys]

    fig, ax = plt.subplots(figsize=(9.6, 6.2))
    ax.axhline(100, color="#c0392b", lw=1.6, ls="--", zorder=1)
    ax.text(78, 108, "百倍线", color="#c0392b", fontsize=11, ha="right")

    ax.plot(xs, ys, "-o", color="#2c3e50", lw=1.8, ms=8, zorder=4,
            label="右侧趋势回踩（不同杠杆）")
    for x, y, l in zip(xs, ys, labs):
        ax.annotate(l, (x, y), textcoords="offset points", xytext=(0, 12),
                    ha="center", fontsize=9.5, color="#2c3e50")

    dip = (21.7, 498.29)
    ax.scatter([dip[0]], [dip[1]], s=190, color="#27ae60", zorder=5,
               edgecolor="white", linewidth=1.6, label="主策略 dip（现役 1.25x）")
    ax.annotate("主策略 dip\n498× / −21.7% / 卡玛 8.44", dip,
                textcoords="offset points", xytext=(10, -26), fontsize=9.5,
                color="#27ae60", ha="left")

    ax.set_yscale("log")
    ax.set_ylim(0.6, 900)
    ax.set_xlim(15, 85)
    ax.set_xlabel("最大回撤（绝对值 %）")
    ax.set_ylabel("1500 日期末倍数（对数轴）")
    ax.set_title("右侧任何杠杆都碰不到百倍线：2.5× 只给 23.8 倍，回撤却已 −77%\n"
                 "同一条回撤轴上，主策略用 1.25× 拿到 498 倍", fontsize=12, pad=12)
    ax.grid(alpha=0.25, ls="--")
    ax.legend(loc="lower right", fontsize=9.5, framealpha=0.92)
    fig.tight_layout()
    p = ROOT.parent / "第187轮_右侧杠杆前沿.png"
    fig.savefig(p, dpi=170)
    print(p)


def fig2():
    lev = json.loads((OUT / "exp_right_lev_1500.json").read_text())["variants"]
    pb = json.loads((OUT / "exp_right_pull_1500.json").read_text())["variants"]
    hist = json.loads((OUT / "exp_right_hist_3053.json").read_text())["variants"]
    boards = json.loads((OUT / "exp_right_boards_1500.json").read_text())["variants"]
    rev = json.loads((OUT / "exp_rev_grid_1500.json").read_text())["variants"]
    rank = json.loads((OUT / "exp_rank_hist_1500.json").read_text())["variants"]
    split = json.loads((OUT / "exp_dip_split_1500.json").read_text())["variants"]
    vcp = json.loads((OUT / "exp_vcp_1500.json").read_text())["variants"]

    def mult(v):
        return max(0.02, 1 + v["总收益%"] / 100)

    rows = [
        ("主策略 dip（现役）", mult(lev["base_dip"]), "#27ae60"),
        ("60/40 主+防御", 116.14, "#27ae60"),
        ("卫星低波 H10", 8.27, "#27ae60"),
        ("右侧回踩 · 1.0×（右侧最优卡玛）", mult(lev["pb_lev10"]), "#e67e22"),
        ("右侧回踩 · 2.0×（右侧最高倍数）", mult(lev["pb_lev20"]), "#e67e22"),
        ("右侧回踩 · 全史 3053 日", mult(hist["pb_def"]), "#e67e22"),
        ("右侧 全板块", mult(boards["all_board"]), "#e67e22"),
        ("VCP 波动收缩贴新高", mult(vcp["v1_def"]), "#e67e22"),
        ("放量深跌反转（最好档）", mult(rev["r5_shallow"]), "#8e44ad"),
        ("dip 第 3 名", mult(rank["r3"]), "#8e44ad"),
        ("dip 深回调 −9~−6%", mult(split["deep"]), "#8e44ad"),
        ("dip 强趋势 mom20 25~60", mult(split["hot"]), "#8e44ad"),
    ]
    rows.sort(key=lambda r: r[1], reverse=True)

    fig, ax = plt.subplots(figsize=(11.2, 6.4))
    ys = list(range(len(rows)))
    ax.barh(ys, [r[1] for r in rows], color=[r[2] for r in rows], alpha=0.88, height=0.62)
    ax.axvline(100, color="#c0392b", lw=1.6, ls="--")
    ax.text(112, len(rows) - 0.35, "百倍", color="#c0392b", fontsize=10.5, va="top")
    ax.axvline(1.0, color="#7f8c8d", lw=1.0, ls=":")
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9.6)
    ax.set_xscale("log")
    ax.set_xlim(0.02, 3000)
    ax.invert_yaxis()
    for y, r in zip(ys, rows):
        ax.text(r[1] * 1.12, y, f"{r[1]:,.1f}×", va="center", fontsize=8.8)
    ax.set_xlabel("1500 日期末倍数（对数轴；右侧全史为 3053 日）")
    ax.set_title("同一个引擎、同一套成本、同一个窗口：十二个方向，只有 dip 越过百倍线\n"
                 "右侧最好的形态（趋势回踩 1.0×）是 10.1 倍，全史 1.9 倍；VCP 与放量深跌为负",
                 fontsize=11.8, pad=12)
    ax.grid(axis="x", alpha=0.25, ls="--")
    fig.tight_layout()
    p = ROOT.parent / "第187轮_十二个方向天花板.png"
    fig.savefig(p, dpi=170)
    print(p)


if __name__ == "__main__":
    fig1()
    fig2()
