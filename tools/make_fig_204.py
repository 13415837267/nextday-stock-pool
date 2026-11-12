"""第 204 轮配图: 12 个"训练出来"的变体, 全部被现行参数包围。"""
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
NAME = {
    "base": "现行配置", "bf100": "风控日满仓", "S08": "评分门槛 0.8",
    "L4": "收盘位置 0.40", "gap0": "风控日不设缺口", "S12": "评分门槛 1.2",
    "bf100_gap0": "满仓+不设缺口", "L3": "跌幅放宽 -7~-2.5",
    "L1": "跌幅+收盘位置放宽", "S15": "评分门槛 1.5",
    "W_mix": "训练权重(半融合)", "W_train": "训练权重(全替换)",
}


def main() -> None:
    d = json.loads((OUT / "exp_cfg_204_3000.json").read_text(encoding="utf-8"))
    rows = sorted(d.items(), key=lambda x: x[1]["总收益率%"])

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.4, 6.2),
                                  gridspec_kw={"width_ratios": [1.25, 1]})

    # 左: 总倍数(对数横条) —— 一眼看出量级差
    names = [NAME.get(k, k) for k, _ in rows]
    vals = [1 + v["总收益率%"] / 100 for _, v in rows]   # 总收益率% -> 倍数
    cols = [C2 if k == "base" else (C4 if k.startswith("W_") else C1) for k, _ in rows]
    y = np.arange(len(rows))
    ax.barh(y, vals, color=cols, height=0.68)
    ax.set_xscale("log")
    ax.set_yticks(y, names, fontsize=9.5)
    for i, v in enumerate(vals):
        ax.text(v * 1.12, i, f"{v:,.0f}×" if v >= 100 else f"{v:.1f}×",
                va="center", fontsize=8.6,
                color=C2 if rows[i][0] == "base" else "#333")
    ax.set_xlim(1.0, 1.1e4)
    ax.set_xlabel("全史 3000 日总倍数（对数刻度）", fontsize=10)
    ax.set_title("十二种\"全史训练出来\"的选股条件，没有一种同时赢过收益与回撤",
                 fontsize=11.5, loc="left")
    ax.grid(axis="x", ls=":", color="#ccc", alpha=.7)
    ax.set_axisbelow(True)

    # 右: 年化 vs 回撤, 现应在左上
    for k, v in d.items():
        x = abs(v["最大回撤%"])
        yv = v["年化收益率%"]
        c = C2 if k == "base" else (C4 if k.startswith("W_") else C1)
        ax2.scatter(x, yv, s=28 + v["交易次数"] * 0.28, color=c,
                    alpha=.85 if k == "base" else .55, edgecolor="white", lw=.8, zorder=3)
    for k in ("base", "bf100", "W_train", "L1", "S12"):
        v = d[k]
        dx, dy = (2.0, 1.2)
        if k == "W_train":
            dx, dy = (2.0, 2.0)         # 标注放点的右上
        if k == "S12":
            dx, dy = (1.6, 1.4)
        if k == "L1":
            dx, dy = (-21.0, 1.0)       # 避免超出右边界
        ax2.annotate(NAME[k], (abs(v["最大回撤%"]), v["年化收益率%"]),
                     textcoords="offset points", xytext=(dx * 4, dy * 3),
                     fontsize=9, color=C2 if k == "base" else "#333")
    ax2.set_xlabel("最大回撤（绝对值 %）", fontsize=10)
    ax2.set_ylabel("年化收益 %", fontsize=10)
    ax2.set_title("现行配置在左上角：收益第二、回撤最好", fontsize=12, loc="left")
    ax2.set_xlim(19, 49)
    ax2.grid(ls=":", color="#ddd")
    ax2.set_axisbelow(True)
    ax2.text(0.98, 0.96, "点大小 = 交易笔数", transform=ax2.transAxes,
             ha="right", va="top", fontsize=8.6, color=C3)

    fig.tight_layout()
    p = OUT / "第204轮_十二个训练变体.png"
    fig.savefig(p, dpi=170, facecolor="white")
    print("saved", p)


if __name__ == "__main__":
    main()
