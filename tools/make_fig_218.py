#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第 218 轮配图: 涨停情绪维度 —— 近 6 年很强, 17 年归零"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

for fp in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
           "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"]:
    try:
        font_manager.fontManager.addfont(fp)
    except Exception:
        pass
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

RED = "#c62828"
GREEN = "#2e7d32"
BLUE = "#1565c0"
GRAY = "#9e9e9e"


def main():
    full = json.loads(Path("output/exp_mood_full_217.json").read_text())
    part = json.loads(Path("output/exp_mood_217.json").read_text())
    pos = json.loads(Path("output/exp_mood_pos_217.json").read_text())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.9),
                                   gridspec_kw={"width_ratios": [1.55, 1]})

    # ---------- 左: 年度 高-低 差 ----------
    rows = [r for r in full["B_年度"] if r["高-低"] is not None]
    years = [r["年"] for r in rows]
    diffs = [r["高-低"] for r in rows]
    colors = [GREEN if d > 0 else RED for d in diffs]
    x = np.arange(len(years))
    ax1.bar(x, diffs, color=colors, width=0.66, zorder=3)
    ax1.axhline(0, color="#37474f", lw=1.2, zorder=4)

    # 2020 分界
    i20 = years.index("2020") if "2020" in years else None
    if i20 is not None:
        ax1.axvline(i20 - 0.5, color="#455a64", ls="--", lw=1.6, zorder=5)
        ax1.text(i20 - 0.5, min(diffs) - 1.15, "2020", ha="center", va="bottom",
                 fontsize=9.0, color="#455a64")

    for xi, d in zip(x, diffs):
        lab = "0.0" if abs(d) < 0.05 else f"{d:+.1f}"
        ax1.text(xi, d + (0.16 if d >= 0 else -0.16), lab, ha="center",
                 va="bottom" if d >= 0 else "top", fontsize=8.6, color="#37474f")
    ax1.set_xticks(x)
    ax1.set_xticklabels(years, rotation=45, fontsize=9.2)
    ax1.set_ylim(min(diffs) - 1.4, max(diffs) + 1.6)
    ax1.set_ylabel("情绪高分档 − 低分档 的单笔收益差（pp）", fontsize=10.5)
    ax1.set_title("A. 逐年看：8 年为正、8 年为负，2020 之前基本反向", fontsize=12.5,
                  fontweight="bold", loc="left", pad=10)
    ax1.grid(axis="y", color="#e8e8e8", lw=0.8, zorder=0)
    ax1.set_axisbelow(True)
    for sp in ("top", "right"):
        ax1.spines[sp].set_visible(False)

    # ---------- 右: 全史 vs 近 6 年 的秩相关 ----------
    labels = ["情绪合成", "最高连板", "炸板率", "涨停家数", "一字板家数"]
    keys_full = ["情绪合成(全史)", "最高连板", "炸板率", "涨停家数", "一字板家数"]
    keys_part = ["情绪质量分(滚动z)", "最高连板", "烂板率", "涨停家数", "一字占比"]

    rho_full = [full["A_秩相关"][k]["rho"] for k in keys_full]
    rho_part = [part["A_秩相关"].get(k, {}).get("rho", np.nan) for k in keys_part]

    yy = np.arange(len(labels))[::-1]
    h = 0.34
    ax2.barh(yy + h/2, rho_part, height=h, color=BLUE, alpha=0.85,
             label="近 6 年（2020+，411 笔）", zorder=3)
    ax2.barh(yy - h/2, rho_full, height=h, color=GRAY, alpha=0.9,
             label="全史（2012+，843 笔）", zorder=3)
    ax2.axvline(0, color="#37474f", lw=1.1, zorder=4)

    for y, v in zip(yy + h/2, rho_part):
        ax2.text(v + (0.004 if v >= 0 else -0.004), y, f"{v:+.3f}", va="center",
                 ha="left" if v >= 0 else "right", fontsize=8.8, color=BLUE)
    for y, v in zip(yy - h/2, rho_full):
        ax2.text(v + (0.004 if v >= 0 else -0.004), y, f"{v:+.3f}", va="center",
                 ha="left" if v >= 0 else "right", fontsize=8.8, color="#555555")

    ax2.set_yticks(yy)
    ax2.set_yticklabels(labels, fontsize=10.2)
    ax2.set_xlim(-0.082, 0.235)
    ax2.set_xlabel("与单笔收益的 Spearman 秩相关", fontsize=10.5)
    ax2.set_title("B. 同样方法扩到 17 年，相关系数全部归零", fontsize=12.5,
                  fontweight="bold", loc="left", pad=10)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), fontsize=9.2,
               frameon=False, ncol=2)
    ax2.grid(axis="x", color="#e8e8e8", lw=0.8, zorder=0)
    ax2.set_axisbelow(True)
    for sp in ("top", "right"):
        ax2.spines[sp].set_visible(False)

    fig.text(0.012, 0.015,
             "情绪合成 = 最高连板↑ − 炸板率↑ + 涨停家数↑（滚动250日标准化）。"
             "左图用全史 843 笔；右图蓝=涨停明细口径(411 笔)，灰=面板自算口径(843 笔)。",
             fontsize=8.4, color="#666666")
    fig.tight_layout(rect=[0, 0.075, 1, 1])
    out = Path("第218轮_涨停情绪维度的全史证伪.png")
    fig.savefig(out, dpi=170)
    print("saved", out)


if __name__ == "__main__":
    main()
