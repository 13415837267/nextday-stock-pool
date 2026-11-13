#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第 215 轮配图: 放宽约束(持有期/分散)对主策略的双窗口判决"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

_font_candidates = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
]
for fp in _font_candidates:
    try:
        font_manager.fontManager.addfont(fp)
    except Exception:
        pass
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

RED = "#d62728"
GREEN = "#2ca02c"
BLUE = "#1f77b4"
GRAY = "#7f7f7f"


def load(days):
    p = Path(f"output/relax_215_{days}.json")
    d = json.loads(p.read_text())
    return {r.get("变体"): r for r in d}


def main():
    full = load(3664)
    recent = load(1500)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.6), dpi=130)

    # ===== 左栏: 持有期延长 -> 年化崩塌 =====
    hold_names = ["持有1日", "持有2日", "持有3日", "持有4日", "持有5日",
                  "持有9日", "持有14日", "持有19日"]
    full_cagr = [full.get(hn + "·单吊", {}).get("年化%") for hn in hold_names]
    recent_cagr = [recent.get(hn + "·单吊", {}).get("年化%") for hn in hold_names]
    x = list(range(len(hold_names)))
    w = 0.36
    b1 = ax1.bar([i - w / 2 for i in x], full_cagr, w, label="全史 3664 日", color=BLUE, alpha=0.85)
    b2 = ax1.bar([i + w / 2 for i in x], recent_cagr, w, label="近 6 年 1500 日", color=RED, alpha=0.85)
    ax1.axhline(0, color="black", lw=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(hold_names, fontsize=10)
    ax1.set_ylabel("年化收益率 %", fontsize=11)
    ax1.set_title("持有期越长 → 年化越崩（单吊满仓）", fontsize=13, fontweight="bold", color=RED)
    for bars in (b1, b2):
        for b in bars:
            v = b.get_height()
            if v is None:
                continue
            off = 1.5 if v >= 0 else -4.5
            ax1.text(b.get_x() + b.get_width() / 2, v + off, f"{v:.0f}",
                     ha="center", va="bottom" if v >= 0 else "top",
                     fontsize=8, color="#1a1a1a", fontweight="bold")
    ax1.legend(fontsize=9, framealpha=0.9, loc="upper right")
    ax1.set_ylim(-35, 125)

    # ===== 右栏: 分散 -> 回撤降但收益腰斩 =====
    div_names = ["单吊×100%", "2只×50%", "3只×33%", "5只×20%"]
    full_c, full_dd, recent_c, recent_dd = [], [], [], []
    for dn in div_names:
        key = "持有1日·单吊" if dn == "单吊×100%" else "T+2·分散" + dn
        full_c.append(full.get(key, {}).get("年化%"))
        full_dd.append(full.get(key, {}).get("回撤%"))
        recent_c.append(recent.get(key, {}).get("年化%"))
        recent_dd.append(recent.get(key, {}).get("回撤%"))

    x2 = list(range(len(div_names)))
    w2 = 0.36
    b3 = ax2.bar([i - w2 / 2 for i in x2], full_c, w2, label="全史·年化", color=BLUE, alpha=0.85)
    b4 = ax2.bar([i + w2 / 2 for i in x2], recent_c, w2, label="近6年·年化", color=RED, alpha=0.85)
    for bars in (b3, b4):
        for b in bars:
            v = b.get_height()
            if v is None:
                continue
            ax2.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}",
                     ha="center", va="bottom", fontsize=9, color="#1a1a1a", fontweight="bold")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(div_names, fontsize=10)
    ax2.set_ylabel("年化收益率 %", fontsize=11, color=BLUE)
    ax2.set_title("分散只降回撤、收益同步腰斩（T+2 原口径）", fontsize=13, fontweight="bold", color=BLUE)
    ax2.set_ylim(0, 125)

    ax2b = ax2.twinx()
    ax2b.plot(x2, full_dd, "o--", color=GREEN, lw=1.6, label="全史·回撤")
    ax2b.plot(x2, recent_dd, "s--", color=GRAY, lw=1.6, label="近6年·回撤")
    ax2b.set_ylabel("最大回撤 %", fontsize=11, color=GREEN)
    ax2b.set_ylim(-80, 0)
    for i, v in enumerate(full_dd):
        ax2b.text(i - 0.2, v - 4.5, f"{v:.1f}", fontsize=8, color=GREEN, ha="center", va="top")
    for i, v in enumerate(recent_dd):
        ax2b.text(i + 0.2, v - 4.5, f"{v:.1f}", fontsize=8, color=GRAY, ha="center", va="top")
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2b.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right", framealpha=0.9)

    fig.suptitle("第 215 轮：放宽硬约束（不加杠杆）—— 持有期延长 / 分散 均为负优化，满仓单吊 + T+2 已最优",
                 fontsize=14, fontweight="bold", color="#1a1a1a", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = "/workspace/第215轮_放宽约束不加杠杆的判决.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print("已生成", out)


if __name__ == "__main__":
    main()
