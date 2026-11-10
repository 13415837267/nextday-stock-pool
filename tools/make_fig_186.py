# -*- coding: utf-8 -*-
"""第186轮作图。

图1 百倍可行性散点: 横轴 = 交易笔数, 纵轴 = 平均单笔收益, 曲线 = "百倍线" r(N)=100^(1/N)-1
     —— 任何一个策略，点落在曲线上方才可能百倍。右侧 7 个形态全部在曲线下方，
        且与曲线不是"差一点"，是差一个数量级。
图2 缺口归因: 四个模式（dip / trend / pullback / limitup）按 T+1 开盘缺口分 8 箱的单笔收益
     —— 用同一引擎、同一成本，横截面比较，回答"右侧到底缺什么"。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
plt.rcParams["font.family"] = ["Noto Serif CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = ROOT / "output"
WORK = ROOT.parent


def load(name):
    return json.loads((OUT / f"{name}.json").read_text(encoding="utf-8"))


def fig1():
    hunt = load("exp_right_hunt_a_1500")["variants"]
    pull = load("exp_right_pull_1500")["variants"]
    sat = load("exp_sat_bp_1500")["variants"]
    gem = load("exp_gem_tune_1500")["variants"]

    pts = []   # (笔数, 单笔%, 标签, 组, 标注偏移)
    pts.append((hunt["base_dip"]["笔数"], hunt["base_dip"]["单笔%"], "主板 dip · 现役", "ok", (0, 12)))
    for k, lab, off in (("tr_def", "追突破", (0, -18)),
                        ("tr_g05", "追突破·只买平开", (-84, 8)),
                        ("pb_def", "趋势回踩", (0, 12)),
                        ("pb_combo", "趋势回踩·严选", (0, 12))):
        pts.append((hunt[k]["笔数"] if k in hunt else pull[k]["笔数"],
                    hunt[k]["单笔%"] if k in hunt else pull[k]["单笔%"], lab, "no", off))
    for k, lab, off in (("bp_h5", "趋势回调 H5", (30, -20)), ("bo_h5", "突破 H5", (30, 12))):
        pts.append((sat[k]["笔数"], sat[k]["单笔%"], lab, "no", off))
    pts.append((gem["g_p10"]["笔数"], gem["g_p10"]["单笔%"], "创业板 dip · 宽区间", "ok", (34, 12)))
    pts.append((gem["g_best"]["笔数"], gem["g_best"]["单笔%"], "创业板 dip · 第二腿", "ok", (-6, -26)))

    fig, ax = plt.subplots(figsize=(11.2, 6.6))
    ns = np.linspace(40, 800, 400)
    ax.plot(ns, (100 ** (1 / ns) - 1) * 100, color="#2c3e50", lw=2.0, ls="--",
            label="百倍线  r = 100^(1/N) − 1")
    ax.fill_between(ns, (100 ** (1 / ns) - 1) * 100, 3.2, color="#27ae60", alpha=0.07)
    ax.axhline(0, color="#999", lw=0.8)

    for n, r, lab, grp, off in pts:
        col = "#27ae60" if grp == "ok" else "#c0392b"
        ax.scatter([n], [r], s=140 if grp == "ok" else 70, color=col,
                   zorder=5, edgecolor="white", linewidth=1.4)
        ax.annotate(lab, (n, r), textcoords="offset points", xytext=off,
                    ha="center", fontsize=8.4, color=col)

    ax.set_xlabel("交易笔数（1500 交易日窗口）", fontsize=11)
    ax.set_ylabel("平均单笔收益 %", fontsize=11)
    ax.set_title("百倍只需要两个数字：单笔 × 笔数\n"
                 "绿色在线上（可能百倍），红色在线下（无论怎么调参会差一个数量级）",
                 fontsize=12.5, pad=12)
    ax.set_xlim(40, 800)
    ax.set_ylim(-1.6, 2.6)
    ax.grid(alpha=0.25, ls="--")
    ax.legend(loc="upper right", fontsize=9.5)
    fig.tight_layout()
    p = WORK / "第186轮_百倍线与右侧位置.png"
    fig.savefig(p, dpi=160)
    print(p)


def fig2():
    d = load("exp_gap_attrib_1500")["modes"]
    order = ["dip", "trend", "pullback", "limitup"]
    cn = {"dip": "主板 dip（缩量阴线 · 现役）", "trend": "追突破（右侧）",
          "pullback": "趋势回踩（右侧）", "limitup": "涨停次日（右侧）"}
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.0))
    for i, m in enumerate(order):
        ax = axes[i // 2][i % 2]
        v = d[m]
        bins = [b for b in v["bins"] if b["n"]]
        xs = np.arange(len(bins))
        vals = [b["avg%"] for b in bins]
        cols = ["#c0392b" if x < 0 else "#27ae60" for x in vals]
        ax.bar(xs, vals, color=cols, alpha=0.85, width=0.72)
        ax.axhline(0, color="#333", lw=0.9)
        ax.set_xticks(xs)
        ax.set_xticklabels([b["bin"] for b in bins], fontsize=7.2, rotation=0)
        ax.set_title(f"{cn[m]}\n低开 {v['low_open']['n']} 笔 {v['low_open']['avg%']:+.2f}%  ·  "
                     f"平/高开 {v['high_open']['n']} 笔 {v['high_open']['avg%']:+.2f}%",
                     fontsize=9.6, pad=7)
        ax.set_ylabel("单笔收益 %", fontsize=8.6)
        ax.set_xlabel("T+1 开盘缺口 %", fontsize=8.6)
        ax.grid(axis="y", alpha=0.22, ls="--")
        for x, y, b in zip(xs, vals, bins):
            ax.text(x, y, f"{y:+.1f}", ha="center", va="bottom" if y >= 0 else "top",
                    fontsize=6.6)
    fig.suptitle("同一引擎、同一成本、同一窗口：左侧 dip 每一格都在赚钱（深低开处 +4.0%），右侧三个形态没有一格能复制这个量级",
                 fontsize=12.5, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    p = WORK / "第186轮_缺口归因.png"
    fig.savefig(p, dpi=160)
    print(p)


if __name__ == "__main__":
    fig1()
    fig2()
