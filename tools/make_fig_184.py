# -*- coding: utf-8 -*-
"""第184轮 图: ①全市场 vs 低波池内 十分位收益 ②四组实验对 base 的相对表现"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"

plt.rcParams["font.family"] = ["Noto Serif CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

CN = {"ivol20": "IVOL20", "max20": "MAX20", "imax20": "IMAX20", "on_abs10": "隔夜跳空",
      "ctc20": "CTC20", "to_ma20": "换手率", "mom20": "动量20", "mom20_ivol": "现役键(动量/波动)"}


def fig1():
    d = json.loads((OUT / "exp_web_decile_1500.json").read_text())
    blocks = d["blocks"]
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.9), sharey=True)
    for ax, tag, title in zip(axes, ["全市场", "低波池内"],
                              ["全市场：IC 显著的因子, 十分位单调",
                               "低波池内(IVOL 前 10%)：全部拉平甚至反向"]):
        cur = blocks[tag]
        xs = np.arange(1, 11)
        for fac, val in cur.items():
            ys = [val["decile"].get(str(i), val["decile"].get(i)) for i in xs]
            hot = fac in ("mom20_ivol", "ivol20")
            ax.plot(xs, ys, marker="o", ms=4.2, lw=2.6 if hot else 1.3,
                    alpha=1.0 if hot else 0.55,
                    color="#c0392b" if fac == "mom20_ivol" else
                          ("#2c3e50" if fac == "ivol20" else None),
                    label=CN.get(fac, fac), zorder=3 if hot else 2)
        ax.axhline(0, color="#999", lw=0.8, ls=":")
        ax.set_title(title, fontsize=11.5, pad=8)
        ax.set_xlabel("当日横截面十分位（1=因子最小, 10=最大）", fontsize=9.5)
        ax.set_xticks(xs)
        ax.grid(alpha=0.22, ls="--")
        if tag == "全市场":
            ax.set_ylabel("未来 10 日平均收益（%，未扣成本）", fontsize=9.5)
        # 标注 D1→D10 落差
        ax.annotate("", xy=(10, 0.05), xytext=(1, 0.05), arrowprops=dict(arrowstyle="-", lw=0))
    axes[1].legend(fontsize=8.2, ncol=2, loc="lower right", framealpha=0.9)
    fig.suptitle("同一批因子，在池外单调、在池内失效 —— 换排序键为什么救不了副策略",
                 fontsize=13, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p = ROOT.parent / "第184轮_池外单调池内失效.png"
    fig.savefig(p, dpi=170)
    print(p)


def _bar(ax, names, vals, base, title, fmt="{:.0f}"):
    """统一画"相对 base"的横条: 1.0 = 现役水平, 越左越差。"""
    ratio = [v / base if base else 0 for v in vals]
    colors = ["#c0392b" if r <= 1.0 else "#27ae60" for r in ratio]
    ax.barh(range(len(vals)), ratio, color=colors, alpha=0.88, height=0.62)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(names, fontsize=8.4)
    ax.axvline(1.0, color="#2c3e50", lw=1.4, ls="--")
    ax.set_xlim(min(0.0, min(ratio) * 1.15), max(1.22, max(ratio) * 1.12))
    ax.set_title(title, fontsize=10.5, pad=6)
    ax.grid(axis="x", alpha=0.25, ls="--")
    ax.set_xlabel("相对现役（1.0=现役）", fontsize=8.5)
    for i, (r, v) in enumerate(zip(ratio, vals)):
        ax.text(r, i, " " + fmt.format(v), va="center", fontsize=7.4)
    ax.invert_yaxis()


def fig2():
    sort = json.loads((OUT / "exp_web_sort_1500.json").read_text())["windows"]["1500"]["variants"]
    excl = json.loads((OUT / "exp_web_excl_1500.json").read_text())["windows"]["1500"]["variants"]
    main = json.loads((OUT / "exp_web_main_1500.json").read_text())["variants"]
    marg = json.loads((OUT / "exp_web_margin_1500.json").read_text())["variants"]

    fig, axes = plt.subplots(2, 2, figsize=(13.4, 9.0))
    b = sort["base"]["年化%"]
    ks = [k for k in sort if k != "base"]
    _bar(axes[0][0], ks, [sort[k]["年化%"] for k in ks], b,
         "副策略：换排序键（年化%）", "{:.0f}%")
    b2 = excl["base"]["年化%"]
    k2 = [k for k in excl if k != "base"]
    _bar(axes[0][1], k2, [excl[k]["年化%"] for k in k2], b2,
         "副策略：排除尾部（年化%）", "{:.0f}%")
    b3 = main["base"]["总收益率%"]
    k3 = [k for k in main if k != "base"]
    _bar(axes[1][0], k3, [main[k]["总收益率%"] for k in k3], b3,
         "主策略：加门槛（总收益，15 年复利）", "{:,.0f}%")
    b4 = marg["base"]["年化%"]
    k4 = [k for k in marg if k != "base"]
    _bar(axes[1][1], k4, [marg[k]["年化%"] or 0 for k in k4], b4,
         "副策略：两融闸门（年化%）", "{:.0f}%")
    fig.suptitle("第184轮：五个文献因子 × 七种用法，全部劣于现役（新增 35 个变体, 0 存活）",
                 fontsize=13, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    p = ROOT.parent / "第184轮_文献因子全否.png"
    fig.savefig(p, dpi=160)
    print(p)


if __name__ == "__main__":
    fig1()
    fig2()
