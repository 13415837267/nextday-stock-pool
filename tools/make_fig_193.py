# -*- coding: utf-8 -*-
"""第 193 轮配图: 副策略"候选池第 1 名之后"没有 alpha —— 名次 / 分散 / 进场价三段分解。"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
plt.rcParams["font.family"] = ["Noto Serif CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
RED, BLUE, ORANGE, GREY, GREEN = "#c0392b", "#2c6fbb", "#e08a1e", "#8a8a8a", "#2e8b57"

rd = lambda f: json.load(open(os.path.join(ROOT, "output", f)))
short = rd("exp_rank_193_1500.json")
full = rd("exp_rank_193_3000.json")
ent = rd("exp_entry_193.json")

fig, axes = plt.subplots(1, 3, figsize=(19.2, 6.6),
                         gridspec_kw={"width_ratios": [1.15, 1.0, 1.15]})

# ---------------- (a) 名次与 10 日收益: 一条平线 ----------------
ax = axes[0]
rs = [r["rank"] for r in short["a"]["rows"]]
se = lambda r: abs(r["mean"] / r["nw_t"]) if r["nw_t"] else 0
ax.errorbar([x - 0.09 for x in rs], [r["mean"] for r in short["a"]["rows"]],
            yerr=[1.96 * se(r) for r in short["a"]["rows"]], fmt="o-", color=BLUE,
            lw=2, ms=6, capsize=3.5, label=f"近 1500 日（n={short['a']['n_days']}）")
ax.errorbar([x + 0.09 for x in rs], [r["mean"] for r in full["a"]["rows"]],
            yerr=[1.96 * se(r) for r in full["a"]["rows"]], fmt="s--", color=ORANGE,
            lw=1.8, ms=5.5, capsize=3.5, alpha=0.9, label=f"全史（n={full['a']['n_days']}）")
ax.axhline(0, color="#bbb", lw=0.9)
ax.set_xticks(rs)
ax.set_xticklabels([f"第 {x} 名" for x in rs])
ax.set_ylabel("次日开盘买 · 持 10 日 · 净收益 %（95% CI）", fontsize=10.5)
ax.set_title("(a) 名次之间没有差异 —— 排序键只在第 1 名处有决策意义", fontsize=12.5, pad=10)
ax.legend(fontsize=9.5, loc="upper left", framealpha=0.95)
ax.grid(axis="y", ls=":", color="#ddd")
ax.set_ylim(-0.72, 1.95)
ax.annotate("第 1 名 vs 第 2 名：全史 +0.001%（t=0.01）",
            xy=(3.5, -0.52), fontsize=9.5, color="#7a3b00", ha="center",
            bbox=dict(boxstyle="round,pad=0.35", fc="#fff6e8", ec="#e0c9a6"))

# ---------------- (b) 加深候选池 = 摊薄, 不是增强 ----------------
ax = axes[1]
b = short["b"]
labs = [f"{r['max_pos']} 并发\n日买 {r['buys']}" for r in b]
mult = [1 + r["total"] / 100 for r in b]
cal = [r["calmar"] for r in b]
cols = [BLUE] + [GREY] * (len(b) - 1)
ax.bar(range(len(b)), mult, 0.6, color=cols, edgecolor="white", linewidth=0.8)
ax.set_yscale("log")
ax.set_ylim(1.2, 1200)
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}×"))
ax.set_xticks(range(len(b)))
ax.set_xticklabels(labs, fontsize=8.8)
ax.set_ylabel("1500 日累计倍数（对数轴）", fontsize=10.5)
for i, m in enumerate(mult):
    ax.text(i, m * 1.12, f"{m:.1f}×", ha="center", fontsize=9.5,
            color=BLUE if i == 0 else "#444", fontweight="bold" if i == 0 else "normal")
ax2 = ax.twinx()
ax2.plot(range(len(b)), cal, "o-", color=RED, lw=2.2, ms=7, zorder=6)
ax2.set_ylabel("卡玛比率", fontsize=10.5, color=RED)
ax2.tick_params(axis="y", colors=RED)
ax2.set_ylim(0.75, 2.55)
for i, c in enumerate(cal):
    ax2.annotate(f"{c:.2f}", (i, c), textcoords="offset points", xytext=(0, 9),
                 ha="center", fontsize=9, color=RED)
ax.set_title("(b) 加深候选池＝把同一笔钱摊薄：卡玛不升、终值腰斩", fontsize=12.5, pad=10)

# ---------------- (c) 进场价才是收益所在 ----------------
ax = axes[2]
rows = {(r["mode"], r["gate"], r["rank"]): r for r in ent["rows"]}
keys = [("open", False, 1), ("open", True, 1), ("limit", False, 1), ("limit", True, 1)]
labels = ["开盘买\n闸门关", "开盘买\n闸门开", "限价 −1%\n闸门关", "限价 −1%\n闸门开"]
vals = [rows[k]["mean"] for k in keys]
ns = [rows[k]["n"] for k in keys]
cols = [GREY, "#7fa6cf", ORANGE, BLUE]
ax.bar(range(4), vals, 0.62, color=cols, edgecolor="white", linewidth=0.8)
for i, (v, n) in enumerate(zip(vals, ns)):
    ax.text(i, v + 0.045, f"{v:+.2f}%", ha="center", fontsize=10,
            fontweight="bold" if i == 3 else "normal", color=cols[i])
    ax.text(i, 0.06, f"n={n}", ha="center", fontsize=8.5, color="white", alpha=0.92)
ax.axhspan(0.80, 4.29, color=RED, alpha=0.07)
ax.axhline(2.546, color=RED, lw=1.8, ls="--")
ax.text(-0.42, 2.82, "引擎 106 笔实测 +2.55%\n95% CI 0.80~4.29", fontsize=8.8,
        color=RED, ha="left", va="top")
ax.set_xticks(range(4))
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylim(0, 3.15)
ax.set_ylabel("持 10 日净收益 %（第 1 名）", fontsize=10.5)
ax.set_title("(c) 限价回踩 +0.3~0.5pp、闸门 +0.4~0.5pp：收益在执行", fontsize=12.5, pad=10)
ax.grid(axis="y", ls=":", color="#ddd")

fig.suptitle('第 193 轮：副策略候选池「第 1 名之后」没有 alpha —— 所以页面只该发布 1 只',
             fontsize=15, y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.955), w_pad=4.5)
out = os.path.join(ROOT, "第193轮_名次无差异与进场分解.png")
fig.savefig(out, dpi=170, facecolor="white")
print("写出", out)
