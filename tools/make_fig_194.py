# -*- coding: utf-8 -*-
"""第 194 轮配图: 主策略"买价"维度的天花板。

(a) 三条限价方案 vs 无条件开盘买: 终值全部更低
(b) 三形态 U 型: 限价单只在最差那一档成交(逆向选择)
(c) 同一个执行规则, 主策略与副策略符号相反
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RED, BLUE, GRAY, ORANGE, GREEN = "#c0392b", "#2c6fb5", "#7f8c8d", "#d68910", "#1e8449"
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Sans CJK SC", "WenQuanYi Zen Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

d = json.loads((ROOT / "output" / "exp_bid_194_1500.json").read_text(encoding="utf-8"))
rows, r3, rl = d["rows"], d["rows_wait3"], d["rows_loc"]
base = rows[0]
dump = d["dump"]

fig, axes = plt.subplots(1, 3, figsize=(16.2, 4.9))
fig.suptitle("第 194 轮：主策略的买价没有可榨的油水 —— 限价单是逆向选择",
             fontsize=15.5, y=0.985)

# ---------- (a) 三条方案 vs base ----------
ax = axes[0]
pb1 = [r["pb"] for r in rows if r["pb"] > 0]
ax.plot(pb1, [r["nav"] / base["nav"] for r in rows if r["pb"] > 0], "o-", color=RED, lw=2, ms=5, label="只挂 T+1（未触及空仓）")
ax.plot([r["pb"] for r in r3 if r["pb"] > 0], [r["nav"] / base["nav"] for r in r3 if r["pb"] > 0],
        "s--", color=BLUE, lw=2, ms=5, label="挂 3 个交易日（未触及空仓）")
ax.plot([r["pb"] for r in rl], [r["nav"] / base["nav"] for r in rl], "^-.", color=ORANGE, lw=2, ms=5,
        label="限价优先 + 未成交按收盘买")
ax.axhline(1.0, color=GRAY, ls=":", lw=2)
ax.text(2.82, 1.06, "现行 = 1.00×", color="#333", fontsize=10, ha="right", va="bottom")
ax.set_yscale("log")
ax.set_xlabel("限价幅度：信号日收盘 × (1 − x%)", fontsize=10.5)
ax.set_ylabel("终值 ÷ 现行（逐笔复利 · 相对口径 · 对数轴）", fontsize=10.5)
ax.set_title("(a) 三种挂单方案，全部低于现行", fontsize=12.5, pad=9)
ax.grid(alpha=0.25, which="both")
ax.legend(fontsize=8.6, loc="lower left", framealpha=0.97)
ax.set_ylim(0.005, 1.35)
ax.set_yticks([0.01, 0.03, 0.1, 0.3, 1.0])
ax.set_yticklabels(["0.01×", "0.03×", "0.1×", "0.3×", "1×"])

# ---------- (b) 三形态 ----------
ax = axes[1]
grp = ["开盘即破 −1%\n（开盘价买）", "盘中才破 −1%\n（限价成交）", "全天不破 −1%\n（限价不成交）"]
vals = []
for how in ("open", "limit", "miss"):
    sub = [x for x in dump if x["how"] == how]
    vals.append((len(sub), float(np.mean([x["ret_old"] for x in sub]))))
bars = ax.bar(range(3), [v[1] for v in vals], color=[GREEN, RED, GREEN], width=0.58, alpha=0.9)
for i, (n, m) in enumerate(vals):
    ax.text(i, m + 0.1, f"{m:+.2f}%", ha="center", va="bottom", fontsize=12, color="#222")
    ax.text(i, -0.28, f"n={n}", ha="center", va="top", fontsize=9.5, color=GRAY)
ax.axhline(0, color="#444", lw=0.9)
ax.set_xticks(range(3))
ax.set_xticklabels(grp, fontsize=9.5)
ax.set_ylim(-0.55, 3.15)
ax.set_ylabel("单笔净收益（%，持到 T+2 开盘卖）", fontsize=10.5)
ax.set_title("(b) U 型：限价单只在最差那一档成交", fontsize=12.5, pad=9)
ax.grid(axis="y", alpha=0.25)
ax.annotate("逆向选择：\n想买的票不成交，\n成交的都是阴跌票",
            xy=(0.96, 0.12), xytext=(0.42, 1.45), fontsize=9.2, color=RED,
            ha="left", va="center",
            arrowprops=dict(arrowstyle="->", color=RED, lw=1.2, connectionstyle="arc3,rad=0.28"))

# ---------- (c) 两策略符号相反 ----------
ax = axes[2]
labels = ["主策略 dip\n（买跌）", "副策略 lowvol\n（追强）"]
open_v = [1.439, 0.783]
lim_v = [1.449, 1.182]
x = np.arange(2)
w = 0.33
ax.bar(x - w / 2, open_v, w, color=GRAY, label="T+1 开盘买")
ax.bar(x + w / 2, lim_v, w, color=RED, label="限价 −1% 回踩买")
for i in range(2):
    ax.text(x[i] - w / 2, open_v[i] + 0.03, f"{open_v[i]:+.2f}%", ha="center", fontsize=10)
    ax.text(x[i] + w / 2, lim_v[i] + 0.03, f"{lim_v[i]:+.2f}%", ha="center", fontsize=10, color=RED)
    dv = lim_v[i] - open_v[i]
    ax.text(x[i], max(open_v[i], lim_v[i]) + 0.30, f"{dv:+.2f}pp", ha="center",
            fontsize=11.5, color=(GREEN if dv > 0.15 else RED), fontweight="bold")
ax.text(0, -0.38, "砍掉 31% 的笔数", ha="center", fontsize=9, color=RED)
ax.text(1, -0.38, "成交 73%，收益升", ha="center", fontsize=9, color=GREEN)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylim(-0.78, 2.05)
ax.set_ylabel("单笔净收益（%）", fontsize=10.5)
ax.set_title("(c) 同一条执行规则，两个策略符号相反", fontsize=12.5, pad=9)
ax.grid(axis="y", alpha=0.25)
ax.legend(fontsize=9, loc="upper right", framealpha=0.95)

fig.tight_layout(rect=(0, 0, 1, 0.945), w_pad=3.6)
out = ROOT / "第194轮_主策略买价已到顶.png"
fig.savefig(out, dpi=155)
print("saved", out)
