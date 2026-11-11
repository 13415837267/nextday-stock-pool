# -*- coding: utf-8 -*-
"""第 195 轮配图: 关闭杠杆 + 参数进化 —— 进化搜不到东西, 现金管理才是真增量。

(a) 1.25x → 1.0x 的分年收益: 每年少赚, 但负年亏得更少
(b) GA 六代进化曲线: 1500 日 fit 只从 7.44 爬到 7.65, 而 3000 日验证反而掉到 4.43 (<4.64)
(c) 无杠杆下 9 个执行侧变体: 只有"闲置现金计息"是正向增量
"""
from __future__ import annotations

import json
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RED, BLUE, GRAY, ORANGE, GREEN = "#c0392b", "#2c6fb5", "#7f8c8d", "#d68910", "#1e8449"
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Sans CJK SC", "WenQuanYi Zen Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def year_ret(eq: list[dict]) -> dict:
    ds = [e["d"] for e in eq]
    nav = np.array([e["nav"] for e in eq], dtype="float64")
    out = {}
    for y in sorted({d[:4] for d in ds}):
        m = np.array([d[:4] == y for d in ds])
        s = nav[m]
        if len(s) >= 20:
            out[y] = (s[-1] / s[0] - 1.0) * 100.0
    return out


base = json.loads((ROOT / "output" / "base_195_3000.json").read_text(encoding="utf-8"))
lev125 = json.loads((ROOT / "output" / "backtest_long.json").read_text(encoding="utf-8"))
ga = json.loads((ROOT / "output" / "exp_ga_195_1500.json").read_text(encoding="utf-8"))
ver = json.loads((ROOT / "output" / "exp_verify_195_3000.json").read_text(encoding="utf-8"))
nl = json.loads((ROOT / "output" / "exp_nolev_195_1500.json").read_text(encoding="utf-8"))

fig, axes = plt.subplots(1, 3, figsize=(16.6, 5.0))
fig.suptitle("第 195 轮：关掉杠杆之后 —— 参数进化搜不到东西，真增量在现金管理",
             fontsize=15.5, y=0.985)

# ---- (a) 分年差额: 1.0x − 1.25x ----
ax = axes[0]
ya, yb = year_ret(lev125["equity"]), year_ret(base["equity"])
years = [y for y in yb if y in ya]
diff = [yb[y] - ya[y] for y in years]
cols = [GREEN if v > 0 else "#c9a0a0" for v in diff]
ax.barh(np.arange(len(years)), diff, color=cols, alpha=0.92, height=0.62)
ax.axvline(0, color="#333", lw=1.0)
ax.set_yticks(np.arange(len(years)))
ax.set_yticklabels([y[2:] for y in years], fontsize=8.6)
ax.set_xlabel("年度收益差：1.0x − 1.25x（pp）", fontsize=10)
ax.set_title("(a) 关杠杆每年少赚；唯一例外 2017（+5.3pp，那一年是亏的）", fontsize=11.6, pad=10)
for i, v in enumerate(diff):
    ax.text(v + (3 if v > 0 else -3), i, f"{v:+.0f}", va="center",
            ha="left" if v > 0 else "right", fontsize=7.8,
            color=GREEN if v > 0 else "#7a3b00")
ax.set_xlim(-360, 45)
ax.invert_yaxis()
ax.text(0.02, 0.02, "绿色 = 1.0x 反而更高（仅 2017）", transform=ax.transAxes,
        fontsize=8.4, color=GREEN, va="bottom")
ax.grid(axis="x", ls=":", alpha=0.45)

# ---- (b) GA 进化曲线 ----
ax = axes[1]
best = [h[0]["fit"] for h in ga["hist"]]
gens = np.arange(len(best))
ax.plot(gens, best, "o-", color=RED, lw=2.2, ms=7, label="1500 日最优个体 fit")
ax.axhline(ga["base"]["fit"], color=GRAY, ls="--", lw=1.8,
           label=f"现行参数 fit={ga['base']['fit']:.2f}")
ax.set_xlabel("代数（种群 18，共 74 次回测）", fontsize=10)
ax.set_ylabel("适应度 = 卡玛 × 笔数折扣", fontsize=10)
ax.set_ylim(7.30, 7.80)
ax.set_title("(b) 六代只爬 0.22（8 个基因里 6 个一动没动）", fontsize=11.8, pad=10)
ax.legend(fontsize=8.8, loc="lower right")
for i, v in enumerate(best):
    ax.annotate(f"{v:.2f}", xy=(i, v), xytext=(0, 8), textcoords="offset points",
                fontsize=8.2, color="#7a3b00", ha="center")
ax.text(2.05, 7.775, "★ 样本外（3000 日）反过来：\nGA 最优 卡玛 4.43 < 现行 4.64",
        fontsize=9.0, color=RED, va="top")
ax.grid(ls=":", alpha=0.45)

# ---- (c) 无杠杆 9 变体的卡玛 ----
ax = axes[2]
rows = nl["rows"]
names = [r["name"] for r in rows]
kal = [r["metrics"]["卡玛比率"] for r in rows]
cols = []
for n, k in zip(names, kal):
    if "现金" in n:
        cols.append(GREEN)
    elif k < nl["rows"][0]["metrics"]["卡玛比率"] - 0.05:
        cols.append("#c9a0a0")
    else:
        cols.append(GRAY)
order = np.argsort(kal)
ax.barh([names[i] for i in order], [kal[i] for i in order],
        color=[cols[i] for i in order], alpha=0.92, height=0.62)
ax.axvline(nl["rows"][0]["metrics"]["卡玛比率"], color=BLUE, ls="--", lw=1.6)
ax.text(nl["rows"][0]["metrics"]["卡玛比率"] + 0.05, 0.6,
        f"基准 {nl['rows'][0]['metrics']['卡玛比率']:.2f}", color=BLUE, fontsize=8.4, va="center")
for i, idx in enumerate(order):
    ax.text(kal[idx] + 0.06, i, f"{kal[idx]:.2f}", va="center", fontsize=8.6)
ax.set_xlim(0, 8.6)
ax.tick_params(axis="y", labelsize=8.4)
ax.set_xlabel("卡玛（1500 日，1.0x）", fontsize=10)
ax.set_title("(c) 只有“闲置现金计息”高于基准，其余全部更低", fontsize=11.8, pad=10)
ax.grid(axis="x", ls=":", alpha=0.45)

fig.tight_layout(rect=(0, 0, 1, 0.945), w_pad=3.2)
out = ROOT / "第195轮_关杠杆与参数进化.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"→ {out}")
