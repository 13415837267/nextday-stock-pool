"""第229轮 图表：全史条件敏感性（17 年 4201 日）"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
d = pd.read_csv(ROOT / "output" / "w229_cond.csv", encoding="utf-8-sig")
base = d[d["参数"] == "基准"].iloc[0]
d = d[d["参数"] != "基准"].copy()

ORDER = ["dip_pct_min", "dip_pct_max", "dip_vol_ratio_max", "dip_mom20_min", "dip_mom20_max",
         "dip_dist_low20_min", "dip_close_pos_max", "bias20_max", "amount_min_yi",
         "turnover_min", "turnover_max", "price_min", "listed_min_bars"]

# ---------- 图1：每个条件的敏感性曲线 ----------
fig, axes = plt.subplots(3, 5, figsize=(19, 10.5))
axes = axes.ravel()
for ax, name in zip(axes, ORDER):
    g = d[d["参数"] == name].sort_values("取值")
    cur = float(g["当前"].iloc[0])
    xs = [float(v) for v in g["取值"]] + [cur]
    ys = [float(v) for v in g["Δ年化"]] + [0.0]
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    xs = [xs[i] for i in order]
    ys = [ys[i] for i in order]
    ax.axhline(0, color="#888", lw=1, ls="--")
    ax.plot(xs, ys, "-o", color="#c0392b", lw=2, ms=6)
    ax.axvline(cur, color="#2980b9", lw=1.2, ls=":", alpha=0.9)
    ax.scatter([cur], [0], s=110, color="#2980b9", zorder=5, marker="D")
    ax.set_title(f"{name}\n当前 {cur:g}", fontsize=10)
    ax.set_ylabel("Δ年化 pp", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.25)
for ax in axes[len(ORDER):]:
    ax.axis("off")
fig.suptitle(f"全史 4201 日 · 13 个选股条件的单维敏感性（基准 {base['年化']:.2f}% / 卡玛 {base['卡玛']:.2f} / {int(base['笔数'])} 笔）",
             fontsize=15)
fig.text(0.5, 0.02, "红点=该档位；蓝色菱形=现行取值。13 个条件里有 11 个的峰就在现行值上；另 2 个的正向档位只有 +0.2~+0.8pp（噪声级）。",
         ha="center", fontsize=11, color="#444")
fig.tight_layout(rect=[0, 0.035, 1, 0.95])
fig.savefig(ROOT / "第229轮_条件敏感性全史.png", dpi=130)
print("fig1 ok")

# ---------- 图2：41 档按 Δ年化 排序 ----------
s = d.sort_values("Δ年化")
fig, ax = plt.subplots(figsize=(11, 12))
colors = ["#c0392b" if v > 0 else "#27ae60" for v in s["Δ年化"]]
ax.barh([f"{r['参数']} = {float(r['取值']):g}" for _, r in s.iterrows()], s["Δ年化"], color=colors)
ax.axvline(0, color="#333", lw=1)
ax.set_xlabel("相对现行配置的年化变化（pp）", fontsize=11)
ax.set_title(f"全史 41 个条件档位：只有 2 档为正，且都 < +1pp\n（基准 {base['年化']:.2f}%，17 年 952 笔）", fontsize=13)
for i, (_, r) in enumerate(s.iterrows()):
    v = r["Δ年化"]
    ax.text(v + (1.5 if v >= 0 else -1.5), i, f"{v:+.1f}", va="center",
            ha="left" if v >= 0 else "right", fontsize=8)
ax.tick_params(labelsize=8)
ax.grid(axis="x", alpha=0.25)
fig.tight_layout()
fig.savefig(ROOT / "第229轮_41档全史排序.png", dpi=130)
print("fig2 ok")

# ---------- 图3：close_pos 连续性 ----------
p = pd.read_csv(ROOT / "output" / "w229_patch.csv", encoding="utf-8-sig")
cp = p[p["方案"].str.startswith("close_pos")].copy()
cp["x"] = cp["方案"].str.extract(r"([\d.]+)").astype(float)
cp = pd.concat([cp, p[p["方案"] == "基准"].assign(x=0.32)]).sort_values("x")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
a1.plot(cp["x"], cp["年化"], "-o", color="#c0392b", lw=2)
a1.axhline(float(base["年化"]), color="#2980b9", ls="--", lw=1.2, label=f"现行 0.32（{base['年化']:.1f}%）")
a1.set_xlabel("dip_close_pos_max（收盘在当日区间的位置上限）")
a1.set_ylabel("全史年化 %")
a1.set_title("全史合计：放宽到 0.55 有 +7.5pp")
a1.legend(fontsize=9)
a1.grid(alpha=0.25)
a2.plot(cp["x"], cp["s3_年化"], "-o", color="#8e44ad", lw=2, label="最近 1/3 段")
a2.plot(cp["x"], cp["s1_年化"], "-s", color="#16a085", lw=1.5, alpha=.8, label="最早 1/3 段")
a2.axhline(float(base["s3_年化"]), color="#8e44ad", ls="--", lw=1)
a2.axhline(float(base["s1_年化"]), color="#16a085", ls="--", lw=1)
a2.set_xlabel("dip_close_pos_max")
a2.set_ylabel("分段年化 %")
a2.set_title("拆开看：赚的是 2009-2015，亏的是最近十年")
a2.legend(fontsize=9)
a2.grid(alpha=0.25)
fig.suptitle("放宽收盘位置上限：全史账面上的 +7.5pp，是拿最近段换来的", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(ROOT / "第229轮_收盘位置放宽的真相.png", dpi=130)
print("fig3 ok")
