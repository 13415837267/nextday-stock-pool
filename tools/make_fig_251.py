"""第251轮: 补位收敛图

四格:
  1. 笔数-卡玛散点: 从 306 笔一路砍到 64 笔, 卡玛在哪里最高
  2. 逐年补位笔数: 306 -> 71 砍在哪几年
  3. 破位分档单笔: 指数深度 / 距高深度, 门槛数字的来源
  4. 净值: 主策略 / 249 口径 / 251 口径
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"

RED, GREEN, BLUE, GOLD, GREY = "#D94A3D", "#2E9E62", "#3A7BD5", "#D4A017", "#8b949e"
DARK = "#22272e"
BG = "#ffffff"

for fp in ("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
           "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
           "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"):
    if Path(fp).exists():
        font_manager.fontManager.addfont(fp)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()
        break
plt.rcParams.update({"axes.unicode_minus": False, "figure.facecolor": BG,
                     "axes.facecolor": BG, "axes.edgecolor": "#d0d7de",
                     "grid.color": "#e8ecf0", "text.color": DARK,
                     "axes.labelcolor": DARK, "xtick.color": DARK, "ytick.color": DARK})


def main() -> None:
    runs = pickle.load(open(OUT / "w251_runs_2500.pkl", "rb"))
    d = {o["name"]: o for o in runs}
    a = d["A 主策略单独"]
    b8 = [o for o in runs if o["name"].startswith("B8")][0]
    b4 = [o for o in runs if o["name"].startswith("B4")][0]

    j249 = json.load(open(OUT / "w249_combo_2500.json"))
    eqA = j249["A 主策略单独"]["equity"]
    eq0 = j249["B 补位只填空档"]["equity"]

    fig, ax = plt.subplots(2, 2, figsize=(13.2, 9.6))
    fig.suptitle("第251轮 · 单吊系统的补位收敛：306 笔 → 71 笔，卡玛 4.64 → 5.37",
                 fontsize=15, color=DARK, y=0.98)

    # ---------- 1 笔数-卡玛（横向条形，避免散点标签重叠） ----------
    a1 = ax[0, 0]
    pts = [
        ("249 口径 · 306 笔", 306, 4.64, BLUE),
        ("只加冷却10 · 106 笔", 106, 3.99, GREY),
        ("门槛-4.2 · 89 笔", 89, 4.09, GREY),
        ("门槛+冷却15 · 79 笔", 79, 4.25, GREY),
        ("仓位60/80 · 76 笔", 76, 4.51, GREY),
        ("仓位80/100 · 76 笔", 76, 3.95, GREY),
        ("251 定稿 · 71 笔", 71, 5.37, RED),
        ("冷却20 · 64 笔", 64, 3.87, GREY),
        ("主策略单独 · 0 笔", 0, 4.52, GOLD),
    ]
    ys = np.arange(len(pts))[::-1]
    vals = [v for _, _, v, _ in pts]
    a1.barh(ys, vals, 0.62, color=[c for _, _, _, c in pts], alpha=0.9)
    a1.axvline(4.52, color=GOLD, ls="--", lw=1.2, zorder=1)
    a1.set_yticks(ys)
    a1.set_yticklabels([nm for nm, _, _, _ in pts], fontsize=9)
    a1.set_xlabel("卡玛比率（虚线 = 主策略单独 4.52）")
    a1.set_title("砍到 71 笔最高；再砍（64 笔）或重仓（80/100）都掉下来", fontsize=12, color=DARK)
    for y, v in zip(ys, vals):
        a1.text(v + 0.05, y, f"{v:.2f}", va="center", fontsize=8.5, color=DARK)
    a1.set_xlim(3.6, 5.62)
    a1.grid(axis="x", alpha=0.5, lw=0.6)

    # ---------- 2 逐年笔数 ----------
    a2 = ax[0, 1]
    f0 = pd.DataFrame(j249["B 补位只填空档"]["metrics"].get("x") or [])
    # 249 的逐年笔数从旧 pkl 拿不到, 用 w248 明细按 buy_year 不可得 -> 用 w251 的 B0 无, 改画两条曲线的净值贡献
    # 这里改用"补位笔数按年" = 从 w248_cands 的触发日 + 引擎口径近似不可比, 故直接画 251 定稿的逐年笔数 + 单笔
    f8 = b8["fill"].copy()
    g = f8.assign(yr=f8.buy_date.str[:4]).groupby("yr").agg(
        n=("ret_pct", "size"), avg=("ret_pct", "mean"))
    yrs = sorted(g.index)
    xs = np.arange(len(yrs))
    a2.bar(xs, g["n"].values, 0.6, color=[RED if v > 0 else GREEN for v in g["avg"].values],
           alpha=0.85)
    a2.set_xticks(xs)
    a2.set_xticklabels([f"{y[2:]}" for y in yrs], fontsize=9)
    a2.set_ylabel("补位笔数")
    a2.set_title("251 定稿的补位分布：一年 3~13 笔（红=当年单笔为正）", fontsize=12, color=DARK)
    for x, n, av in zip(xs, g["n"].values, g["avg"].values):
        a2.text(x, n + 0.25, f"{av:+.1f}%", ha="center", fontsize=8,
                color=RED if av > 0 else GREEN)
    a2.set_ylim(0, max(g["n"].values) * 1.32)
    a2.grid(axis="y", alpha=0.45, lw=0.6)

    # ---------- 3 分档单笔 ----------
    a3 = ax[1, 0]
    c = pd.read_csv(OUT / "w248_cands.csv")
    c["r"] = c["y"] - 0.3
    b = c[c.brk]
    lab, val, cnt = [], [], []
    for lo, hi, nm in ((-99, -5.0, "指数 ≤ -5%"), (-5.0, -4.2, "指数 -5~-4.2%"),
                       (-4.2, -3.51, "指数 -4.2~-3.51%")):
        s = b[(b.idx > lo) & (b.idx <= hi)]
        lab.append(nm); val.append(s.r.mean()); cnt.append(len(s))
    for lo, hi, nm in ((-99, -55, "距高 ≤ -55%"), (-55, -45, "距高 -55~-45%"),
                       (-45, -38, "距高 -45~-38%"), (-38, -35, "距高 -38~-35%")):
        s = b[(b.dh60 > lo) & (b.dh60 <= hi)]
        lab.append(nm); val.append(s.r.mean()); cnt.append(len(s))
    ys = np.arange(len(lab))[::-1]
    cols = [RED if v > 0.8 else (GOLD if v > 0 else GREEN) for v in val]
    a3.barh(ys, val, 0.62, color=cols, alpha=0.88)
    a3.axvline(0, color=DARK, lw=1)
    a3.set_yticks(ys)
    a3.set_yticklabels(lab, fontsize=9)
    a3.set_xlabel("单笔收益 %（已扣 0.3% 成本）")
    a3.set_title("门槛数字的来源：浅破位与浅超跌是负贡献", fontsize=12, color=DARK)
    for y, v, n in zip(ys, val, cnt):
        a3.text(v + (0.06 if v > 0 else -0.06), y, f"{v:+.2f}%  ({n} 笔)",
                va="center", ha="left" if v > 0 else "right", fontsize=8, color=DARK)
    a3.set_xlim(min(val) * 3.0 - 0.3, max(val) * 1.55 + 0.6)
    a3.axhline(3.5, color="#c9d1d9", lw=1, ls=":")
    a3.grid(axis="x", alpha=0.45, lw=0.6)

    # ---------- 4 净值 ----------
    a4 = ax[1, 1]
    for eq, nm, col, lw in ((eqA, "主策略单独", GREY, 1.6),
                            (eq0, "249 口径（306 笔补位）", BLUE, 1.6),
                            (b8["equity"], "251 定稿（71 笔）", RED, 2.0)):
        ds = pd.to_datetime([x["d"] for x in eq])
        nav = np.array([x["nav"] for x in eq], dtype=float)
        a4.plot(ds, nav / nav[0], color=col, lw=lw, label=nm)
    a4.set_yscale("log")
    a4.set_ylabel("净值（对数轴，起点=1）")
    a4.set_title("251 比 249 少赚一截，但回撤从 -23.3% 收到 -17.3%", fontsize=12, color=DARK)
    a4.legend(fontsize=8.5, loc="upper left", framealpha=0.9)
    a4.grid(alpha=0.45, lw=0.6)
    a4.tick_params(axis="x", labelsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.955))
    out = ROOT / "第251轮_补位收敛.png"
    fig.savefig(out, dpi=150, facecolor=BG)
    print("已存", out)


if __name__ == "__main__":
    main()
