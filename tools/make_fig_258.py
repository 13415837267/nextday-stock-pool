# -*- coding: utf-8 -*-
"""第258轮配图: 三窗口卡玛/回撤 + 3200 日按年对比"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

for fp in ("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
           "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
           "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"):
    if Path(fp).exists():
        font_manager.fontManager.addfont(fp)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130

RED, BLUE, GREY, DARK = "#c0392b", "#2c6fbb", "#95a5a6", "#2c3e50"

WIN = ["1500 日\n(2020-2026)", "2500 日\n(2016-2026)", "3200 日\n(2013-2026)"]
CAL_A = [8.65, 5.56, 4.72]
CAL_M = [8.84, 5.57, 4.83]
DD_A = [17.21, 17.21, 23.49]
DD_M = [17.21, 17.21, 22.93]
ANN_A = [148.91, 95.73, 110.83]
ANN_M = [152.10, 95.89, 110.84]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.9))

# ---------- 左: 三窗口 ----------
x = np.arange(3)
w = 0.34
b1 = ax1.bar(x - w / 2, CAL_A, w, color=GREY, label="现状（11-15 只满仓）", zorder=2)
b2 = ax1.bar(x + w / 2, CAL_M, w, color=BLUE, label="本轮（11-15 只半仓）", zorder=2)
for b, v, a, dd in zip(b1, CAL_A, ANN_A, DD_A):
    ax1.text(b.get_x() + b.get_width() / 2, v + 0.16, f"{v:.2f}", ha="center",
             va="bottom", fontsize=11, color=DARK)
    ax1.text(b.get_x() + b.get_width() / 2, 0.45, f"年化\n{a:.0f}%", ha="center",
             va="bottom", fontsize=9, color="white", linespacing=1.5)
for b, v, a, dd in zip(b2, CAL_M, ANN_M, DD_M):
    ax1.text(b.get_x() + b.get_width() / 2, v + 0.16, f"{v:.2f}", ha="center",
             va="bottom", fontsize=11.5, color=BLUE, fontweight="bold")
    ax1.text(b.get_x() + b.get_width() / 2, 0.45, f"年化\n{a:.0f}%", ha="center",
             va="bottom", fontsize=9, color="white", linespacing=1.5)
for i in range(3):
    ax1.text(x[i] - w / 2, -0.60, f"-{DD_A[i]:.2f}%", ha="center", va="top",
             fontsize=8.4, color="#7f8c8d")
    ax1.text(x[i] + w / 2, -0.60, f"-{DD_M[i]:.2f}%", ha="center", va="top",
             fontsize=8.4, color=BLUE)

ax1.set_xticks(x)
ax1.set_xticklabels(WIN, fontsize=9.6)
ax1.tick_params(axis="x", pad=24)
ax1.set_xlabel("柱顶＝卡玛　柱内＝年化　柱下＝最大回撤", fontsize=8.6, color="#566573", labelpad=6)
ax1.set_ylabel("卡玛比率", fontsize=10)
ax1.set_ylim(0, 10.8)
ax1.set_title("三个窗口，半仓都不差\n"
              "1500 日年化 +3.19pp ｜ 2500 日持平 ｜ 3200 日回撤少 0.56pp",
              fontsize=11.0, pad=10)
ax1.legend(fontsize=8.6, frameon=False, loc="upper right")
ax1.grid(axis="y", ls=":", alpha=0.4, zorder=0)
ax1.spines["top"].set_visible(False)

# ---------- 右: 3200 日按年 ----------
yrs = [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020,
       2021, 2022, 2023, 2024, 2025, 2026]
RA = [3.8, 62.5, 453.4, 115.2, 4.2, 27.1, 19.4, 76.8,
      267.9, 303.5, 41.0, 133.3, 115.2, 120.0]
RM = [3.6, 63.3, 446.3, 117.6, -0.6, 18.8, 20.5, 82.7,
      281.9, 300.5, 44.7, 128.3, 122.2, 121.3]
xx = np.arange(len(yrs))
ax2.bar(xx - 0.2, RA, 0.4, color=GREY, label="现状", zorder=2)
ax2.bar(xx + 0.2, RM, 0.4, color=BLUE, label="本轮", zorder=2)
ax2.set_yscale("symlog", linthresh=1)
ax2.set_yticks([-1, 0, 1, 3, 10, 30, 100, 300, 500])
ax2.set_yticklabels(["-1%", "0", "1%", "3%", "10%", "30%", "100%", "300%", "500%"], fontsize=8.4)
for i, (a, b) in enumerate(zip(RA, RM)):
    d = b - a
    col = BLUE if d > 0 else (RED if d < 0 else GREY)
    top = max(a, b)
    ax2.text(xx[i], top * 1.30 + 0.8, f"{d:+.1f}", ha="center", va="bottom",
             fontsize=7.8, color=col, fontweight="bold")
ax2.set_xticks(xx)
ax2.set_xticklabels([str(y) for y in yrs], fontsize=7.8, rotation=45)
ax2.set_ylabel("年度收益 %（对数刻度）", fontsize=9.6)
ax2.set_title("3200 日逐年：8 胜 6 负\n（2017/2018 退步最多，2021/2020 补回；年化持平）",
              fontsize=11.2, pad=10)
ax2.legend(fontsize=8.8, frameon=False, loc="upper left")
ax2.grid(axis="y", ls=":", alpha=0.35, zorder=0)
ax2.axhline(0, color=DARK, lw=0.8)
for sp in ("top", "right"):
    ax2.spines[sp].set_visible(False)

plt.tight_layout()
out = Path("/workspace/repo/第258轮_11-15半仓.png")
plt.savefig(out, bbox_inches="tight", facecolor="white")
print("saved", out)
