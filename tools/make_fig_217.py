#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第 217 轮配图: 数据源的历史覆盖区间 —— 谁能进 14 年回测

数据全部来自 tools/probe_ds_217.py 的实测(output/probe_ds_217.json)
与 Tushare 官方接口文档标注的历史起点, 不引用二手测评。
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch

for fp in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
           "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"]:
    try:
        font_manager.fontManager.addfont(fp)
    except Exception:
        pass
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OK = "#2e7d32"      # 覆盖回测起点
SHORT = "#c62828"   # 历史不足
DEAD = "#9e9e9e"    # 已断供

# (名称, 起点年.月, 终点年.月(None=至今), 状态, 成本)
ROWS = [
    ("龙虎榜席位明细  Tushare top_inst", 2005.0, None, OK, "¥500/年"),
    ("龙虎榜每日汇总  Tushare top_list", 2005.0, None, OK, "¥200/年"),
    ("个股大单资金流  Tushare moneyflow", 2010.0, None, OK, "¥200/年"),
    ("1 分钟 K 线  zzshare stk_mins", 2010.0, None, OK, "免费"),
    ("涨停明细+原因  zzshare review", 2020.0, None, SHORT, "免费"),
    ("同花顺涨停榜  limit_list_ths", 2023.83, None, SHORT, "¥500/年"),
    ("龙虎榜席位  zzshare lhb_detail", 2025.0, None, SHORT, "免费"),
    ("北向个股持股  港交所", 2017.2, 2024.6, DEAD, "已断供"),
]

BT_START = 2012.0   # panel 覆盖起点(3664 交易日约 2012-01 起)
NOW = 2026.87


def main():
    fig, ax = plt.subplots(figsize=(12.6, 6.4))
    n = len(ROWS)
    for i, (name, s, e, color, cost) in enumerate(ROWS):
        y = n - 1 - i
        end = NOW if e is None else e
        ax.barh(y, end - s, left=s, height=0.52, color=color, alpha=0.88,
                edgecolor="white", linewidth=1.2, zorder=3)
        # 成本标签放在条形右端外侧
        label = f"{cost}   {end - s:.0f} 年" if e is None else f"2024-08 停更   共 {end - s:.1f} 年"
        ax.text(end + 0.25, y, label, va="center", ha="left", fontsize=9.6,
                color=color if e is None else DEAD, zorder=4)

    ax.set_yticks(range(n))
    ax.set_yticklabels([r[0] for r in ROWS][::-1], fontsize=10.6)
    ax.set_xlim(2004.4, 2030.2)
    ax.set_ylim(-0.9, n - 0.3)
    ax.set_xticks(range(2005, 2027, 3))
    ax.tick_params(axis="x", labelsize=10)
    ax.grid(axis="x", color="#e0e0e0", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)

    # 回测起点阈值线
    ax.axvline(BT_START, color="#37474f", ls="--", lw=1.6, zorder=5)
    ax.text(BT_START - 0.25, -0.58, "主回测起点 2012-01",
            ha="right", va="center", fontsize=9.8, color="#37474f", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none"))

    ax.set_title("数据源实测: 只有三项同时满足「长历史 + 新维度」", fontsize=14.5,
                 fontweight="bold", pad=14, loc="left")
    ax.set_xlabel("数据覆盖年份(实测最早可回补日)", fontsize=10.5)

    ax.legend(handles=[
        Patch(facecolor=OK, label="覆盖 2012 年前 → 可进主回测"),
        Patch(facecolor=SHORT, label="历史不足 → 只能做近端样本外"),
        Patch(facecolor=DEAD, label="已停止披露"),
    ], loc="upper center", bbox_to_anchor=(0.5, -0.10), fontsize=10, frameon=False, ncol=3)

    fig.text(0.012, 0.008,
             "实测: tools/probe_ds_217.py 逐一打通接口扫描 16 个年份锚点; Tushare 起点取自官方接口文档。"
             "回测 = 3664 个交易日(2012-01 起)。",
             fontsize=8.4, color="#666666")
    fig.tight_layout(rect=[0, 0.075, 1, 1])
    out = Path("第217轮_数据源覆盖区间.png")
    fig.savefig(out, dpi=170)
    print("saved", out)


if __name__ == "__main__":
    main()
