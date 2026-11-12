"""第 205 轮配图: 追高为什么不行 + 副策略的钱到底从哪来."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

C1, C2, C3, C4 = "#B22222", "#2E7D5B", "#4A4A4A", "#8C8C8C"

top1 = pd.read_csv(OUT / "scan_right_top1_205.csv")
d = json.load(open(OUT / "satellite_bt_3000.json", encoding="utf-8"))
t = pd.DataFrame(d["trades"])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.2))

# ---- 左: 六种追法 ----
top1 = top1.sort_values("net")
lab = {"pct(追最强)": "追当日最强涨幅", "vr(追最放量)": "追最放量",
       "dist_ph20(追最贴前高)": "追最贴 20 日前高", "mom20(追最强趋势)": "追最强 20 日趋势",
       "excess(追最抗跌后转强)": "追当日超额最大", "ivol20(追最稳)": "追波动最低(最不激进)"}
ys = list(range(len(top1)))
for i, (_, r) in enumerate(top1.iterrows()):
    c = C1 if r["net"] < -1.0 else C4
    ax1.barh(i, r["net"], color=c, height=0.62)
    ax1.text(r["net"] - 0.06, i, f"{r['net']:+.2f}%", va="center", ha="right",
             fontsize=9, color=C3)
ax1.set_yticks(ys)
ax1.invert_yaxis()          # 最差(最负)排在最上面
ax1.set_yticklabels([lab.get(k, k) for k in top1["key"]], fontsize=9.5)
ax1.axvline(0, color=C3, lw=1)
ax1.set_xlim(-2.6, 0.35)
ax1.set_xlabel("每日只买 1 只、持有 10 个交易日的净收益（已扣 0.3% 成本）", fontsize=9.5)
ax1.set_title("六种“追高”各追 3,943 个交易日：没有一种为正", fontsize=11.5, loc="left")
ax1.grid(axis="x", ls=":", color="#ddd")


# ---- 右: 215 笔按退出原因 ----
g = t.copy()
g["grp"] = g["note"].str.replace(r" 跌停顺延1日开盘", "", regex=True)
# 两类固定止损合计只有 2 笔(单笔 -13%/-18%), 单独成柱会把其余柱子压扁 -> 并入止损类
g.loc[g["grp"].str.startswith("止损"), "grp"] = "移动止损 8%(自高点)"
agg = (g.groupby("grp")
        .agg(n=("ret_pct", "size"), mean=("ret_pct", "mean"), med=("ret_pct", "median"))
        .sort_values("mean", ascending=True))
name_cn = {"大盘转弱清仓": "大盘转弱清仓", "持满 10 个交易日": "持满 10 个交易日",
           "移动止损 8%(自高点)": "移动止损 8%", "止损 -13.0%": "止损 -13%",
           "止损 -13.5%": "止损 -13.5%"}
ys2 = list(range(len(agg)))
for i, (k, r) in enumerate(agg.iterrows()):
    c = C2 if r["mean"] > 0 else C1
    ax2.barh(i, r["mean"], color=c, height=0.55)
    # 长柱子用白字写在柱内(深灰压深红看不清), 短柱子写在柱外空白
    if r["mean"] < -1.0:
        ax2.text(r["mean"] + 0.18, i, f"{r['mean']:+.2f}%  ({r['n']:.0f} 笔)",
                 va="center", ha="left", fontsize=9.3, color="white")
    else:
        ax2.text(r["mean"] + (0.55 if r["mean"] > 0 else 0.28), i,
                 f"{r['mean']:+.2f}%  ({r['n']:.0f} 笔)",
                 va="center", ha="left", fontsize=9.3, color=C3)
ax2.set_yticks(ys2)
ax2.set_yticklabels([name_cn.get(k, k) for k in agg.index], fontsize=9.5)
ax2.axvline(0, color=C3, lw=1)
ax2.set_xlim(-4.6, 10.4)
ax2.set_xlabel("每笔净收益（%）", fontsize=9.5)
ax2.set_title("副策略 215 笔：54% 被大盘闸门打断，钱全在“能拿满”的那 74 笔", fontsize=11.5,
              loc="left")
ax2.grid(axis="x", ls=":", color="#ddd")


fig.text(0.01, -0.03,
         "左：样本 973,035 个上涨股票日（全史 3,943 天，无抽样），每日按各键只买 1 只。"
         "右：副策略 3,000 日回测的 215 笔真实成交，按退出原因分组；括号内为笔数。",
         fontsize=8.3, color=C4)
plt.tight_layout()
p = OUT / "第205轮_追高无解与副策略的钱从哪来.png"
plt.savefig(p, dpi=170, bbox_inches="tight")
print("saved", p, file=__import__("sys").stderr)
