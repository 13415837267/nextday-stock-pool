# -*- coding: utf-8 -*-
"""第 198 轮配图: 副策略参数面扫描 + 收益集中度 + 2015 回撤解剖 + 挂单有效期。"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Sans CJK SC", "WenQuanYi Zen Hei",
                                   "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

C_MAIN = "#c0392b"
C_SUB = "#2c6fbb"
C_GREY = "#7f8c8d"
C_GREEN = "#1e8449"
C_RED = "#c0392b"

FILES_1500 = ["exp_sat_198_1500.json", "exp_sat_198b_1500.json", "exp_sat_198c_1500.json",
              "exp_sat_198d_1500.json", "exp_sat_198e_1500.json", "exp_sat_198f_1500.json",
              "exp_sat_198g_1500.json"]

LABEL = {
    "base": "现行", "mp2": "4→2 只", "mp3": "4→3 只", "mp4": "4 只并发",
    "h5": "持 5 日", "h7": "持 7 日", "h15": "持 15 日", "h20": "持 20 日",
    "stop8": "止损 -8%", "stop15": "止损 -15%", "trail0": "关移动止损", "trail12": "移动止损 12%",
    "buf0": "入场缓冲 0", "buf10": "入场缓冲 1%", "buf20": "入场缓冲 2%",
    "pb0": "开盘直接买", "pb2": "限价 -2%", "ivol20": "低波池放宽到 20%",
    "gxloser": "只清浮亏", "gxoff": "关清仓闸门", "ma60": "闸门换 MA60", "ma60_loser": "MA60+只清浮亏",
    "mp4_loser": "4 只+只清浮亏", "mp4_ma60": "4 只+MA60", "mp4_pb2": "4 只+限价 -2%",
    "mp4_b2": "4 只每日买 2", "gxloser_pb2": "只清浮亏+限价 -2%",
    "e_bmin30": "宽度地板 30%", "e_bmin35": "宽度地板 35%", "e_vol25": "波动 >25% 停买",
    "e_vol30": "波动 >30% 停买", "e_nearhi": "须近 60 日高点", "e_pool30": "池子 ≥30 只",
    "e_gap3": "跳空上限 3%", "e_gap2": "跳空上限 2%",
    "xb05": "清仓缓冲 0.5%", "xb10": "清仓缓冲 1%", "xb20": "清仓缓冲 2%",
    "xc2": "连 2 日跌破才清", "xc3": "连 3 日跌破才清", "xb10_c2": "缓冲 1%+连 2 日",
    "wait1": "挂单 1 日(旧实盘)", "wait2": "挂单 2 日", "wait3": "挂单 3 日(现行)",
    "wait4": "挂单 4 日", "wait5": "挂单 5 日", "wait6": "挂单 6 日", "wait8": "挂单 8 日",
    "wait10": "挂单 10 日", "wait12": "挂单 12 日", "live_old": "挂单 1 日(旧实盘)",
    "xma60": "入场须站上 MA60", "xma120": "入场须站上 MA120", "xma250": "入场须站上 MA250",
    "xma120_mp4": "MA120+4 只", "t3": "只做 3% 移动止损", "t4": "只做 4% 移动止损",
    "t5": "只做 5% 移动止损", "t6": "只做 6% 移动止损", "t4_gate_loser": "4%+只清浮亏",
    "t5_gate_all": "5%+清仓闸门", "xlim15": "持满挂 +1.5%", "xlim20": "持满挂 +2%",
    "xlim30": "持满挂 +3%", "xlim20d5": "持满挂 +2%(5 日)", "mp2_b1": "2 只每日买 1",
}


def sweep() -> tuple[list[tuple[str, float, float]], float]:
    base_ann = None
    rows = []
    for f in FILES_1500:
        p = OUT / f
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        b = d.get("base") or {}
        base_ann = base_ann or float(b.get("年化收益率%") or 0)
        for k, v in d.items():
            if k == "base":
                continue
            rows.append((k, float(v["年化收益率%"]) - float(b["年化收益率%"]),
                         float(v["卡玛比率"]) - float(b["卡玛比率"])))
    # 去重(同一变体可能出现在多个文件): 保留第一次
    seen, out = set(), []
    for k, da, dk in rows:
        if k in seen:
            continue
        seen.add(k)
        out.append((k, da, dk))
    out.sort(key=lambda x: x[1])
    return out, base_ann


def noise_band(n: int = 3000, block: int = 10, seed: int = 198) -> float:
    """基准净值做块自助: 年化收益的标准差 = "什么都不改, 只换样本"的噪声尺度。"""
    d = json.loads((OUT / "satellite_bt.json").read_text(encoding="utf-8"))
    nav = pd.Series({str(x["d"]).replace("-", "")[:8]: float(x["nav"]) for x in d["equity"]})
    r = nav.pct_change().dropna().values
    T = len(r)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        starts = rng.integers(0, T - block, size=int(np.ceil(T / block)))
        s = np.concatenate([r[i:i + block] for i in starts])[:T]
        out.append((np.prod(1 + s) ** (250 / T) - 1) * 100)
    return float(np.std(out))


def fig():
    rows, base_ann = sweep()
    band = noise_band()
    fig = plt.figure(figsize=(13.6, 9.6))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.26,
                          left=0.085, right=0.975, top=0.885, bottom=0.075)

    # ---------- (a) 参数面: 年化差 vs 卡玛差 ----------
    ax = fig.add_subplot(gs[0, 0])
    xs = [r[1] for r in rows]
    ys = [r[2] for r in rows]
    ax.axhline(0, color="#333", lw=0.9)
    ax.axvline(0, color="#333", lw=0.9)
    ax.axhspan(-0.05, 0.05, xmin=0.0, xmax=1.0, color="#ecf0f1", zorder=0)
    ax.scatter(xs, ys, s=34, c=[C_MAIN if (x > 0 and y > 0) else
                                ("#e67e22" if x > 0 else C_GREY) for x, y in zip(xs, ys)],
               edgecolor="white", linewidth=0.6, zorder=3)
    tagged = {"wait8": (18, -8), "wait4": (14, -26), "xc2": (-40, 26), "mp4": (-24, 30),
              "h5": (-26, -14), "xma60": (-14, -18), "pb0": (10, -6), "gxoff": (-16, -22)}
    for k, (dx, dy) in tagged.items():
        if k not in [r[0] for r in rows]:
            continue
        i = [r[0] for r in rows].index(k)
        ax.annotate(LABEL.get(k, k), (xs[i], ys[i]), xytext=(dx, dy), textcoords="offset points",
                    fontsize=7.2, color="#2c3e50",
                    arrowprops=dict(arrowstyle="-", color="#95a5a6", lw=0.6))
    ax.annotate("改善的一侧挤在原点附近（最大 +3.3pp / +0.15）",
                xy=(3.3, 0.15), xytext=(-58, 0.52), fontsize=7.6, color=C_MAIN,
                arrowprops=dict(arrowstyle="->", color=C_MAIN, lw=0.9))
    ax.annotate("变差的一侧拉得很远（-58pp / -2.3）", xy=(-58, -2.27), xytext=(-57, -1.62),
                fontsize=7.8, color=C_GREY,
                arrowprops=dict(arrowstyle="->", color=C_GREY, lw=0.9))
    ax.set_ylim(-2.45, 0.72)
    ax.set_xlim(-63, 14)
    ax.set_xlabel("年化收益差（pp, 1500 日）", fontsize=9)
    ax.set_ylabel("卡玛差", fontsize=9)
    ax.set_title(f"(a) {len(rows)} 个变体：改善微乎其微，回退代价很大", fontsize=10.5, loc="left")
    ax.grid(alpha=0.2, ls="--", lw=0.5)

    # ---------- (b) 挂单有效期 ----------
    ax2 = fig.add_subplot(gs[0, 1])
    e15 = json.loads((OUT / "exp_sat_198e_1500.json").read_text(encoding="utf-8"))
    e30 = json.loads((OUT / "exp_sat_198e_3000.json").read_text(encoding="utf-8"))
    x15 = [(3, e15["wait3"]["年化收益率%"]), (4, e15["wait4"]["年化收益率%"]),
           (6, e15["wait6"]["年化收益率%"]), (8, e15["wait8"]["年化收益率%"]),
           (10, e15["wait10"]["年化收益率%"]), (12, e15["wait12"]["年化收益率%"])]
    x30 = [(1, e30["live_old"]["年化收益率%"]), (3, e30["base"]["年化收益率%"]),
           (4, e30["wait4"]["年化收益率%"]), (8, e30["wait8"]["年化收益率%"])]
    ax2.plot(*zip(*x15), "o-", color=C_MAIN, lw=1.8, ms=5, label="1500 日")
    ax2.plot(*zip(*x30), "s--", color=C_SUB, lw=1.8, ms=5, label="3000 日（全史）")
    ax2.axhline(e15["base"]["年化收益率%"], color=C_MAIN, lw=0.8, ls=":", alpha=0.6)
    ax2.axhline(e30["base"]["年化收益率%"], color=C_SUB, lw=0.8, ls=":", alpha=0.6)
    ax2.annotate("第 198 轮前的实盘口径\n（只撮合 T+1 一次）", xy=(1, e30["live_old"]["年化收益率%"]),
                 xytext=(4.6, e30["live_old"]["年化收益率%"] - 3.4), fontsize=7.6, color=C_SUB,
                 arrowprops=dict(arrowstyle="->", color=C_SUB, lw=0.9))
    ax2.set_ylim(min(e30["live_old"]["年化收益率%"], 25) - 2, 50)
    ax2.set_xlabel("回踩限价单有效交易日数（pullback_days）", fontsize=9)
    ax2.set_ylabel("年化收益（%）", fontsize=9)
    ax2.set_title("(b) 挂单多等几天：回测与实盘的 4.8~7.2pp 差距", fontsize=10.5, loc="left")
    ax2.legend(fontsize=8, loc="lower right")
    ax2.grid(alpha=0.22, ls="--", lw=0.5)

    # ---------- (c) 收益集中度 ----------
    ax3 = fig.add_subplot(gs[1, 0])
    d = json.loads((OUT / "satellite_bt.json").read_text(encoding="utf-8"))
    r = np.sort(np.array([t["ret_pct"] for t in d["trades"]]))[::-1]
    cum = np.cumsum(r)
    ax3.bar(np.arange(1, len(r) + 1), r, color=np.where(r > 0, C_MAIN, C_GREEN), width=0.85)
    ax3.plot(np.arange(1, len(r) + 1), cum, color="#333", lw=1.2, ls="--")
    ax3.axhline(0, color="#333", lw=0.8)
    top6 = float(cum[5] / cum[-1] * 100)
    ax3.axvline(6.5, color=C_RED, lw=1.0, ls=":")
    ax3.annotate(f"前 6 笔合计 {cum[5]:.0f}pp\n= 全部 {cum[-1]:.0f}pp 的 {top6:.0f}%",
                 xy=(6.5, cum[5]), xytext=(22, cum[5] * 0.92), fontsize=8.2, color=C_RED,
                 arrowprops=dict(arrowstyle="->", color=C_RED, lw=1.0))
    ax3.set_xlabel("逐笔收益（按大小排序，共 106 笔）", fontsize=9)
    ax3.set_ylabel("单笔收益 / 累计（%）", fontsize=9)
    ax3.set_title("(c) 收益靠右尾：砍左尾的规则一定撞到这个分布", fontsize=10.5, loc="left")
    ax3.grid(axis="y", alpha=0.22, ls="--", lw=0.5)

    # ---------- (d) 2015 回撤解剖 ----------
    ax4 = fig.add_subplot(gs[1, 1])
    d3 = json.loads((OUT / "satellite_bt_3000.json").read_text(encoding="utf-8"))
    nav = pd.Series({str(x["d"]).replace("-", "")[:8]: float(x["nav"]) for x in d3["equity"]})
    bm = pd.Series({str(x["d"]).replace("-", "")[:8]: float(x["nav"]) for x in d3["benchmark"]})
    lo, hi = "20151125", "20160715"
    n2 = (nav[(nav.index >= lo) & (nav.index <= hi)] / nav[nav.index >= lo].iloc[0])
    b2 = (bm[(bm.index >= lo) & (bm.index <= hi)] / bm[bm.index >= lo].iloc[0])
    x = np.arange(len(n2))
    ax4.plot(x, (n2.values - 1) * 100, color=C_MAIN, lw=2.0, label="副策略净值")
    ax4.plot(x[:len(b2)], (b2.values - 1) * 100, color=C_GREY, lw=1.3, label="市场等权基准")
    ax4.axhline(0, color="#333", lw=0.7)
    i = int(np.argmin(n2.values))
    ax4.annotate(f"最深 {n2.values[i] * 100 - 100:.1f}%\n区间只有 11 笔交易",
                 xy=(i, (n2.values[i] - 1) * 100), xytext=(len(x) * 0.42, 9),
                 fontsize=8.4, color=C_RED,
                 arrowprops=dict(arrowstyle="->", color=C_RED, lw=1.0))
    ax4.set_ylim(-52, 22)
    ax4.set_xlabel(f"{lo[:4]}-{lo[4:6]} ~ {hi[:4]}-{hi[4:6]} 交易日序号", fontsize=9)
    ax4.set_ylabel("相对区间起点的涨跌（%）", fontsize=9)
    ax4.set_title("(d) 全史卡玛 0.62 的窟窿：2015-11 → 2016-06", fontsize=10.5, loc="left")
    ax4.legend(fontsize=8, loc="lower left")
    ax4.grid(alpha=0.22, ls="--", lw=0.5)

    fig.suptitle("第 198 轮：副策略参数面已扫平 —— 能改的只剩「账本没按回测的规则挂单」",
                 fontsize=12.5)
    p = ROOT.parent / "第198轮_副策略参数面与实盘口径.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print("写出", p, "噪声带", round(band, 2), "变体", len(rows))


if __name__ == "__main__":
    fig()
