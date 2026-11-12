"""第207轮配图: 左=副策略四组执行面参数(相对现行), 右=百倍要的年化在哪个位置.

左图用横向条形而不是折线: 四组的档位数不同(6/4/5/5), 折线会把"档位"和"参数值"
混成一个轴。横条 + 组间留白能一眼看出"哪一组整组都在基准线以下"。
参数标签走 y 轴刻度(而不是 text), 否则长标签会把 x 轴挤出一段负区间。
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

C1, C2, C3, C4, C5 = "#2F6F9F", "#C0392B", "#7F8C8D", "#E67E22", "#27AE60"


def main() -> None:
    d = json.loads((OUT / "exp_sat_207_3645.json").read_text(encoding="utf-8"))
    base = d["base"]["总收益率%"]
    groups = [
        ("▍挂单深度 pullback_pct", C1, [("0%（开盘直接买）", "pb00"), ("0.5%", "pb05"),
                                        ("1.5%", "pb15"), ("2.0%", "pb20"), ("3.0%", "pb30")]),
        ("▍挂单窗口 pullback_days", C5, [("1 日", "pbd1"), ("2 日", "pbd2"), ("5 日", "pbd5")]),
        ("▍止损 stop_pct", C3, [("-8%", "stop08"), ("-10%", "stop10"),
                                ("-15%", "stop15"), ("-20%", "stop20")]),
        ("▍移动止损 trail_pct", C4, [("关", "trail00"), ("5%", "trail05"),
                                     ("10%", "trail10"), ("12%", "trail12")]),
    ]

    labels, ys, bars = [], [], []
    y = 0
    for gname, col, items in groups:
        labels.append(gname)
        ys.append(y)
        y += 1
        for lab, k in items:
            labels.append("   " + lab)
            ys.append(y)
            bars.append((y, d[k]["总收益率%"] / base, col))
            y += 1
        y += 0.4
    top = y

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14.4, 6.8),
                                  gridspec_kw={"width_ratios": [1.1, 1]})

    for yy, r, col in bars:
        ax.barh(yy, r, height=0.62, color=col, alpha=.85, zorder=3)
        ax.text(r + 0.012, yy, f"{r:.2f}×", fontsize=8.8, color=col, va="center",
                ha="left", zorder=5,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=.85))
    ax.axvline(1.0, color=C2, ls="--", lw=1.6, zorder=2)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=8.8)
    for t in ax.get_yticklabels():
        if t.get_text().startswith("▍"):
            t.set_fontweight("bold")
    ax.set_ylim(-0.8, top)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.45)
    ax.set_xlabel("全史总倍数 ÷ 现行（3,645 日，1.00 = 现行 19.3×）", fontsize=10.2)
    ax.grid(axis="x", ls=":", color="#ddd", zorder=0)
    ax.set_title("四组执行面参数：只有“多等两天”在基准线之上", fontsize=11.6, loc="left")

    # ---- 右: 年化-回撤平面 ----
    pts = [("现行", "base", C1, (9, 5), "left"),
           ("挂 5 日", "pbd5", C5, (9, 5), "left"),
           ("1.5× 杠杆", "lev15", C4, (-9, 6), "right"),
           ("2.0× 杠杆", "lev20", C2, (7, 8), "left")]
    for name, k, c, off, ha in pts:
        v = d[k]
        x, y = v["最大回撤%"], 1 + v["总收益率%"] / 100
        ax2.scatter(x, y, s=160, color=c, zorder=4, edgecolor="white", linewidth=1.4)
        ax2.annotate(f"{name}  {y:,.1f}× / {v['年化收益率%']:.1f}%", (x, y),
                     textcoords="offset points", xytext=off, ha=ha, va="bottom",
                     fontsize=8.9, color=c)
    ax2.axhline(100, color=C2, ls="--", lw=1.4, zorder=2)
    ax2.text(-79.4, 106, "百倍线", color=C2, fontsize=9.2, ha="left", va="bottom")
    ax2.set_yscale("log")
    ax2.set_yticks([20, 30, 50, 100, 200])
    ax2.set_yticklabels(["20", "30", "50", "100", "200"])
    ax2.minorticks_off()
    ax2.set_xlim(-81, -41)
    ax2.set_ylim(14, 380)
    ax2.set_xlabel("最大回撤（%）", fontsize=10.2)
    ax2.set_ylabel("全史总倍数（对数轴）", fontsize=10.2)
    ax2.grid(ls=":", color="#ddd", zorder=0)
    need = (100 ** (252 / 3645) - 1) * 100
    ax2.set_title("百倍要年化 %.1f%%，杠杆到 −76%% 回撤也只有 58×" % need,
                  fontsize=11.2, loc="left")

    fig.text(0.01, -0.015,
             "全部为全史 3,645 个交易日、同一份面板与同一套风控，只改被检验的那一个键；"
             "止损四档与现行逐位相同（低波池 10 日内收盘跌不到 −8%），故条形高度全为 1.00×。",
             fontsize=8.4, color=C3)
    plt.tight_layout()
    p = OUT / "第207轮_副策略的执行面.png"
    fig.savefig(p, dpi=170, bbox_inches="tight")
    print("saved", p)


if __name__ == "__main__":
    main()
