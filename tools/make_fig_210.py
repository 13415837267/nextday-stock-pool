"""第210轮配图: 左=单笔×笔数(复利的两个乘子), 右=池均值 vs 排序键第1名.

左图讲的是什么: 右侧的笔数多(738/524 vs 106), 但单笔薄(0.28%/0.09% vs 1.16%);
  总倍数 = (1+单笔)^笔数, 所以"笔数多"补不上"单笔薄"。
右图讲的是本轮最硬的发现: A 格(深跌反包)全池 h1 均值 +1.188%, 但**任何排序键**
  的第 1 名都只有 0.26~0.41% —— 等权均值里的收益分布在"哪只票"上无法事前识别。
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

C1, C2, C3, C4 = "#2F6F9F", "#C0392B", "#7F8C8D", "#E67E22"


def main() -> None:
    d = json.loads((OUT / "exp_right_210_3000.json").read_text(encoding="utf-8"))

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15.6, 6.2),
                                  gridspec_kw={"width_ratios": [1.12, 1]})

    # ---------------- 左: 单笔 × 笔数 ----------------
    def mul(k): return 1.0 + d[k]["总收益率%"] / 100.0

    rows = [
        ("主策略 dip",      d["main"]["平均单笔收益%"], int(d["main"]["交易次数"]), mul("main"), C1),
        ("右侧动量",        d["mom_gnone"]["平均单笔收益%"], int(d["mom_gnone"]["交易次数"]), mul("mom_gnone"), C4),
        ("深跌反包 A 格",   d["osd_A"]["平均单笔收益%"], int(d["osd_A"]["交易次数"]), mul("osd_A"), C2),
    ]
    names = [r[0] for r in rows]
    ys = range(len(rows))
    for i, (nm, per, n, mult, col) in enumerate(rows):
        ax.barh(i, per, color=col, height=0.52, alpha=0.92)
        ax.text(per + 0.018, i, f"{per:.3f}% × {n} 笔 = {mult:,.1f}×",
                va="center", fontsize=10.4, fontweight="bold", color=col)
    ax.set_yticks(list(ys))
    ax.set_yticklabels(names, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.86)
    ax.set_xlabel("平均单笔收益（3000 日，含费用）", fontsize=10)
    ax.set_title("笔数多补不上单笔薄：1,058 笔 × 0.46% 追不上 717 笔 × 1.25%",
                 fontsize=11.4, fontweight="bold", pad=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", ls=":", alpha=0.45)

    # ---------------- 右: 池均值 vs 第 1 名 ----------------
    keys = ["跌得\n最少", "量\n最小", "收得\n最低", "离低点\n最远", "体量\n最大", "体量\n最小", "涨幅\n最小"]
    vals = [0.312, 0.288, 0.262, 0.407, 0.299, 0.389, 0.329]
    x = range(len(vals))
    ax2.bar(x, vals, color=C3, width=0.62, alpha=0.9)
    ax2.axhline(1.188, color=C2, lw=2.0, ls="--")
    ax2.text(len(vals) - 0.45, 1.238, "全池等权 1.188%", color=C2, fontsize=10.2,
             fontweight="bold", ha="right")
    for i, v in enumerate(vals):
        ax2.text(i, v + 0.028, f"{v:.2f}", ha="center", fontsize=9.2, color="#333")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(keys, fontsize=8.8)
    ax2.set_ylim(0, 1.42)
    ax2.set_ylabel("T+1 日内收益（%）", fontsize=10)
    ax2.set_title("七个排序键，第 1 名全部只有池均值的 1/3",
                  fontsize=11.4, fontweight="bold", pad=10)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)
    ax2.grid(axis="y", ls=":", alpha=0.45)

    fig.tight_layout()
    p = OUT / "第210轮_右侧的可实现上限.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print("saved", p)


if __name__ == "__main__":
    main()
