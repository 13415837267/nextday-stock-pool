"""第二百零三轮配图: 右侧追高的年化-回撤前沿, 与"百倍"的卡玛门槛."""
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

C1, C2, C3, C4 = "#2c3e50", "#c0392b", "#2980b9", "#e67e22"
NEED = 47.25   # 3000 日做到 100 倍所需年化: 100^(1/11.9)-1


def main() -> None:
    d = json.loads((OUT / "exp_right_203_3000.json").read_text(encoding="utf-8"))
    pts = {k: v for k, v in d.items()
           if v.get("总收益率%") is not None and k not in ("bo_pb2_h5",)}

    def kind(k):
        if k.startswith(("lv2", "lv15", "gb0_lv")):
            return "lev"
        if k.startswith(("bo_", "lu_", "bp_", "dip_")):
            return "chase"
        return "lowvol"

    fig = plt.figure(figsize=(13.4, 5.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0], wspace=0.24)

    # ── 左: 年化 vs 最大回撤
    ax = fig.add_subplot(gs[0, 0])
    for k, v in pts.items():
        x = abs(v["最大回撤%"])
        y = v["年化收益率%"]
        c = {"lev": C4, "chase": C2, "lowvol": C3}[kind(k)]
        ax.scatter(x, y, s=54, color=c, alpha=.85, edgecolor="white", linewidth=.8, zorder=3)
    lbl = {"lowvol_h10": (8, -14), "lv15_h10": (-16, 8), "lv20_h10": (-4, 10),
           "lv25_h10": (6, 6), "lu_h3": (7, -12), "bp_h10": (6, 4), "lu_h3": (-34, -12)}
    for k, off in lbl.items():
        if k not in pts:
            continue
        v = pts[k]
        ax.annotate(k.replace("_h10", "").replace("_h5", "").replace("_h3", ""),
                    (abs(v["最大回撤%"]), v["年化收益率%"]),
                    textcoords="offset points", xytext=off, fontsize=8.8,
                    color={"lev": C4, "chase": C2, "lowvol": C3}[kind(k)])
    xs = [i / 10 for i in range(0, 900)]
    ax.plot(xs, [0.63 * x for x in xs], color=C3, ls="--", lw=1.2, alpha=.75)
    ax.text(16, 0.63 * 16 + 1.0, "副腿卡玛 0.63", color=C3, fontsize=9)
    ax.axhline(NEED, color=C1, ls=":", lw=1.4)
    ax.text(4, NEED + 1.5, f"3000 日做到 100 倍需年化 {NEED:.1f}%", color=C1, fontsize=9.2)
    ax.text(2, 41.2, "注：2.0× 的 -76.5% 已越过自身 -70% 平仓线", color=C2, fontsize=8.6)
    ax.set_xlim(0, 104)
    ax.set_ylim(-36, 62)
    ax.set_xlabel("最大回撤（绝对值 %）")
    ax.set_ylabel("年化收益（%）")
    ax.set_title("右侧各定义：年化与回撤几乎落在同一条线上（3000 日，2014-07 起）",
                 fontsize=11, loc="left")
    ax.grid(alpha=.22, ls=":")
    ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=c, label=l) for c, l in
                       [(C2, "严格追高（突破/涨停/回踩）"), (C3, "低波趋势（现行副腿）"),
                        (C4, "加杠杆")]], loc="lower left", fontsize=8.6, frameon=False)

    # ── 右: 卡玛对比
    ax2 = fig.add_subplot(gs[0, 1])
    names = ["主腿\n1500日", "主腿\n3000日", "主腿\n3664日", "副腿\n3000日", "副腿\n3645日"]
    kama = [6.25, 4.04, 3.35, 0.63, 0.48]
    cols = [C1, C1, C1, C4, C4]
    bars = ax2.bar(names, kama, color=cols, width=.58)
    for b, v in zip(bars, kama):
        ax2.text(b.get_x() + b.get_width() / 2, v + .1, f"{v:.2f}", ha="center",
                 fontsize=9.5, color=C1)
    ax2.axhline(NEED / 50, color=C2, ls="--", lw=1.4)
    ax2.text(4.42, NEED / 50 + .10, f"{NEED/50:.2f} 百倍门槛\n(-50% 回撤内)", color=C2,
             fontsize=8.6, ha="right")
    ax2.set_ylim(0, 7.2)
    ax2.set_ylabel("卡玛比率（年化 ÷ 最大回撤）")
    ax2.set_title("差距不在敢不敢加杠杆，在卡玛", fontsize=11, loc="left")
    ax2.grid(axis="y", alpha=.22, ls=":")

    fig.savefig(OUT / "第203轮_右侧百倍的卡玛门槛.png", dpi=170, bbox_inches="tight")
    print("saved", OUT / "第203轮_右侧百倍的卡玛门槛.png")


if __name__ == "__main__":
    main()
