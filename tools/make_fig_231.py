"""第231轮配图：分散度（净值曲线 + 三指标）与限价承接的逆向选择"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context, run_backtest  # noqa: E402
from src.config import load_config  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP"]
plt.rcParams["axes.unicode_minus"] = False


def mk(cfg, n):
    c = copy.deepcopy(cfg)
    c["execution"]["max_buy_per_day"] = n
    c["execution"]["alloc_frac_pct"] = round(100.0 / n, 4)
    c["execution"]["pos_frac_cap"] = round(1.0 / n, 6)
    return c


def nav_of(res):
    s = pd.DataFrame(res["equity"])
    s["d"] = pd.to_datetime(s["d"].astype(str))
    return s.set_index("d")["nav"].astype(float)


def fig_diversify(cfg, ctx):
    curves, stats = {}, {}
    for n, lab, color in [(1, "单吊（现行）", "#c0392b"), (2, "2 只等权", "#2980b9"), (3, "3 只等权", "#7f8c8d")]:
        r = run_backtest(mk(cfg, n), None, ctx)
        curves[lab] = nav_of(r)
        m = r["metrics"]
        stats[lab] = (float(m["年化收益率%"]), float(m["最大回撤%"]), float(m["卡玛比率"] or 0), color)
        print(f"{lab} {stats[lab][:3]}", flush=True)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 6.2), gridspec_kw={"width_ratios": [1.55, 1]})
    for lab, s in curves.items():
        a1.plot(s.index, s.values / s.iloc[0], lw=1.9, label=lab, color=stats[lab][3])
    a1.set_yscale("log")
    a1.set_ylabel("净值（对数轴，起点=1）")
    a1.set_title("分散把曲线变平了——也把它压低了", fontsize=12)
    a1.legend(loc="upper left", fontsize=10)
    a1.grid(alpha=0.25, which="both")
    for lab, s in curves.items():
        a1.annotate(f"{s.iloc[-1]/s.iloc[0]:.1f}×", (s.index[-1], s.iloc[-1] / s.iloc[0]),
                    xytext=(6, 0), textcoords="offset points", fontsize=10,
                    color=stats[lab][3], va="center", fontweight="bold")

    labs = list(stats)
    xs = range(len(labs))
    ann = [stats[l][0] for l in labs]
    mdd = [abs(stats[l][1]) for l in labs]
    a2.bar([x - 0.2 for x in xs], ann, width=0.38, color="#c0392b", label="年化 %")
    a2.bar([x + 0.2 for x in xs], mdd, width=0.38, color="#95a5a6", label="|回撤| %")
    a2.set_xticks(list(xs))
    a2.set_xticklabels([l.replace("（现行）", "\n（现行）") for l in labs], fontsize=10)
    a2.set_title("年化跌 45~57pp，回撤只收窄 6~9pp", fontsize=12)
    a2.legend(fontsize=9)
    a2.grid(alpha=0.25, axis="y")
    for x, (l, v) in enumerate(zip(labs, ann)):
        a2.annotate(f"{v:.0f}%", (x - 0.2, v), ha="center", va="bottom", fontsize=9, color="#c0392b")
        a2.annotate(f"−{abs(stats[l][1]):.1f}%", (x + 0.2, abs(stats[l][1])), ha="center", va="bottom",
                    fontsize=9, color="#555")
    for x, l in enumerate(labs):
        a2.annotate(f"卡玛 {stats[l][2]:.2f}", (x, max(ann) * 1.12), ha="center", fontsize=9, color="#2c3e50")
    a2.set_ylim(0, max(ann) * 1.22)

    fig.suptitle("第231轮 · 把每天的钱拆成 N 份：买到的是平滑，付出去的是复利速度", fontsize=13, y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = ROOT / "第231轮_分散度.png"
    fig.savefig(out, dpi=150)
    print(f"已写 {out}")


def fig_limit():
    d = pd.read_csv(ROOT / "output" / "w231d_limit.csv")
    d = d[d["方案"].str.startswith(("现行", "承接"))]
    labs = [s.replace("承接", "挂 ") for s in d["方案"]]
    xs = range(len(d))
    fig, a1 = plt.subplots(figsize=(11, 6))
    a1.bar(xs, d["笔数"], width=0.5, color="#bdc3c7", label="成交笔数（左轴）")
    a1.set_ylabel("笔数", color="#555")
    a1.set_xticks(list(xs))
    a1.set_xticklabels(labs, fontsize=10)
    a1.set_ylim(0, max(d["笔数"]) * 1.25)
    for x, v in zip(xs, d["笔数"]):
        a1.annotate(f"{v}", (x, v), xytext=(0, -16), textcoords="offset points",
                    ha="center", va="top", fontsize=9, color="#555")
    a2 = a1.twinx()
    a2.plot(list(xs), d["单笔"], "-o", color="#c0392b", lw=2, ms=8, label="单笔收益 %（右轴）")
    a2.set_ylabel("单笔收益 %", color="#c0392b")
    a2.tick_params(axis="y", colors="#c0392b")
    a2.set_ylim(0.3, 1.45)
    for x, v in zip(xs, d["单笔"]):
        a2.annotate(f"{v:.2f}%", (x, v), xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=9, color="#c0392b", fontweight="bold")
    a1.set_title("想买得更便宜 → 单笔反而从 1.25% 掉到 0.47%\n"
                 "（越等越差：限价单成交的那天，正是它继续往下砸的那天）", fontsize=12)
    a1.grid(alpha=0.2, axis="y")
    h1, l1 = a1.get_legend_handles_labels()
    h2, l2 = a2.get_legend_handles_labels()
    a1.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=9)
    fig.text(0.5, 0.015, "承接-1.5% 笔数几乎不减（557 vs 559），单笔却掉 0.48pp —— 这不是运气，是逆向选择。",
             ha="center", fontsize=10, color="#444")
    fig.tight_layout(rect=[0, 0.045, 1, 1])
    out = ROOT / "第231轮_越等越差.png"
    fig.savefig(out, dpi=150)
    print(f"已写 {out}")


def fig_order():
    d2 = pd.read_csv(ROOT / "output" / "w231b_order.csv")
    d4 = pd.read_csv(ROOT / "output" / "w231c_fullhist.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.6))
    labs = ["score\n（现行）", "band\n[-3,-1]", "gap\n最深优先"]
    v2500 = [d2.loc[d2["方案"].str.startswith(k), "年化"].iloc[0]
             for k in ["score（现行）", "band[-3,-1]", "gap 最深优先"]]
    vfull = [d4.loc[d4["方案"].str.startswith(k), "年化"].iloc[0]
             for k in ["score（现行）", "band[-3,-1]", "gap 最深优先"]]
    xs = range(3)
    a1.bar([x - 0.19 for x in xs], v2500, width=0.36, color="#2980b9", label="2500 日（10年）")
    a1.bar([x + 0.19 for x in xs], vfull, width=0.36, color="#c0392b", label="4148 日（17年）")
    a1.set_xticks(list(xs))
    a1.set_xticklabels(labs, fontsize=10)
    a1.set_ylabel("年化 %")
    a1.set_ylim(0, 105)
    a1.legend(fontsize=9)
    a1.grid(alpha=0.25, axis="y")
    a1.set_title("排序口径：10 年窗口赢 6.6pp，17 年窗口输 4.6pp", fontsize=12)
    for x, (y1, y2) in enumerate(zip(v2500, vfull)):
        a1.annotate(f"{y1:.1f}", (x - 0.19, y1), ha="center", va="bottom", fontsize=9, color="#2980b9")
        a1.annotate(f"{y2:.1f}", (x + 0.19, y2), ha="center", va="bottom", fontsize=9, color="#c0392b")

    seg = d4.set_index("方案")[["s1_年化", "s2_年化", "s3_年化"]]
    tags = ["2009–2015", "2015–2021", "2021–2026"]
    for i, (name, color, mk_) in enumerate([("score（现行）", "#c0392b", "o"),
                                            ("band[-3,-1]", "#2980b9", "s")]):
        a2.plot(range(3), seg.loc[name].values, f"-{mk_}", color=color, lw=2, ms=7, label=name)
        _is_s = name.startswith("score")
        _offs = [(10, 6 if _is_s else -15), None, (-12, -8 if _is_s else 14)]
        _ha = ["left", "center", "right"]
        for x, v in enumerate(seg.loc[name].values):
            if _offs[x] is None:
                continue          # 中段两条几乎重合，标注会打架；数值在图注里
            a2.annotate(f"{v:.0f}%", (x, v), xytext=_offs[x], textcoords="offset points",
                        ha=_ha[x], fontsize=9, color=color)
    a2.set_xticks(range(3))
    a2.set_xticklabels(tags, fontsize=10)
    a2.set_ylabel("分段年化 %")
    a2.set_title("翻转发生在最早那一段（−15pp）", fontsize=12)
    a2.legend(fontsize=9)
    a2.grid(alpha=0.25)
    a2.set_ylim(0, max(seg.values.max() * 1.2, 10))
    a2.set_xlim(-0.35, 2.35)
    fig.suptitle("第231轮 · 排序口径过不了全史这道门（与229轮 close_pos 同形）", fontsize=13, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = ROOT / "第231轮_排序口径过不了全史.png"
    fig.savefig(out, dpi=150)
    print(f"已写 {out}")


if __name__ == "__main__":
    only = sys.argv[1] if len(sys.argv) > 1 else ""
    if only != "order":          # order 图不需要回测，可单独重画（省 5 分钟）
        cfg = load_config()
        ctx = prepare_context(cfg, 2500)
        fig_diversify(cfg, ctx)
        fig_limit()
    fig_order()
