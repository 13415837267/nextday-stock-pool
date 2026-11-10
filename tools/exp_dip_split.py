# -*- coding: utf-8 -*-
"""同一个 alpha 的互斥子集：深回调 / 强趋势（第187轮收尾二）。

为什么这是最后一条路
--------------------
第 187 轮已经把"换形态"全部测死（右侧 86 变体最好 22.5 倍且回撤 -62.6%，
放量深跌反转 9 变体最好 209%，扩板块更差，扩槽位单笔腰斩）。
结论：单笔 1%+ 的 edge 只存在于"缩量阴线 + 趋势内 + 次日低开承接"这一族。

那么第二条腿只能从**同一族的互斥子集**里切 —— 条件互斥 = 选股不重叠 =
两个账户可以各自满仓复利，不是同一策略加仓。dip 的两个天然切分维度：

    ① 回调深度:  现役 -6~-3%   |  深回调 -9~-6%（跌得更狠，低开更深，修复空间更大）
    ② 趋势强度:  现役 mom20 5~25%  |  强趋势 25~60%（趋势更陡，回踩更急）

两者都只动 dip_pct_min/max 与 dip_mom20_min/max —— 参数早已存在，
现役口径逐位不变（缺省不启用）。

判据：单笔 > 1.0% 且笔数 > 250（1500 日）才有资格独立成腿。
      单笔 < 0.8% 说明切出来的是尾部噪音，不是另一个 alpha。

用法:
    python3 tools/exp_dip_split.py --days 1500
"""
import argparse
import copy
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["BT_FULL_TRADES"] = "1"

from src.backtest import prepare_context, run_backtest  # noqa: E402
from src.config import load_config  # noqa: E402


def _row(res):
    m = res["metrics"]
    g = lambda k: float(str(m[k]).replace(",", ""))          # noqa: E731
    return {
        "总收益%": round(g("总收益率%"), 1),
        "年化%": round(g("年化收益率%"), 1),
        "回撤%": round(g("最大回撤%"), 1),
        "卡玛": round(g("卡玛比率"), 2),
        "笔数": int(g("交易次数")),
        "单笔%": round(g("平均单笔收益%"), 2),
        "胜率%": round(g("胜率%"), 1),
    }


def dp(c, pct=None, mom=None, lev=None, gap=None):
    if pct:
        c["selection"]["dip_pct_min"], c["selection"]["dip_pct_max"] = pct
    if mom:
        c["selection"]["dip_mom20_min"], c["selection"]["dip_mom20_max"] = mom
    if gap is not None:
        c["execution"]["skip_open_gain"] = gap
    if lev is not None:
        c["execution"]["leverage"] = lev


PLAN = [
    ("base", lambda c: None),
    ("deep", lambda c: dp(c, pct=(-9.0, -6.0))),
    ("deep2", lambda c: dp(c, pct=(-12.0, -8.0))),
    ("hot", lambda c: dp(c, mom=(25.0, 60.0))),
    ("deep_hot", lambda c: dp(c, pct=(-9.0, -6.0), mom=(0.0, 60.0))),
    ("deep_lev125", lambda c: dp(c, pct=(-9.0, -6.0), lev=1.25)),
    ("hot_lev125", lambda c: dp(c, mom=(25.0, 60.0), lev=1.25)),
    ("deep_g0", lambda c: dp(c, pct=(-9.0, -6.0), gap=0.0)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=1500)
    ap.add_argument("--only", default="")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    cfg0 = load_config()
    ctx = prepare_context(cfg0, args.days)
    print(f"窗口 {len(ctx['main_dates'])} 日", flush=True)

    plan = PLAN
    if args.only:
        keep = {s.strip() for s in args.only.split(",")}
        plan = [x for x in PLAN if x[0] in keep]

    tag = args.tag or str(args.days)
    out = {"days": args.days, "n_days": len(ctx["main_dates"]), "variants": {}}
    print(f"{'变体':14s}{'总收益%':>14s}{'年化%':>9s}{'回撤%':>9s}{'卡玛':>7s}"
          f"{'笔数':>6s}{'单笔%':>8s}{'胜率%':>7s}", flush=True)
    for name, fn in plan:
        c = copy.deepcopy(cfg0)
        fn(c)
        try:
            r = _row(run_backtest(c, args.days, ctx))
        except Exception as e:                       # noqa: BLE001
            print(f"{name:14s} ERROR {e}", flush=True)
            continue
        out["variants"][name] = r
        print(f"{name:14s}{r['总收益%']:>14,.0f}{r['年化%']:>8.1f}%{r['回撤%']:>8.1f}%"
              f"{r['卡玛']:>7.2f}{r['笔数']:>6d}{r['单笔%']:>+8.2f}%{r['胜率%']:>6.1f}%",
              flush=True)

    p = ROOT / f"output/exp_dip_split_{tag}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"写出 {p}", flush=True)


if __name__ == "__main__":
    main()
