"""第二百三十四轮: 每个策略条件的真实贡献(留一法回测)

特征表上的"留一法"只能算均值, 不能算复利。这里用真实引擎: 从现行配置出发,
逐个把一条条件放宽到"等于没有", 看年化/笔数/单笔怎么变。

  去掉后年化大掉  -> 这条在支撑收益
  去掉后年化上升  -> 这条是负担(但可能仍在控回撤)

同时给出"全关"作为下界, 衡量整套条件合计值多少。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context, run_backtest      # noqa: E402
from src.config import load_config                          # noqa: E402

OUT = ROOT / "output" / "w234_contrib_bt.csv"


def brief(res: dict) -> dict:
    m = res["metrics"]
    nav = pd.DataFrame(res["equity"])["nav"].astype(float).values
    n = len(nav)
    seg = {}
    for i, tag in enumerate(["s1", "s2", "s3"]):
        x = nav[int(n * i / 3): int(n * (i + 1) / 3)]
        yrs = max(0.3, len(x) / 244.0)
        seg[tag] = round(((x[-1] / x[0]) ** (1 / yrs) - 1) * 100, 1) if len(x) > 40 else None
    return {"年化": round(float(m["年化收益率%"]), 2),
            "回撤": round(float(m["最大回撤%"]), 2),
            "卡玛": round(float(m["卡玛比率"] or 0), 2),
            "笔数": int(m["交易次数"]),
            "胜率": round(float(m["胜率%"]), 1),
            "单笔": round(float(m["平均单笔收益%"]), 3), **seg}


def mk(base: dict, sel_kw: dict | None = None, uni_kw: dict | None = None) -> dict:
    c = dict(base)
    c["universe"] = dict(base["universe"])
    c["selection"] = dict(base["selection"])
    c["execution"] = dict(base["execution"])
    c["selection"]["dip_weights"] = dict(base["selection"]["dip_weights"])
    c["universe"].update(uni_kw or {})
    c["selection"].update(sel_kw or {})
    return c


# 每条: 名称 -> (selection 覆盖, universe 覆盖)
REL = {
    "去掉 跌幅区间": (dict(dip_pct_min=-11.0, dip_pct_max=-0.5), {}),
    "去掉 收盘位置": (dict(dip_close_pos_max=1.0), {}),
    "去掉 动量区间": (dict(dip_mom20_min=-50.0, dip_mom20_max=200.0), {}),
    "去掉 距20日低点": (dict(dip_dist_low20_min=0.0), {}),
    "去掉 量比上限": (dict(dip_vol_ratio_max=99.0), {}),
    "去掉 乖离上限": (dict(bias20_max=99.0), {}),
    "去掉 成交额下限": ({}, dict(amount_min_yi=0.1)),
    "去掉 换手上限": ({}, dict(turnover_max=99.0)),
    "去掉 股价上限": ({}, dict(price_max=9999.0)),
}


def main() -> None:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 2500
    base = load_config()
    t0 = time.time()
    ctx = prepare_context(base, days)
    print(f"ctx {time.time()-t0:.0f}s, {len(ctx['main_dates'])} 日\n", flush=True)

    rows = []
    r0 = brief(run_backtest(base, None, ctx))
    r0["方案"] = "现行基准"
    r0["Δ年化"], r0["Δ笔数"] = 0.0, 0
    rows.append(r0)
    print(f"{'现行基准':22s} 年化 {r0['年化']:7.2f}%  回撤 {r0['回撤']:7.2f}%  "
          f"笔数 {r0['笔数']:4d}  单笔 {r0['单笔']:+.3f}%", flush=True)

    for name, (sk, uk) in REL.items():
        t = time.time()
        r = brief(run_backtest(mk(base, sk, uk), None, ctx))
        r["方案"] = name
        r["Δ年化"] = round(r["年化"] - r0["年化"], 2)
        r["Δ笔数"] = r["笔数"] - r0["笔数"]
        rows.append(r)
        print(f"{name:22s} 年化 {r['年化']:7.2f}% ({r['Δ年化']:+6.2f})  "
              f"回撤 {r['回撤']:7.2f}%  卡玛 {r['卡玛']:5.2f}  "
              f"笔数 {r['笔数']:4d} ({r['Δ笔数']:+4d})  单笔 {r['单笔']:+.3f}%  "
              f"[{time.time()-t:.0f}s]", flush=True)
        pd.DataFrame(rows)[["方案", "年化", "回撤", "卡玛", "笔数", "胜率", "单笔",
                            "s1", "s2", "s3", "Δ年化", "Δ笔数"]].to_csv(
            OUT, index=False, encoding="utf-8-sig")

    # 全关
    t = time.time()
    r = brief(run_backtest(mk(base, dict(dip_pct_min=-11.0, dip_pct_max=-0.5,
                                        dip_close_pos_max=1.0,
                                        dip_mom20_min=-50.0, dip_mom20_max=200.0,
                                        dip_dist_low20_min=0.0,
                                        dip_vol_ratio_max=99.0, bias20_max=99.0),
                            dict(amount_min_yi=0.1, turnover_max=99.0, price_max=9999.0)),
                           None, ctx))
    r["方案"] = "条件全关(只剩评分)"
    r["Δ年化"] = round(r["年化"] - r0["年化"], 2)
    r["Δ笔数"] = r["笔数"] - r0["笔数"]
    rows.append(r)
    print(f"{'条件全关':22s} 年化 {r['年化']:7.2f}% ({r['Δ年化']:+6.2f})  "
          f"回撤 {r['回撤']:7.2f}%  笔数 {r['笔数']:4d}  单笔 {r['单笔']:+.3f}%  "
          f"[{time.time()-t:.0f}s]", flush=True)

    pd.DataFrame(rows)[["方案", "年化", "回撤", "卡玛", "笔数", "胜率", "单笔",
                        "s1", "s2", "s3", "Δ年化", "Δ笔数"]].to_csv(
        OUT, index=False, encoding="utf-8-sig")
    print("\ndone", flush=True)


if __name__ == "__main__":
    main()
