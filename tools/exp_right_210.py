"""第 210 轮: 把"右侧交易"当成独立策略上引擎 —— 它到底能到多少倍.

背景
----
用户: 「继续训练右侧交易」+「副策略要达到主策略的水平」。

第 209 轮把"副策略的上限"落到了排序键上: 主策略第 1 名 3,488x, 第 2 名 19.4x,
第 3 名 25.1x —— 3,488x 是"第 1 名溢价", 不是"策略水平"。副策略 19.3x 与之同量级。

那"右侧交易"自己在引擎里是什么水平? 仓库里其实已经有一族右侧选股, 一直只当
"主力没信号时的补位"用(config.selection.fallbacks 里的 momentum), 从没独立跑过全史:
  momentum: pct 1~7% + 量比 1.2~5 + MA5>MA10>MA20 + bias20<=11.4 + mom20 0~25
第 91 轮六窗口补位实测单笔 +0.77/+1.14/+1.44/+0.29/+0.84/+1.34%, 六窗口全正 ——
但那是"补位", 每年只有几十笔, 从没算过它自己当主力的复利。

本轮做三件事:
  A. momentum 独立跑全史(关掉补位漏斗, 关掉池内评分门槛), 得到右侧的引擎基线;
  B. 名次实验(pick_offset 0/1/2): 右侧是不是也有"第 1 名溢价"? 还是反过来(越极端越差);
  C. oversold(20 日深跌后 T 日阳线反包) 同样跑一遍 —— 面板扫描里它是唯一
     "多年份都为正"的一格(第 210 轮 A 格), 语义偏反转但 K 线是右侧入场。

纪律: 不改 config.yaml, 只在实验里覆盖。
用法: python3 tools/exp_right_210.py [days] [variant ...]
"""
from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context, run_backtest   # noqa: E402
from src.config import load_config                      # noqa: E402

OUT = ROOT / "output"
# 第210轮: 全史 3645 日 + BT_FULL_TRADES=1 会 OOM(137); 只在显式需要成交明细时才开。
if os.environ.get("RIGHT_FULL_TRADES"):
    os.environ["BT_FULL_TRADES"] = "1"

MOM = {"selection": {"mode": "momentum", "fallback_slot": False,
                     "display_top_n": 1, "exec_top_n": 1},
       "execution": {"max_buy_per_day": 1, "alloc_frac_pct": 100.0}}
OSD = {"selection": {"mode": "oversold", "fallback_slot": False,
                     "display_top_n": 1, "exec_top_n": 1},
       "execution": {"max_buy_per_day": 1, "alloc_frac_pct": 100.0}}

VARIANTS = {
    # 对照: config 原样(主策略 dip), 同一 ctx / 同一窗口
    "main": {},
    # A. 右侧基线: 池内评分门 1.0(dip 的原门槛, 量纲不同, 只作对照)
    "mom_g10": MOM,
    # A2. 关掉评分门 —— 右侧每天都有候选, 门槛会砍掉大部分日子
    "mom_gnone": {**MOM, "execution": {**MOM["execution"], "min_top1_score": None}},
    # B. 名次: 第 2 / 第 3 名(判断右侧是"越极端越好"还是相反)
    "mom_off1": {**MOM, "execution": {**MOM["execution"], "min_top1_score": None},
                 "selection": {**MOM["selection"], "pick_offset": 1}},
    "mom_off2": {**MOM, "execution": {**MOM["execution"], "min_top1_score": None},
                 "selection": {**MOM["selection"], "pick_offset": 2}},
    # C0. 面板 A 格 + 风控放开(blocked_frac 1.0): 反包的机会集中在弱市, 而引擎在弱市空仓
    "osd_A_nofr": {**OSD, "execution": {**OSD["execution"], "min_top1_score": None,
                                        "blocked_frac": 1.0},
                   "selection": {**OSD["selection"],
                                 "os_pct_min": 0.0, "os_pct_max": 11.0,
                                 "os_mom20_max": -5.0, "os_dist_low20_max": 100.0,
                                 "os_dist_high20_max": -15.0,
                                 "os_vol_ratio_min": 1.8, "os_vol_ratio_max": 3.0,
                                 "os_close_pos_min": 0.60}},
    # C1. A 格 + 全买(每日候选平均 4 只, 52% 的日子只有 1 只): 检验"等权篮子均值"能否实现
    "osd_A_disp8": {**OSD, "execution": {**OSD["execution"], "min_top1_score": None,
                                         "max_buy_per_day": 8, "alloc_frac_pct": 12.5},
                    "selection": {**OSD["selection"], "exec_top_n": 8,
                                  "os_pct_min": 0.0, "os_pct_max": 11.0,
                                  "os_mom20_max": -5.0, "os_dist_low20_max": 100.0,
                                  "os_dist_high20_max": -15.0,
                                  "os_vol_ratio_min": 1.8, "os_vol_ratio_max": 3.0,
                                  "os_close_pos_min": 0.60}},
    # C. 深跌阳线反包(面板 A 格)
    "osd_gnone": {**OSD, "execution": {**OSD["execution"], "min_top1_score": None}},
    # C2. 精确复现面板 A 格: 离前高 >=15% + 放量 1.8~3 倍 + 涨幅不设上限(含反包涨停)
    "osd_A": {**OSD, "execution": {**OSD["execution"], "min_top1_score": None},
              "selection": {**OSD["selection"],
                            "os_pct_min": 0.0, "os_pct_max": 11.0,
                            "os_mom20_max": -5.0, "os_dist_low20_max": 100.0,
                            "os_dist_high20_max": -15.0,
                            "os_vol_ratio_min": 1.8, "os_vol_ratio_max": 3.0,
                            "os_close_pos_min": 0.60}},
    "osd_off1": {**OSD, "execution": {**OSD["execution"], "min_top1_score": None},
                 "selection": {**OSD["selection"], "pick_offset": 1}},
}


def merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = v
    return out


def main() -> None:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    picks = sys.argv[2:] or list(VARIANTS)
    cfg = load_config()
    print(f"== 第210轮 右侧独立回测 {days} 日 ==", flush=True)
    ctx = prepare_context(cfg, days)
    path = OUT / f"exp_right_210_{days}.json"
    res = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    for name in picks:
        if name in res:
            print(f"{name:12s} 复用 {res[name]['总收益率%']:.1f}%", flush=True)
            continue
        c = merge(cfg, VARIANTS[name])
        r = run_backtest(c, days, ctx)
        m = r["metrics"]
        row = {k: float(m.get(k, 0.0)) for k in
               ("总收益率%", "年化收益率%", "最大回撤%", "卡玛比率", "夏普比率",
                "交易次数", "胜率%", "平均单笔收益%")}
        res[name] = row
        path.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{name:12s} 总 {row['总收益率%']:>11.1f}%  年化 {row['年化收益率%']:>6.2f}%  "
              f"回撤 {row['最大回撤%']:>7.2f}%  卡玛 {row['卡玛比率']:>5.2f}  "
              f"夏普 {row['夏普比率']:>5.2f}  笔 {row['交易次数']:>5.0f}  "
              f"单笔 {row['平均单笔收益%']:+.3f}%  胜率 {row['胜率%']:.1f}%", flush=True)

    print("\n汇总:")
    for k, v in res.items():
        print(f"  {k:12s} {v['总收益率%']:>11.1f}%  年化 {v['年化收益率%']:>6.2f}%  "
              f"回撤 {v['最大回撤%']:>7.2f}%  卡玛 {v['卡玛比率']:>5.2f}  笔 {v['交易次数']:.0f}")


if __name__ == "__main__":
    main()
