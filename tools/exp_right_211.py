"""第 211 轮: 右侧追高的"赢家画像"训练(用户指定的方法: 先找赚钱的股票, 再画像, 再归纳条件, 再验证)。

与 209/210 轮的区别:
  209/210 是"先设条件格子 -> 看哪格均值高"(自上而下穷举);
  本轮反过来: 先定义赢家(ret10>20%), 用单特征 AUC + 二维交叉画出赢家上涨前的样子,
  归纳出"逼近 250 日新高 + 温和放量 + 低波动蓄势 + 无涨停"这一条件, 再验证它能否落地。

输出(仅打印, 不写文件):
  1) 单特征判别 AUC;
  2) 逼近新高细分与逐层叠加;
  3) 排序键头部溢价(单吊是否可行);
  4) 等权分散的年度复利(分散的上限)。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "output" / "panel_all_205.parquet"
COLS = ["d", "pct", "close_pos", "amp_pct", "mom3", "mom5", "mom10", "mom20", "mom60",
        "bias5", "bias20", "dist_low20", "dist_ph20", "dist_ph250", "ma20_slope", "ma60_slope",
        "bull", "vr", "amt_ratio", "turnover_rate", "up_streak", "lu5", "limit_up",
        "prev_pct", "vol20", "ivol20", "excess", "gap", "ret1", "ret3", "ret5", "ret10"]


def load() -> pd.DataFrame:
    df = pd.read_parquet(PANEL, columns=COLS)
    df = df.dropna(subset=["ret10"])
    df["d"] = df["d"].astype(str)
    return df


def win_auc(df: pd.DataFrame) -> None:
    """① 单特征判别: 赢家(ret10>20%) vs 输家(ret10<0) 的 AUC。"""
    win = df["ret10"] > 20
    lose = df["ret10"] < 0
    sub = df[win | lose].copy()
    y = win[win | lose].astype(int)
    feats = [c for c in COLS if c not in ("ret1", "ret3", "ret5", "ret10", "gap", "d")]
    rows = []
    for f in feats:
        x = sub[f].astype("float64")
        m = x.notna()
        if m.sum() < 1000:
            continue
        auc = roc_auc_score(y[m].astype(int), x[m])
        rows.append((f, auc))
    rows.sort(key=lambda r: -abs(r[1] - 0.5))
    print("== ① 单特征判别 AUC (|AUC-0.5| 降序) ==")
    for f, auc in rows[:12]:
        print(f"  {f:14s} {auc:.4f}  {'越大越赢' if auc > 0.5 else '越小越赢'}")


def near_high_ladder(df: pd.DataFrame) -> None:
    """② 右侧池内按距 250 日高点细分 + 逐层叠加过滤。"""
    right = df[(df["pct"] > 0) & (df["mom60"] > 0) & (df["dist_ph20"] > -5)].copy()
    nd = right["d"].nunique()
    print(f"\n== ② 右侧池 n={len(right):,} 每日{len(right)/nd:.0f} 基准均值 {right.ret10.mean():+.2f}% ==")
    right["k"] = pd.cut(right["dist_ph250"], [-999, -30, -15, -5, -3, -1, 0, 1, 3, 999],
                        labels=["<-30", "-30~-15", "-15~-5", "-5~-3", "-3~-1", "-1~0", "0~1", "1~3", ">3"])
    g = right.groupby("k", observed=True)
    print(f'  {"dist_ph250":12s}{"每日":>6s}{"赢家率":>7s}{"ret10均值":>9s}{"胜率":>7s}')
    for k in ["<-30", "-30~-15", "-15~-5", "-5~-3", "-3~-1", "-1~0", "0~1", "1~3", ">3"]:
        if k in g.size().index:
            n = g.size()[k]
            print(f'  {k:12s}{n/nd:>6.1f}{(g.ret10.apply(lambda x:(x>20).mean()*100)[k]):>7.2f}'
                  f'{g.ret10.mean()[k]:>9.2f}{(g.ret10.apply(lambda x:(x>0).mean()*100)[k]):>7.1f}')

    layers = [
        ("逼近新高(-5~0)", right["dist_ph250"].between(-5, 0)),
        ("+温和放量(vr1~2)", right["vr"].between(1.0, 2.0)),
        ("+低波动(vol20<3)", right["vol20"] < 3.0),
        ("+无涨停(lu5=0)", right["lu5"] == 0),
        ("+刚启动(mom60 0~25)", right["mom60"].between(0, 25)),
    ]
    m = pd.Series(True, index=right.index)
    print("\n  逐层叠加:")
    for nm, c in layers:
        m = m & c
        s = right[m]
        print(f'    {nm:22s} 每日{len(s)/nd:>5.1f}  ret10均值 {s.ret10.mean():+.2f}%  胜率{(s.ret10>0).mean()*100:.1f}%')


def head_premium(df: pd.DataFrame) -> None:
    """③ 逼近新高池内, 各排序键第 1 名 vs 全池均值(单吊是否可行)。"""
    pool = df[(df["pct"] > 0) & (df["mom60"] > 0) & (df["dist_ph20"] > -5)
              & df["dist_ph250"].between(-5, 0)].copy()
    print(f"\n== ③ 逼近新高池 全池 ret10 均值 {pool.ret10.mean():+.2f}% ==  各键第 1 名:")
    for col, asc, nm in [("dist_ph250", False, "距新高最近"), ("vr", True, "量比最小"),
                         ("vol20", True, "波动最低"), ("mom60", True, "刚启动"),
                         ("lu5", True, "涨停最少"), ("pct", True, "涨幅最小")]:
        p = pool.dropna(subset=[col]).copy()
        p["rk"] = p.groupby("d")[col].rank(ascending=asc, method="first")
        n = p.groupby("d")[col].transform("size")
        t1 = p[(p.rk == 1) & (n >= 2)]
        print(f'    {nm:12s} 第1名 n={len(t1):>5}  ret10 {t1.ret10.mean():+.2f}%')


def equal_weight(df: pd.DataFrame) -> None:
    """④ 等权分散: 每天买下全部信号, ret10 均值按年复利(分散的上限)。"""
    core = df[(df["pct"] > 0) & (df["mom60"] > 0) & (df["dist_ph20"] > -5)
              & df["dist_ph250"].between(-5, 0)
              & df["vr"].between(1.0, 2.0) & (df["vol20"] < 3.0) & (df["lu5"] == 0)].copy()
    core["yr"] = core["d"].str[:4]
    print(f"\n== ④ 核心池 n={len(core):,} 每日{len(core)/core.d.nunique():.1f} 等权分散年度复利 ==")
    batch = core.groupby("d")["ret10"].mean()
    byyr = batch.groupby(batch.index.str[:4]).mean()
    daily = byyr / 100 / 10
    ann = (1 + daily) ** 250 - 1
    tot = (1 + ann).prod()
    for yr in sorted(byyr.index):
        print(f'    {yr}: 近似年化 {ann[yr]*100:+6.1f}%')
    print(f'    复利总倍数 ≈ {tot:,.1f}x  年化 {(tot**(1/16.3)-1)*100:.1f}%')


def main() -> None:
    df = load()
    print(f"面板 {len(df):,} 股票日 / {df.d.nunique():,} 天 / 赢家(ret10>20%) {(df.ret10>20).mean()*100:.2f}%")
    win_auc(df)
    near_high_ladder(df)
    head_premium(df)
    equal_weight(df)


if __name__ == "__main__":
    main()
