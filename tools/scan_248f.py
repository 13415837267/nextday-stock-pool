"""第248轮 F: 合入主策略前的口径校准

补位规则里的 -3.51% 阈值来自 tools/build_features_234.py 的等权指数口径:
  eq_ret = panel.pct.mean()  (不过滤停牌), rolling(20, min_periods=10)
主策略引擎 src/selector.py 的 regime 口径:
  reg.mkt_pct = panel[is_paused==0].pct.mean(), rolling(20) 全窗
两者若不等, 阈值不能直接搬。这里实测差值。
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"


def main() -> None:
    files = sorted((ROOT / "data" / "panel").glob("*.csv.gz"))
    files = [f for f in files if f.stem.replace(".csv", "") >= "20180101"]
    print(f"面板 {len(files)} 天")
    rows = []
    for f in files:
        d = pd.read_csv(f, usecols=["trade_date", "pct", "is_paused"])
        rows.append({"trade_date": int(d.trade_date.iloc[0]),
                     "pct_all": d.pct.mean(),
                     "pct_np": d.loc[d.is_paused == 0, "pct"].mean()})
    st = pd.DataFrame(rows).sort_values("trade_date").reset_index(drop=True)

    a = st.copy()
    a["idx"] = (1 + a.pct_all / 100).cumprod()
    a["ma"] = a.idx.rolling(20, min_periods=10).mean()
    a["d_a"] = (a.idx / a.ma - 1) * 100          # feat234 口径

    b = st.copy()
    b["idx"] = (1 + b.pct_np / 100).cumprod()
    b["ma"] = b.idx.rolling(20).mean()
    b["d_b"] = (b.idx / b.ma - 1) * 100          # selector 口径

    m = pd.DataFrame({"trade_date": a.trade_date, "feat234": a.d_a, "selector": b.d_b}).dropna()
    m["diff"] = m.selector - m.feat234
    print("\n【1】两种指数口径的差（百分点）")
    print(m["diff"].describe().round(3).to_string())
    print(f"  绝对差 >0.05pp 的天数: {int((m['diff'].abs()>0.05).sum())} / {len(m)}")
    print(f"  >0.2pp: {int((m['diff'].abs()>0.2).sum())}")

    # 阈值附近的判定翻转
    for th in (-3.51,):
        fa = m.feat234 <= th
        fb = m.selector <= th
        print(f"\n【2】阈值 {th}% 的判定翻转: 不一致 {int((fa != fb).sum())} 天 / 破位 "
              f"{int(fa.sum())} vs {int(fb.sum())}")

    # 与 248 用的状态表对照
    ds = pd.read_csv(OUT / "w248_daily_state.csv")
    j = m.merge(ds[["trade_date", "idx_ma20_dist"]], on="trade_date", how="inner")
    j["diff2"] = j.selector - j.idx_ma20_dist
    print(f"\n【3】与 248 状态表({len(j)} 天)对照: 均值差 {j.diff2.mean():+.4f}  "
          f"最大绝对差 {j.diff2.abs().max():.3f}")
    print(j.loc[j.diff2.abs().nlargest(3).index, ["trade_date", "idx_ma20_dist", "selector"]]
          .to_string(index=False))
    m.to_csv(OUT / "w248f_calib.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
