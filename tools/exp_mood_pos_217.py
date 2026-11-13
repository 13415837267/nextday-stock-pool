"""第二百一十七轮(续三): 情绪质量分能不能变成钱 —— 仓位倍率模拟

前两版已确认事实:
  1) 情绪质量分(连板高度↑ 烂板率↓ 游资占比↑ 题材集中度↑ 封单比↓)与单笔收益秩相关
     rho=0.163 (t=3.32), 滚动标准化无前视版本 rho=0.130 (t=2.60);
  2) 控制大盘后仍成立(偏相关 0.169>原始 0.163), 且大盘本身与单笔收益几乎无关(rho=-0.036);
  3) 三个大盘档内"质量高-低"差分别为 +2.39 / +1.68 / +0.42, 全部为正。

本脚本只回答"值多少钱", 不改引擎、不改 config:
  用**滚动分位**(过去 250 日, 无前视)把质量分切成三档, 给不同仓位倍率,
  在 411 笔真实成交上重算净值曲线, 对比基准的年化/回撤/卡玛。

口径限制(必须随结论一起报):
  · ret_pct 是价格收益, 不含仓位; 这里把倍率乘上去是**近似**, 忽略了现金拖累与
    资金占用(单吊, 同期只有一笔, 影响可忽略);
  · 倍率档位是看完分组结果才定的, 属于样本内设定; 因此必须看**年度**是否每年都改善,
    任一年明显恶化即否决(188 轮的教训: 最优组合全靠 2024)。

用法: python3 tools/exp_mood_pos_217.py
产物: output/exp_mood_pos_217.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "exp_mood_pos_217.json"


def metrics(nav: pd.Series, years: float) -> dict:
    nav = nav.dropna()
    if len(nav) < 2 or years <= 0:
        return {}
    total = float(nav.iloc[-1] / nav.iloc[0])
    cagr = total ** (1 / years) - 1
    dd = float((nav / nav.cummax() - 1).min())
    return {"总倍": round(total, 2), "年化%": round(cagr * 100, 2),
            "回撤%": round(dd * 100, 2),
            "卡玛": round(cagr / abs(dd), 3) if dd else None}


def main() -> None:
    emo = pd.read_parquet(ROOT / "output" / "uplimit_emo.parquet")
    emo["date"] = emo["date"].astype(str)

    # 滚动 250 日分位(不含当日), 无前视
    q = emo["quality"].astype(float)
    mu = q.shift(1).rolling(250, min_periods=60).mean()
    sd = q.shift(1).rolling(250, min_periods=60).std()
    emo["q_roll"] = (q - mu) / sd.replace(0, np.nan)
    emo["q_rank"] = emo["q_roll"].rolling(250, min_periods=60).apply(
        lambda s: float((s.iloc[-1] > s.iloc[:-1]).mean()), raw=False)

    bt = json.loads((ROOT / "output" / "backtest.json").read_text(encoding="utf-8"))
    tr = pd.DataFrame(bt["trades_all"])
    tr = tr[tr["ret_pct"].notna()].copy()
    tr["signal_date"] = tr["signal_date"].astype(str)
    tr = tr[tr["signal_date"] >= "20200102"].sort_values("signal_date")
    m = tr.merge(emo[["date", "q_roll", "q_rank", "quality"]],
                 left_on="signal_date", right_on="date", how="left")

    # 只用能对齐的样本做公平比较(否则基准占便宜)
    m = m[m["q_rank"].notna()].reset_index(drop=True)
    d0 = pd.to_datetime(m["signal_date"].min())
    d1 = pd.to_datetime(m["signal_date"].max())
    years = max(0.5, (d1 - d0).days / 365.25)
    base_px = float(m["ret_pct"].mean())
    print(f"样本 {len(m)} 笔 {m['signal_date'].min()} ~ {m['signal_date'].max()} ({years:.2f} 年)")
    print(f"基准平均单笔 {base_px:.3f}%")

    schemes = {
        "基准(全程满仓)": lambda r: 1.0,
        "三档 1.0/0.7/0.4": lambda r: 1.0 if r > 2/3 else (0.7 if r > 1/3 else 0.4),
        "三档 1.0/0.8/0.6": lambda r: 1.0 if r > 2/3 else (0.8 if r > 1/3 else 0.6),
        "三档 1.2/1.0/0.5": lambda r: 1.2 if r > 2/3 else (1.0 if r > 1/3 else 0.5),
        "线性 0.4+0.6r": lambda r: 0.4 + 0.6 * r,
        "二分 1.0/0.5": lambda r: 1.0 if r > 0.5 else 0.5,
        "反向(对照)": lambda r: 0.4 if r > 2/3 else (0.7 if r > 1/3 else 1.0),
    }

    res: dict = {"样本": {"n": int(len(m)), "起": m["signal_date"].min(),
                        "止": m["signal_date"].max()}, "方案": {}, "年度": {}}

    for name, fn in schemes.items():
        mult = np.array([fn(r) for r in m["q_rank"]])
        r = m["ret_pct"].to_numpy() / 100.0
        nav = np.cumprod(1 + r * mult)
        s = pd.Series(nav)
        # 每单位仓位的平均单笔: > 基准单笔 说明"降仓降在了该降的地方"
        per_unit = float(np.average(r * 100, weights=mult)) / max(1e-9, float(mult.mean()))
        res["方案"][name] = {**metrics(s, years),
                             "平均仓位": round(float(mult.mean()), 3),
                             "总收益%": round((nav[-1] - 1) * 100, 1),
                             "单位仓位单笔%": round(per_unit, 3),
                             "选股alpha": round(per_unit - base_px, 3)}

    # 年度分解: 每个方案在每一年的收益
    m["year"] = m["signal_date"].str[:4]
    for name, fn in schemes.items():
        rows = []
        for y, gy in m.groupby("year"):
            mult = np.array([fn(r) for r in gy["q_rank"]])
            r = gy["ret_pct"].to_numpy() / 100.0
            rows.append({"年": y, "n": int(len(gy)),
                         "收益%": round(float(np.prod(1 + r * mult) - 1) * 100, 1)})
        res["年度"][name] = rows

    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'方案':<18}{'总倍':>9}{'年化%':>8}{'回撤%':>8}{'卡玛':>7}{'均仓':>6}{'单位仓位单笔%':>12}")
    for k, v in res["方案"].items():
        print(f"{k:<18}{v['总倍']:>9.1f}{v['年化%']:>8.1f}{v['回撤%']:>8.2f}"
              f"{str(v['卡玛']):>7}{v['平均仓位']:>6.2f}{v['单位仓位单笔%']:>12.3f}")

    print("\n年度收益%:")
    years = [r["年"] for r in res["年度"]["基准(全程满仓)"]]
    print(f"{'方案':<18}" + "".join(f"{y:>9}" for y in years))
    for k, rows in res["年度"].items():
        print(f"{k:<18}" + "".join(f"{r['收益%']:>9.1f}" for r in rows))
    print(f"\n写入 {OUT}")


if __name__ == "__main__":
    main()
