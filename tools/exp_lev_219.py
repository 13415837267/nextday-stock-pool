"""第219轮: 把杠杆价目表在同一份 config、同一窗口上重算一遍

用途:
  回答"年化 200% 有什么方法"时, 杠杆是唯一能到达的路径, 但历史数字(第 170 轮)是在
  旧 config(基线年化 104.04%)上跑的, 与当前基线 111.76% 不同口径, 放进同一张图会误导。
  本轮只改 execution.leverage, 其它一个数不动, 1500 日窗口重跑 6 档。

注意:
  · leverage > 1 时引擎允许透支, 按 margin_rate=5.5% 对透支额逐日计息;
  · pos_frac_cap=1.0 是"单笔目标仓位上限", 与 leverage 是两件事, 不冲突;
  · 本实验**不改变** config 落盘值, 只用于给"代价"标价。
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.backtest import run_backtest  # noqa: E402

LEVS = [1.0, 1.25, 1.4, 1.5, 1.75, 2.0]


def main(days: int = 1500) -> None:
    base = load_config()
    out = {"days": days, "margin_rate": base["execution"].get("margin_rate"), "runs": {}}
    for lev in LEVS:
        cfg = copy.deepcopy(base)
        cfg["execution"]["leverage"] = lev
        t0 = time.time()
        res = run_backtest(cfg, days)
        m = res["metrics"]
        row = {"年化%": m.get("年化收益率%"), "总收益%": m.get("总收益率%"),
               "回撤%": m.get("最大回撤%"), "卡玛": m.get("卡玛比率"),
               "笔数": m.get("交易次数"), "单笔%": m.get("平均单笔收益%"),
               "费用": m.get("总费用(元)"), "秒": round(time.time() - t0, 1)}
        out["runs"][str(lev)] = row
        print(f"lev={lev:<5} " + "  ".join(f"{k}={v}" for k, v in row.items()), flush=True)
    (ROOT / "output" / "exp_lev_219.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=1500)
    main(ap.parse_args().days)
