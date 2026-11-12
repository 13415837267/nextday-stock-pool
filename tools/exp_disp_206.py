"""第206轮: 分散度 —— 满仓单吊是数学上的自我惩罚.

几何均值 = 算术均值 - 方差/2。单吊把方差顶到最大, 所以"每天挑 1 只、全仓压上"
在算术均值相同的候选里, 拿到的复利是最差的那一档。
本脚本只改三个互相绑定的量(买几只/截断几只/每只多少仓), 不动任何选股条件。

注: config 里 exec_top_n 的历史注释提到过 "2/3/5 仍只买 1 只"(只多 9~12 笔),
    那是"多看几只但只买最前一只", 与本轮的"真分散"不是一件事。
用法: python3 tools/exp_disp_206.py [days] [n2 n3 ...]
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
os.environ["BT_FULL_TRADES"] = "1"


def variant(n: int) -> dict:
    return {"selection": {"exec_top_n": n},
            "execution": {"max_buy_per_day": n, "alloc_frac_pct": 100.0 / n}}


VARIANTS = {f"n{k}": variant(k) for k in (1, 2, 3, 4, 5)}


def merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = v
    return out


def main() -> None:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 3645
    want = sys.argv[2:] or ["n2", "n3", "n4"]
    cfg = load_config()
    print(f"== 第206轮 分散度实验 {days} 日: {want} ==", flush=True)
    ctx = prepare_context(cfg, days)
    path = OUT / f"exp_disp_206_{days}.json"
    res = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    for name in want:
        if name in res:
            print(f"{name:4s} 复用 {res[name]['总收益率%']:.2f}%", flush=True)
            continue
        c = merge(cfg, VARIANTS[name])
        r = run_backtest(c, days, ctx)
        mm = r["metrics"]
        m = {k: mm.get(k, 0.0) for k in
             ("总收益率%", "年化收益率%", "最大回撤%", "卡玛比率", "交易次数", "胜率%")}
        res[name] = m
        path.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{name:4s} 总 {m['总收益率%']:>12.1f}%  年化 {m['年化收益率%']:>6.2f}%  "
              f"回撤 {m['最大回撤%']:>7.2f}%  卡玛 {m['卡玛比率']:>5.2f}  "
              f"笔 {m['交易次数']:>5.0f}  胜率 {m['胜率%']:.1f}%", flush=True)

    print("\n汇总(按卡玛):")
    for k, v in sorted(res.items(), key=lambda x: -(x[1]["卡玛比率"])):
        print(f"  {k:4s} {v['总收益率%']:>12.1f}%  年化 {v['年化收益率%']:>6.2f}%  "
              f"回撤 {v['最大回撤%']:>7.2f}%  卡玛 {v['卡玛比率']:>5.2f}  笔 {v['交易次数']}")


if __name__ == "__main__":
    main()
