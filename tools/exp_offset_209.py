"""第 209 轮: 副策略能不能用"主策略的同一种选股、换一个名次"达到主策略的水平.

背景
----
用户: 「副策略要达到主策略的水平」。主策略 3,488x / 年化 75.46% / 839 笔 / 单笔 1.161%;
副策略 19.3x / 22.63% / 265 笔 / 单笔 1.595%。**单笔更高, 但一年只做 18 笔。**

拆开看: 单笔 log 收益 主策略 0.977% vs 副策略 1.117%(副策略更高 14%), 差距 100% 来自笔数
(839 vs 265 = 3.16 倍)。而笔数 x 持有期是守恒的 —— 缩短持有期不增加总收益, 只增加摩擦
(第 192 轮已证)。**唯一能变的是"每天赚多少": 主策略 0.80%/天, 副策略 0.14%/天, 差 5.7 倍。**

这个密度由"低波池"这个定义决定(ivol 前 10%)。所以本轮问的不是"怎么调参", 而是一个
更硬的问题: **副策略若改用主策略的选股、只换一个名次(第 2/3 名, 与主策略不重叠),
能到什么水平?** 这就是 engine 里的 selection.pick_offset(第 187 轮加的开关, 缺省 0)。

纪律: 不改 config.yaml。只在实验里覆盖 pick_offset。
用法: python3 tools/exp_offset_209.py [days]
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


def merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = v
    return out


VARIANTS = {"off0": 0, "off1": 1, "off2": 2}


def main() -> None:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 3645
    cfg = load_config()
    print(f"== 第209轮 名次偏移 {days} 日 ==", flush=True)
    ctx = prepare_context(cfg, days)
    path = OUT / f"exp_offset_209_{days}.json"
    res = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    for name, off in VARIANTS.items():
        if name in res:
            print(f"{name:5s} 复用 {res[name]['总收益率%']:.1f}%", flush=True)
            continue
        c = merge(cfg, {"selection": {"pick_offset": off}})
        r = run_backtest(c, days, ctx)
        m = r["metrics"]
        row = {k: float(m.get(k, 0.0)) for k in
               ("总收益率%", "年化收益率%", "最大回撤%", "卡玛比率", "夏普比率",
                "交易次数", "胜率%", "平均单笔收益%")}
        res[name] = row
        path.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{name:5s} 总 {row['总收益率%']:>12.1f}%  年化 {row['年化收益率%']:>6.2f}%  "
              f"回撤 {row['最大回撤%']:>7.2f}%  卡玛 {row['卡玛比率']:>5.2f}  "
              f"笔 {row['交易次数']:>5.0f}  单笔 {row['平均单笔收益%']:+.3f}%", flush=True)

    print("\n汇总:")
    for k, v in res.items():
        print(f"  {k:5s} {v['总收益率%']:>12.1f}%  年化 {v['年化收益率%']:>6.2f}%  "
              f"回撤 {v['最大回撤%']:>7.2f}%  卡玛 {v['卡玛比率']:>5.2f}  笔 {v['交易次数']:.0f}")


if __name__ == "__main__":
    main()
