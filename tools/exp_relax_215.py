#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第 215 轮: 放宽硬约束(不加杠杆) —— 持有期 × 分散 全矩阵扫描
  用户裁定: 不加杠杆, 放宽「T+1买T+2卖、满仓单吊」的硬约束, 继续提盈。
  保持选股逻辑(config selection)逐位不变, 只动 execution 三个开关:
    hold_extra_days  持有期延长(0=原口径, N=T+2后再多持N日 -> 共持有1+N个交易日)
    max_buy_per_day  每日最多买入只数(1=单吊)
    alloc_frac_pct   每笔目标仓位(分散时同步下调)
  双窗口交叉: 全史 3664 日 + 近 6 年 1500 日。
"""
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtest import prepare_context, run_backtest  # noqa: E402
from src.config import load_config  # noqa: E402


def M(m):
    return {
        "年化%": round(m["年化收益率%"], 2),
        "回撤%": round(m["最大回撤%"], 2),
        "卡玛": round(m["卡玛比率"], 2),
        "笔数": m["交易次数"],
        "胜率%": round(m["胜率%"], 2),
        "均单%": round(m["平均单笔收益%"], 3),
    }


def build_variants(base):
    """返回 [(名称, config深拷贝), ...]"""
    variants = []
    # 1) 持有期扫描(单吊, 满仓)
    for he in [0, 1, 2, 3, 4, 8, 13, 18]:
        c = copy.deepcopy(base)
        c["execution"]["hold_extra_days"] = he
        variants.append((f"持有{1 + he}日·单吊", c))
    # 2) 分散扫描(T+2 原口径, 等权)
    for n, frac in [(2, 50.0), (3, 33.0), (5, 20.0)]:
        c = copy.deepcopy(base)
        c["execution"]["max_buy_per_day"] = n
        c["execution"]["alloc_frac_pct"] = frac
        variants.append((f"T+2·分散{n}只×{frac:.0f}%", c))
    # 3) 组合(持有期 × 分散)
    for n, frac, he in [(2, 50.0, 2), (2, 50.0, 8), (3, 33.0, 2), (3, 33.0, 8)]:
        c = copy.deepcopy(base)
        c["execution"]["max_buy_per_day"] = n
        c["execution"]["alloc_frac_pct"] = frac
        c["execution"]["hold_extra_days"] = he
        variants.append((f"分散{n}只×{frac:.0f}%·持有{1 + he}日", c))
    return variants


def main():
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    base = load_config()
    ctx = prepare_context(base, days)
    rows = []
    for name, c in build_variants(base):
        try:
            r = run_backtest(c, days, ctx)
            row = M(r["metrics"])
            row["变体"] = name
        except Exception as e:  # noqa: BLE001
            row = {"变体": name, "错误": str(e)[:150]}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    out = Path(f"output/relax_215_{days}.json")
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    print(f"== 已写 {out} ==", flush=True)


if __name__ == "__main__":
    main()
