# -*- coding: utf-8 -*-
"""时段切分检验: 参数完全冻结, 按自然年/固定窗口切段独立回测。

用途: 量化表观收益对"时段"的依赖度 —— 回答两个问题:
  ① 盈利是集中在某几段(拟合), 还是逐年都有(结构)?
  ② 最差的一段有多差(尾部风险的真实形状, 而非全样本最大回撤)?

**重要: 这不是样本外验证**。参数本身是在全部历史上选出来的, 切段只是改变了
"在哪段上结账", 不改变"参数见过这段". 真正的样本外请用 tools/walk_forward.py。
判读标准: 正收益段数占比、最差段、单笔收益中位数; 若多数段为正且最差段仅为
小额负值, 说明不存在"某年崩掉"的过拟合特征(必要条件, 非充分条件)。

用法:
    python tools/period_split.py                 # 全史按年
    python tools/period_split.py --by 250        # 按 250 交易日窗口
    python tools/period_split.py --days 1500     # 只切最近 1500 日
"""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context, run_backtest  # noqa: E402
from src.config import load_config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=3377, help="总窗口交易日数")
    ap.add_argument("--by", default="year", help="year 或窗口长度(交易日), 如 250")
    ap.add_argument("--min-len", type=int, default=60, help="过短段跳过阈值")
    a = ap.parse_args()

    cfg = load_config(str(ROOT / "config.yaml"))
    print(f"[build] 构建上下文({a.days} 日)...", flush=True)
    ctx = prepare_context(cfg, a.days)
    md = ctx["main_dates"]
    print(f"[build] {md[0]} ~ {md[-1]}  {len(md)} 日\n", flush=True)

    if a.by == "year":
        groups = [(y, [d for d in md if d[:4] == y]) for y in sorted({d[:4] for d in md})]
    else:
        n = int(a.by)
        groups = [(f"W{i // n + 1:02d}", md[i:i + n]) for i in range(0, len(md), n)]

    rows = []
    for tag, sub in groups:
        if len(sub) < a.min_len:
            print(f"{tag:8s} {len(sub):4d}日  -- 过短跳过", flush=True)
            continue
        c2 = dict(ctx)
        c2["main_dates"] = sub
        r = run_backtest(cfg, None, c2)
        m = r["metrics"]
        del r
        gc.collect()
        print(f"{tag:8s} {len(sub):4d}日  {sub[0]}~{sub[-1]}  "
              f"收益={m['总收益率%']:>9.2f}%  回撤={m['最大回撤%']:>7.2f}%  "
              f"笔数={m['交易次数']:>3d}  胜率={m['胜率%']:>5.1f}%  "
              f"单笔={m['平均单笔收益%']:>6.2f}%", flush=True)
        rows.append(m)

    if not rows:
        print("无有效分段")
        return 1
    rets = [r["总收益率%"] for r in rows]
    avgs = [r["平均单笔收益%"] for r in rows if r["交易次数"] > 0]
    print(f"\n{'=' * 96}")
    print(f"段数={len(rows)}  正收益 {sum(1 for x in rets if x > 0)}/{len(rows)}  "
          f"中位={sorted(rets)[len(rets) // 2]:.2f}%  最好={max(rets):.2f}%  最差={min(rets):.2f}%  "
          f"均值={sum(rets) / len(rets):.2f}%")
    print(f"单笔收益 中位={sorted(avgs)[len(avgs) // 2]:.2f}%  最差={min(avgs):.2f}%  "
          f"负单笔段数={sum(1 for x in avgs if x <= 0)}")
    print("注: 分段独立起算(各段 100% 初始资金), 段间收益不可直接连乘对齐全样本。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
