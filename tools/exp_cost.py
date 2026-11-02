# -*- coding: utf-8 -*-
"""执行成本敏感性 —— 第一百二十八轮。

研究问题
--------
回测里的成本假设是"佣金万2.5(双边) + 印花税万5(卖出) + 滑点千1(双边)"。
其中**滑点是唯一无法从公开数据校准、只能靠实盘经验估计的一项**, 而它直接乘在
每一笔的买卖价上: 单笔期望只有 +1.15%(全史 891 笔), 双边各多 0.2% 就是 -0.4pp,
相当于砍掉三分之一的边际。

本工具回答的不是"策略还能不能更好", 而是"这套 11 年 3000 倍的曲线, 有多少
是建立在'能在开盘价买到、在收盘价卖掉'这个假设上的"。这决定了:
  · 实盘该不该用市价单硬吃开盘/尾盘;
  · 若实际成交比回测差 0.3%, 目标的年化该怎么下调;
  · 大资金(冲击成本)下这套口径还剩多少。

口径
----
· 只改 execution.slippage, 其余参数/选股/买卖时点全部不动 —— 比的是"同一批交易、更差的成交";
· 滑点是**双边**的: slippage=0.003 意味着买入价 +0.3%、卖出价 -0.3%, 往返 0.6%;
· 默认跑 1500 日(与首页同口径), 可 --days 3378 跑全史(慢约 3 倍)。

用法
----
  python3 tools/exp_cost.py                 # 1500 日, 5 档滑点
  python3 tools/exp_cost.py --days 3378
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context, run_backtest  # noqa: E402
from src.config import load_config  # noqa: E402

# (标签, 滑点, 说明)
VARIANTS = [
    ("base_10bp", 0.001, "现行假设: 双边各千1"),
    ("slip_20bp", 0.002, "双边各千2(开盘价成交但有一档滑价)"),
    ("slip_30bp", 0.003, "双边各千3(小资金市价单的现实上限)"),
    ("slip_50bp", 0.005, "双边各千5(流动性差的票/大资金)"),
    ("slip_100bp", 0.010, "双边各 1%(压力测试: 还能不能赚)"),
]

FIELDS = ["总收益率%", "年化收益率%", "最大回撤%", "卡玛比率", "夏普比率",
          "交易次数", "胜率%", "平均单笔收益%", "总费用(元)"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=1500)
    ap.add_argument("--out", default="output/exp_cost.json")
    args = ap.parse_args()

    cfg0 = load_config()
    t0 = time.time()
    print(f"[1/2] 构建上下文 {args.days} 日 ...", flush=True)
    ctx = prepare_context(cfg0, args.days)
    print(f"      ctx ready {time.time() - t0:.0f}s", flush=True)

    rows = []
    for name, slip, desc in VARIANTS:
        cfg = copy.deepcopy(cfg0)
        cfg["execution"]["slippage"] = slip
        t = time.time()
        try:
            res = run_backtest(cfg, args.days, ctx)
        except Exception as e:  # noqa: BLE001
            print(f"{name} FAIL {e!r}", flush=True)
            rows.append({"name": name, "slippage": slip, "desc": desc, "error": repr(e)[:200]})
            continue
        m = res.get("metrics") or {}
        row = {"name": name, "slippage": slip, "desc": desc,
               **{k: m.get(k) for k in FIELDS}, "sec": round(time.time() - t)}
        rows.append(row)
        print("  " + json.dumps(row, ensure_ascii=False), flush=True)

    b = next((r for r in rows if r["name"] == "base_10bp" and "error" not in r), None)
    if b:
        print(f"\n=== 相对基线(滑点千1, {args.days} 日) ===", flush=True)
        print(f"  {'变体':11s} {'滑点':>6s} {'收益%':>12s} {'回撤%':>8s} {'卡玛':>6s} {'笔数':>5s} {'单笔%':>7s} {'相对':>8s}", flush=True)
        for r in rows:
            if "error" in r:
                continue
            rel = (r["总收益率%"] / b["总收益率%"] - 1) * 100 if b["总收益率%"] else 0
            print(f"  {r['name']:11s} {r['slippage']*1e4:5.0f}bp {r['总收益率%']:12.2f} "
                  f"{r['最大回撤%']:8.2f} {r['卡玛比率']:6.2f} {r['交易次数']:5d} "
                  f"{r['平均单笔收益%']:7.3f} {rel:+7.1f}%", flush=True)

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"days": args.days, "rows": rows},
                                   ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"\n结果已写 {out_path} (总耗时 {time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
