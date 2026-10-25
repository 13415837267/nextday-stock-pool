# -*- coding: utf-8 -*-
"""第四十五轮: 防线审计(事前注册, 零参数变更, 纯诊断)。

背景: 第四十四轮实证"参数不可调"后, 策略的两道防线(大盘风控层+失效监测层)
从未在十年历史里被量化检验过。本轮回答三个问题:
  A1) 大盘风控层十年贡献——关掉后 era 前/后各差多少? (防的是急跌还是阴跌?)
  A2) 恐慌熔断(vol_max)与市场宽度(breadth)的单独贡献拆分;
  B)  失效监测器十年回放——2016年起如果它在线, 多快报警? era后误报多少?
      假想"halt即停机"能避免多少亏损? (反事实诊断, 监测器实际仅告警)

事前注册口径:
  - Part A 变体仅3组: no_regime(enabled=false) / no_panic(vol_max=0) /
    no_breadth(breadth_min=-1, 宽度条件恒过而趋势+恐慌仍在);
    regime序列依赖cfg, 每组独立prepare_context(本地缓存命中约74s/组)。
  - Part B 按sell_date结算序逐笔滚动compute_monitor(与实盘"卖出结算后更新"一致);
    era切分锚点20240901沿用第四十三/四十四轮。
  - 假想停机为描述性反事实: halt首次触发后不再开新仓, 统计其后已执行笔合计;
    不构成执行逻辑变更(监测器仍仅告警)。

用法: python3 tools/defense_audit.py
输出: 控制台 + output/defense_audit.json
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context, run_backtest  # noqa: E402
from src.config import load_config  # noqa: E402
from src.monitor import compute_monitor  # noqa: E402

ERA = "20240901"
DAYS = 2500
OUT = ROOT / "output" / "defense_audit.json"

VARIANTS = [   # (key, 说明, 修改路径, 值)
    ("no_regime", "全关(均线+宽度+恐慌熔断)", ("regime", "enabled"), False),
    ("no_panic",  "仅关恐慌熔断",             ("regime", "vol_max"), 0),
    ("no_breadth","仅关市场宽度",             ("regime", "breadth_min"), -1.0),
]


def seg_stats(trades: list, era_after: bool) -> dict:
    sel = [t for t in trades if t.get("ret_pct") is not None
           and ((t["buy_date"] >= ERA) == era_after)]
    if not sel:
        return {"n": 0, "avg": None, "wr": None, "comp": None}
    rets = [float(t["ret_pct"]) for t in sel]
    comp = 1.0
    for r in rets:
        comp *= (1 + r / 100)
    return {"n": len(sel), "avg": round(sum(rets) / len(rets), 3),
            "wr": round(100 * sum(r > 0 for r in rets) / len(rets), 1),
            "comp": round((comp - 1) * 100, 1)}


def part_a() -> list:
    cfg0 = load_config()
    bt = json.loads((ROOT / "output" / "backtest_long.json").read_text())
    rows = [{"key": "BASE(全开)",
             **{f"tr_{k}": v for k, v in seg_stats(bt["trades_all"], False).items()},
             **{f"va_{k}": v for k, v in seg_stats(bt["trades_all"], True).items()},
             "mdd": float(bt["metrics"]["最大回撤%"]), "prep_s": 0, "sim_s": 0}]
    r = rows[0]
    print(f"[BASE] 训练段 comp={r['tr_comp']}% (n={r['tr_n']}) || "
          f"验证段 comp={r['va_comp']}% (n={r['va_n']}) || 全窗回撤{r['mdd']}%", flush=True)
    for key, desc, (blk, prm), val in VARIANTS:
        cfg = copy.deepcopy(cfg0)
        old = cfg[blk][prm]
        cfg[blk][prm] = val
        t0 = time.time()
        ctx = prepare_context(cfg, DAYS)
        p_s = round(time.time() - t0)
        t0 = time.time()
        res = run_backtest(cfg, DAYS, ctx)
        s_s = round(time.time() - t0)
        tr = res.get("trades_all") or []
        row = {"key": f"{key}({blk}.{prm}:{old}->{val}) [{desc}]",
               **{f"tr_{k}": v for k, v in seg_stats(tr, False).items()},
               **{f"va_{k}": v for k, v in seg_stats(tr, True).items()},
               "mdd": float(res["metrics"]["最大回撤%"]), "prep_s": p_s, "sim_s": s_s}
        rows.append(row)
        print(f"[{key:10s}] 训练段 comp={row['tr_comp']}% (n={row['tr_n']}) || "
              f"验证段 comp={row['va_comp']}% (n={row['va_n']}) || "
              f"全窗回撤{row['mdd']}%  (prep {p_s}s+sim {s_s}s)", flush=True)
        del ctx, res
    base = rows[0]
    print("\n  -- 风控层贡献拆解(相对BASE差值) --", flush=True)
    for r in rows[1:]:
        print(f"  {r['key']}\n"
              f"    训练段: BASE {base['tr_comp']}% -> {r['tr_comp']}% "
              f"(风控贡献{r['tr_comp']-base['tr_comp']:+.1f}pp) | "
              f"验证段: BASE {base['va_comp']}% -> {r['va_comp']}% "
              f"(风控代价{base['va_comp']-r['va_comp']:+.1f}pp) | "
              f"全窗回撤 {base['mdd']}% -> {r['mdd']}%", flush=True)
    return rows


def part_b() -> dict:
    bt = json.loads((ROOT / "output" / "backtest_long.json").read_text())
    tr = [t for t in bt["trades_all"] if t.get("ret_pct") is not None]
    tr = sorted(tr, key=lambda t: str(t.get("sell_date", "")))   # 结算序(stable)
    cfg = load_config()
    mon = cfg.get("monitor") or {"window": 20}
    win = int(mon.get("window", 20))
    rets = [float(t["ret_pct"]) for t in tr]
    labels = [str(t.get("sell_date", "")) for t in tr]
    era_flag = [str(t.get("buy_date", "")) >= ERA for t in tr]

    states, changes = [], []
    prev = None
    for i in range(len(rets)):
        lv = compute_monitor(rets[: i + 1], mon)["level"]
        states.append(lv)
        if lv != prev:
            changes.append((i, labels[i], lv, era_flag[i]))
            prev = lv
    short = {"绿-正常": "绿", "黄-观察带": "黄", "红-失效警报": "红",
             "深红-建议停机复检": "深红", "样本不足": "样本不足"}
    out = {"n": len(rets), "window": win, "changes": [], "era": {}}
    print(f"\n-- 检验B: 监测器十年回放(结算序{len(rets)}笔, 窗口{win}) --", flush=True)
    print("  状态跳变时间线(全量):", flush=True)
    for i, d, lv, ea in changes:
        tag = "era后" if ea else "era前"
        print(f"    第{i+1:3d}笔 sell={d} [{tag}] -> {short.get(lv, lv)}", flush=True)
        out["changes"].append({"i": i + 1, "sell_date": d, "level": lv,
                               "era_after": ea})
    for name, mask in (("era前", [not e for e in era_flag]),
                       ("era后", era_flag)):
        seg = [s for s, m in zip(states, mask) if m]
        dist = {}
        for s in seg:
            dist[short.get(s, s)] = dist.get(short.get(s, s), 0) + 1
        out["era"][name] = {k: round(v / len(seg) * 100, 1) for k, v in dist.items()}
        print(f"  {name}({len(seg)}笔结算时点)状态占比: {out['era'][name]}", flush=True)
    # 首次warn/halt延迟(era前第一笔起)
    first_warn = next((c for c in changes if "红" in c[2] and "深" not in c[2]), None)
    first_halt = next((c for c in changes if "深红" in c[2]), None)
    for tag, c in (("首次warn", first_warn), ("首次halt", first_halt)):
        if c:
            print(f"  era前{tag}: 第{c[0]+1}笔(sell={c[1]}), 距首笔延迟{c[0]}笔", flush=True)
            out[f"first_{tag[-4:]}"] = {"i": c[0] + 1, "sell_date": c[1], "lag_bars": c[0]}
    # 反事实: era前halt首次触发即停机 -> 剩余era前已执行笔的合计
    if first_halt:
        h_i = first_halt[0]
        rest_idx = [i for i in range(len(rets)) if not era_flag[i] and i > h_i]
        rest_rets = [rets[i] for i in rest_idx]
        comp_rest = 1.0
        for r in rest_rets:
            comp_rest *= (1 + r / 100)
        avoided = (1 - comp_rest) * 100   # 停机避免的进一步亏损(占总净值比)
        print(f"  反事实(描述性): era前halt首触发(第{h_i+1}笔)即停机, "
              f"可避开其后{len(rest_idx)}笔、进一步净值损耗{abs(avoided):.1f}%", flush=True)
        out["counterfactual_halt"] = {"halt_i": h_i + 1, "avoided_trades": len(rest_idx),
                                      "avoided_loss_pct": round(abs(avoided), 1)}
    return out


def main() -> None:
    print("== 防线审计(第四十五轮, 事前注册, 零参数变更) ==", flush=True)
    print("-- 检验A: 风控层十年贡献(开/关对比) --", flush=True)
    rows_a = part_a()
    out_b = part_b()
    OUT.write_text(json.dumps({"meta": {"era": ERA, "days": DAYS,
                                        "note": "事前注册诊断, 零参数变更"},
                               "part_a_regime": rows_a, "part_b_monitor": out_b},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n== 完成, 落盘 {OUT} ==", flush=True)


if __name__ == "__main__":
    main()
