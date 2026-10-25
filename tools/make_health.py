# -*- coding: utf-8 -*-
"""生成 output/health.json —— 网页"策略体检"区数据(可重复运行刷新)。

探针口径: 复用 src.monitor.compute_monitor 的滚动终态(第三十九轮注册阈值),
与第四十六轮 edge_profile 的"滚动20笔均值"完全一致。

数据源(第四十八轮起双源择新, 解决云端日更):
  1) output/backtest_long.json 的 trades_all —— 十年回测(2500日), 本地权威口径;
  2) output/backtest.json 的 trades_all(存在时)或 trades_executed_recent
     —— 每日云端 workflow 刷新的500日回测(已结算明细, 上限200笔)。
  取"最后一笔结算日"更新者为数据源: 云端没有十年重跑, 靠每日500日回测
  保持体检面板新鲜; 其末20笔已结算与十年口径完全一致(同一策略同一数据),
  仅era最前端季度可能差个别早期成交(窗口起点略晚于2024-09)。

季度剖面与滚动峰值: 自所选成交序列现算(era过滤 buy_date>=20240901,
按buy_date季度分桶; 滚动20笔均值轨迹取最大), 不再读 edge_profile.json ——
该文件保留为第46轮证据产物。

用法: python3 tools/make_health.py
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.monitor import compute_monitor  # noqa: E402

ERA = "20240901"   # era切分锚点, 与第42~46轮一致
ERA_VERDICT = ("edge依赖2024-09后市场微观结构: 2016-2023八年连亏(-91%), "
               "2024-09后128笔+3.14%/笔; 参数经12组走前反证不可调(第44轮); "
               "风控era后贡献+1183pp、趋势条件为主力(第45/46轮); "
               "监测器era前90%时间告警/era后85%正常(第45轮)")
RECHECK = [
    {"name": "苗头④ 基差5日斜率", "state": "t=1.64, 观察级"},
    {"name": "苗头③ 营收同比[-10,0)", "state": "n=55, 十年复核优势收窄"},
    {"name": "苗头② 同花顺热度冷门桶", "state": "n=8, 观察级"},
    {"name": "苗头① 龙虎榜T-5上榜", "state": "n=2, 观察级"},
]
SOURCES = ("来源: 第43轮(edge稳健性)/44(参数反证)/45(防线审计)/46(退潮探针), "
           "详见仓库 CHANGELOG.md; 均为事前注册诊断, 零参数变更; "
           "探针/季度剖面随每日回测自动刷新(第48轮)")


def _settled(trades: list) -> list:
    """已结算成交, 按结算日序(旧→新)。float32面板下ret_pct为np.float32需转float。"""
    return sorted((t for t in trades if t.get("ret_pct") is not None),
                  key=lambda t: t.get("sell_date", ""))


def _pick_source() -> tuple[list, str]:
    """双源择新: 十年文件 vs 每日500日回测, 取最后一笔结算日更新者(平局取十年)。"""
    cands: list[tuple[str, list]] = []
    long_path = ROOT / "output" / "backtest_long.json"
    if long_path.exists():
        bt = json.loads(long_path.read_text())
        if bt.get("trades_all"):
            cands.append(("十年回测(2500日)", _settled(bt["trades_all"])))
    daily_path = ROOT / "output" / "backtest.json"
    if daily_path.exists():
        bt = json.loads(daily_path.read_text())
        tr = bt.get("trades_all") or bt.get("trades_executed_recent") or []
        if tr:
            cands.append(("日更回测(500日)", _settled(tr)))
    if not cands:
        return [], ""
    best = max(cands, key=lambda c: c[1][-1].get("sell_date", ""))
    return best[1], best[0]


def _quarters(settled: list) -> list:
    """era后成交按buy_date季度分桶(与第46轮edge_profile口径一致)。"""
    era = sorted((t for t in settled if t.get("buy_date", "") >= ERA),
                 key=lambda t: (t.get("buy_date", ""), t.get("sell_date", "")))
    buckets: dict[str, list] = {}
    for t in era:
        d = t["buy_date"]
        buckets.setdefault(f"{d[:4]}Q{(int(d[4:6]) - 1) // 3 + 1}", []).append(float(t["ret_pct"]))
    out = []
    for q in sorted(buckets):
        rs = buckets[q]
        comp = 1.0
        for r in rs:
            comp *= (1 + r / 100)
        out.append({"q": q, "n": len(rs),
                    "avg": round(sum(rs) / len(rs), 3),
                    "wr": round(100 * sum(r > 0 for r in rs) / len(rs), 1),
                    "comp": round((comp - 1) * 100, 1)})
    return out


def _rolling_peak(settled: list, window: int) -> tuple[float | None, str]:
    """滚动window笔均值轨迹的峰值(与第46轮rolling20口径一致, 日期=窗口末笔结算日)。"""
    rets = [float(t["ret_pct"]) for t in settled]
    best, best_d = None, ""
    for i in range(window - 1, len(rets)):
        m = sum(rets[i - window + 1:i + 1]) / window
        if best is None or m > best:
            best, best_d = m, settled[i].get("sell_date", "")
    return (round(best, 2) if best is not None else None), best_d


def main() -> None:
    cfg = load_config()
    mon = cfg.get("monitor") or {"window": 20}
    probe, quarters = {}, []
    settled, source = _pick_source()
    if settled:
        m = compute_monitor([float(t["ret_pct"]) for t in settled], mon)   # 滚动终态
        peak, peak_d = _rolling_peak(settled, m["window"])
        probe = {"level": m["level"], "mean_pct": m["mean_pct"],
                 "win_pct": m["win_pct"], "n": m["n"], "window": m["window"],
                 "asof": settled[-1].get("sell_date", ""),
                 "source": source, "source_n": len(settled),
                 "peak_pct": peak, "peak_date": peak_d,
                 "note": ("回测口径探针(与实盘监测灯互补; 单季波动属常态, "
                          "实盘滚动均值<0%→红、<-1%→深红按注册行为处置)")}
        quarters = _quarters(settled)

    health = {"updated": date.today().isoformat(), "probe": probe,
              "quarters": quarters, "era": ERA_VERDICT,
              "recheck": RECHECK, "sources": SOURCES}
    out = ROOT / "output" / "health.json"
    out.write_text(json.dumps(health, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"health.json 已生成 -> {out}  (数据源: {source or '无(仅静态结论)'}, "
          f"已结算{len(settled)}笔)")
    print(json.dumps(probe, ensure_ascii=False))


if __name__ == "__main__":
    main()
