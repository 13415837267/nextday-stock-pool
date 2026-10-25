# -*- coding: utf-8 -*-
"""T-5上榜基因 满桶重验(第五十七轮, 事前注册纯诊断, 零参数变更)。

背景: 第三十一轮发现"信号票近期上过龙虎榜=情绪退潮, 深低开承接更差"(观察级,
n=3); 第三十八轮用日榜文件复核仍 n=2(当时宇宙受限于 backtest.json 仅存近期
成交)。第三十一轮注册待办①本为"榜单积累后对全量信号日做严格检验"——现
data/lhb/ 已回填 2025-05-19 起 349 个交易日 + backtest_long.json 已存十年全量
成交(442笔), 宇宙可从 12 笔扩到 era后全量(约86笔), 上榜桶有望首次达到注册
的"满桶10笔"重验线。

口径(沿用31/38轮注册定义, 不改):
  - 宇宙: backtest_long.json executed 且 signal_date >= 2025-05-26
    (保证窗口A的5个前置交易日榜单文件齐全)
  - 窗口A(严格信号时点前): 信号日前5个交易日榜单
  - 窗口B(实盘15:30可得): 含信号日当日, 前5+当日共6个交易日榜单
  - ret口径: trades_all 的 ret_pct(净口径, 含费用; 两桶同口径, 对比不受影响)
  - h1/h2 锚 20260306(第三十八轮注册锚点)
  - 判据: 上榜桶 n>=10 且方向(h1/h2)一致 才进入"是否接入"讨论; 否则维持观察级

输出: 打印 + output/lhb_t5_audit.json
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LHB_DIR = ROOT / "data" / "lhb"
H_SPLIT = "20260306"          # h1/h2 锚(第三十八轮注册)
UNIVERSE_FROM = "20250526"    # 宇宙起点(窗口A前置5个榜单交易日齐全)
T5 = 5                        # T-5 窗口(注册口径)


def _load_trades() -> list[dict]:
    bl = json.loads((ROOT / "output" / "backtest_long.json").read_text(encoding="utf-8"))
    ta = bl.get("trades_all") or []
    out = [t for t in ta
           if t.get("ret_pct") is not None and not t.get("skip")
           and str(t.get("signal_date", "")) >= UNIVERSE_FROM]
    out.sort(key=lambda t: str(t["signal_date"]))
    return out


def _load_calendar() -> list[str]:
    cal = sorted(p.stem.replace(".csv", "") for p in (ROOT / "data" / "panel").glob("*.csv.gz"))
    return [d for d in cal if d >= "20250519"]


def _lhb_set(day: str) -> set[str] | None:
    """单日榜单 ts_code 集合; 文件缺失返回 None(调用方计数)。"""
    p = LHB_DIR / f"{day}.csv.gz"
    if not p.exists():
        return None
    with gzip.open(p, "rt") as f:
        df = pd.read_csv(f, usecols=["ts_code"])
    return set(df["ts_code"].astype(str))


def _window_codes(cal: list[str], sig: str, include_sig: bool) -> tuple[set[str], int]:
    """信号日前T5个交易日(可选含当日)榜单并集; 返回(代码集, 缺失文件数)。"""
    try:
        pos = cal.index(sig)
    except ValueError:
        return set(), 0
    start = pos - T5
    if start < 0:
        return set(), 0
    days = cal[start:pos] if not include_sig else cal[start:pos + 1]
    codes: set[str] = set()
    missing = 0
    for d in days:
        s = _lhb_set(d)
        if s is None:
            missing += 1
        else:
            codes |= s
    return codes, missing


def _stats(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    rets = [float(t["ret_pct"]) for t in rows]
    comp = 1.0
    for r in rets:
        comp *= 1 + r / 100
    return {
        "n": len(rows),
        "avg_pct": round(sum(rets) / len(rets), 3),
        "median_pct": round(sorted(rets)[len(rets) // 2], 3),
        "win_pct": round(100.0 * sum(r > 0 for r in rets) / len(rets), 1),
        "comp_pct": round((comp - 1) * 100, 2),
    }


def main() -> None:
    trades = _load_trades()
    cal = _load_calendar()
    cal = [d for d in cal if d >= "20250519"]
    print(f"宇宙: {len(trades)}笔执行样本 (signal_date >= {UNIVERSE_FROM}, LHB覆盖{cal[0]}~{cal[-1]})")

    rows_a: list[dict] = []   # (trade, on_lhb)
    rows_b: list[dict] = []
    missing_files = 0
    for t in trades:
        sig = str(t["signal_date"])
        ca, m1 = _window_codes(cal, sig, include_sig=False)
        cb, m2 = _window_codes(cal, sig, include_sig=True)
        missing_files += m1 + m2
        rows_a.append({"t": t, "on": str(t["code"]) in ca})
        rows_b.append({"t": t, "on": str(t["code"]) in cb})
    if missing_files:
        print(f"⚠ 榜单文件缺失 {missing_files} 个(窗口日期), 结果可能偏乐观")

    result: dict = {
        "universe_n": len(trades),
        "universe": _stats([r["t"] for r in rows_a]),
        "missing_files": missing_files,
        "h_split": H_SPLIT,
        "windowA_strict": {},
        "windowB_with_sig": {},
    }
    for name, rows in (("windowA_strict", rows_a), ("windowB_with_sig", rows_b)):
        on = [r["t"] for r in rows if r["on"]]
        off = [r["t"] for r in rows if not r["on"]]
        seg = {}
        for tag, pred in (("h1", lambda t: str(t["signal_date"]) < H_SPLIT),
                          ("h2", lambda t: str(t["signal_date"]) >= H_SPLIT)):
            seg[tag] = {
                "on": _stats([t for t in on if pred(t)]),
                "off": _stats([t for t in off if pred(t)]),
            }
        result[name] = {"on": _stats(on), "off": _stats(off), "seg": seg}

    for name, label in (("windowA_strict", "窗口A(严格信号时点前)"), ("windowB_with_sig", "窗口B(含信号日, 实盘可得)")):
        w = result[name]
        print(f"\n== {label} ==")
        print(f"  上榜:   {w['on']}")
        print(f"  未上榜: {w['off']}")
        for tag in ("h1", "h2"):
            s = w["seg"][tag]
            print(f"  {tag}: 上榜 {s['on']} | 未上榜 {s['off']}")

    (ROOT / "output" / "lhb_t5_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print("\n已写出 output/lhb_t5_audit.json")


if __name__ == "__main__":
    main()
