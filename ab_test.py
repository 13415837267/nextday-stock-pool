# -*- coding: utf-8 -*-
"""A/B 回测框架: 复用 prepare_context 的指标面板, 对多个参数变体快速回测对比。

用法:
  BT_VARIANTS='[{...}, ...]' BT_DAYS=1500 python3 ab_test.py
  BT_DAYS 支持逗号分隔多窗口, 如 "1500,3378"(双窗口验证, 每个窗口各建一次上下文)。

变体字段:
  name   (str)  展示名
  path   (str)  点路径, 如 "regime.market_ma"
  value  (any)  新值
  set    (dict) 直接深度合并到 config(用于一次性改多个键)
"""
from __future__ import annotations

import copy
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import load_config, ROOT
from src.backtest import prepare_context, run_backtest

BASE_CONFIG = os.environ.get("BT_CONFIG", str(ROOT / "config.yaml"))
BT_DAYS_RAW = os.environ.get("BT_DAYS", "1500")
BT_VARIANTS = json.loads(os.environ.get("BT_VARIANTS", "[]"))


def _set_path(obj: dict, path: str, value):
    keys = path.split(".")
    cur = obj
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value


def _deep_merge(dst: dict, src: dict) -> dict:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v
    return dst


def apply_variant(cfg: dict, var: dict) -> dict:
    cfg = copy.deepcopy(cfg)
    if "path" in var and "value" in var:
        _set_path(cfg, var["path"], var["value"])
    if "set" in var:
        _deep_merge(cfg, var["set"])
    return cfg


KEYS = ["总收益率%", "最大回撤%", "卡玛比率", "夏普比率", "交易次数", "胜率%", "平均单笔收益%", "盈亏比"]


def _fmt(m: dict) -> str:
    return " | ".join(f"{k}={m.get(k)}" for k in KEYS)


def main():
    base_cfg = load_config(BASE_CONFIG)
    days_list = [int(x) for x in BT_DAYS_RAW.split(",") if x.strip()]

    for days in days_list:
        t0 = time.time()
        print(f"\n{'='*90}\n窗口 {days} 日: 构建上下文...")
        ctx = prepare_context(base_cfg, days)
        print(f"上下文就绪, 耗时 {time.time()-t0:.0f}s")

        rows = []
        base_res = run_backtest(base_cfg, days, ctx)
        rows.append(("baseline", base_res["metrics"]))
        print(f"\n[baseline] {_fmt(base_res['metrics'])}")

        for var in BT_VARIANTS:
            cfg = apply_variant(base_cfg, var)
            res = run_backtest(cfg, days, ctx)
            rows.append((var.get("name", "?"), res["metrics"]))
            print(f"[{var.get('name', '?')}] {_fmt(res['metrics'])}", flush=True)

        base_m = rows[0][1]
        print(f"\n--- 窗口 {days} 日汇总 ---")
        for name, m in rows:
            d_ret = m["总收益率%"] - base_m["总收益率%"]
            rel = d_ret / abs(base_m["总收益率%"]) * 100 if base_m["总收益率%"] else 0.0
            d_cal = (m.get("卡玛比率") or 0) - (base_m.get("卡玛比率") or 0)
            if name == "baseline":
                tag = ""
            elif abs(rel) < 3.0 and abs(d_cal) < 0.15:
                # 第 121 轮：5 档参数的全史收益跨度仅 1.68 倍，年度差异达 40 倍，
                # 3% 以内的相对变化完全落在噪声内，不应据此改 config。
                tag = "  ≈噪声"
            elif d_ret < 0:
                tag = "  ↓"
            else:
                tag = "  ↑"
            print(f"{name:24s} 收益={m['总收益率%']:>10.2f}%  回撤={m['最大回撤%']:>8.2f}%  "
                  f"卡玛={m['卡玛比率']:>5.2f}  笔数={m['交易次数']:>5}  胜率={m['胜率%']:>5.2f}%  "
                  f"单笔={m['平均单笔收益%']:>6.3f}%  Δ收益={d_ret:>+9.2f}pp({rel:>+5.1f}%){tag}")

        tiny = [n for n, m in rows[1:]
                if abs((m["总收益率%"] - base_m["总收益率%"]) / abs(base_m["总收益率%"] or 1) * 100) < 3.0
                and abs((m.get("卡玛比率") or 0) - (base_m.get("卡玛比率") or 0)) < 0.15]
        if tiny:
            print(f"提示: {len(tiny)}/{len(rows) - 1} 个变体落在噪声带（±3% 收益、±0.15 卡玛），"
                  f"不可作为改参依据；参数是否可辨识请用 tools/param_rank_stability.py。")


if __name__ == "__main__":
    main()
