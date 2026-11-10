# -*- coding: utf-8 -*-
"""第185轮·D: 池宽度是"窄丘"还是"针尖" —— 副策略 8.27 倍的可信度体检

为什么必须做
------------
池宽度扫描(ivol_pct_max)给出一个极陡的峰, 峰正好落在现役的 0.10 上:

    iv02 0.73x | iv03 0.84x | iv05 1.00x | **iv10 8.27x** | iv15 4.16x | iv20 2.37x | iv30 1.34x

左右两侧各跌一半, 收益差 8 倍。这有两种完全不同的解释:
  A. 现役参数恰好在真 alpha 的甜点上(好)
  B. 8.27 倍是 2021 年某几只票的运气, 参数稍微一动就没了(坏)

区分方法: 看**分年度**。若 A, iv10 应该年年优于 iv05/iv15; 若 B, 差距应集中在某一年。
产物: output/exp_sat_pool_year.json
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context  # noqa: E402
from src.config import load_config  # noqa: E402
from src.satellite import entry_ok_series  # noqa: E402
import tools.satellite_backtest as sb  # noqa: E402

BASE = {"mode": "lowvol", "max_pos": 1, "buys_per_day": 1, "hold_days": 10,
        "sort_by": "mom20_ivol", "min_score": 0,
        "pullback_pct": 1.0, "pullback_days": 3, "pullback_mode": "low",
        "market_gate": True, "gate_entry_buffer": 0.3,
        "gate_exit": True, "gate_exit_mode": "all",
        "trail_pct": 8.0, "stop_pct": -12.0, "min_pool": 20,
        "exit_limit_pct": 0.0, "leverage": 1.0, "margin_rate": 0.055}


def yearly(res: dict) -> dict:
    out = {}
    for pt in res["equity"]:
        y = pt["d"][:4]
        out.setdefault(y, [pt["nav"], pt["nav"]])
        out[y][1] = pt["nav"]
    return {y: round((v[1] / v[0] - 1) * 100, 1) for y, v in out.items()}


def main():
    cfg = load_config()
    ctx = prepare_context(cfg, 1500)
    print(f"===== {len(ctx['main_dates'])} 日 =====", flush=True)
    out = {"days": len(ctx["main_dates"]), "variants": {}}
    for iv in (0.05, 0.10, 0.15):
        c = copy.deepcopy(cfg)
        c.setdefault("satellite", {}).update({**BASE, "ivol_pct_max": iv})
        res = sb.run(dict(ctx), c, entry_ok=entry_ok_series(ctx.get("reg"), c))
        m, y = res["metrics"], yearly(res)
        nm = f"iv{int(iv*100):02d}"
        out["variants"][nm] = {"倍数": round(1 + m["总收益率%"] / 100, 2),
                               "年化%": m["年化收益率%"], "回撤%": m["最大回撤%"],
                               "笔数": m["交易次数"], "单笔%": m["平均单笔收益%"],
                               "年度": y}
        print(f"{nm} {m['总收益率%']:>10.0f}% 笔{m['交易次数']:4d} 单笔{m['平均单笔收益%']:+.2f}% "
              f"年度 {y}", flush=True)
    p = ROOT / "output/exp_sat_pool_year.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"写出 {p}", flush=True)


if __name__ == "__main__":
    main()
