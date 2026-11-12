# -*- coding: utf-8 -*-
"""第二百零三轮 · 唯一可能同时满足"百倍"与"不被强平"的路径: 回撤自适应杠杆。

为什么恒定杠杆的百倍是假的
--------------------------
两融维持担保比例 = 账户权益 / 融资负债。1.5x 满仓时初始是 150/50 = 300%,
跌到 130% 就被强平 —— 那对应权益 65、即净值回撤 -35%。
而恒定 1.5x 的全史回撤是 -64.7%: 这条曲线在实盘活不到一半就没了。
所以"1.5x 全史 60 倍"只是账面数字。

本脚本测的是: 杠杆跟着自己的回撤走 ——
  · 回撤 <= -cut  : 杠杆降到 1.0x(先活下来)
  · 回撤 >= -cut/2: 恢复原杠杆(再谈收益)
只看昨日收盘净值, 双阈值迟滞, 不含任何未来信息。

判据: 最大回撤必须 > -35%(强平线), 再看年化能不能保住 40%(全史百倍所需)。

用法: python tools/exp_levdd_203.py 3000
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from src.backtest import prepare_context          # noqa: E402
from src.config import load_config                # noqa: E402
from src.satellite import entry_ok_series         # noqa: E402
from satellite_backtest import run                # noqa: E402

OUT = ROOT / "output"

CASES = [
    ("base_1x",       1.0, None),
    ("lv15_cut20",    1.5, 0.20),
    ("lv15_cut25",    1.5, 0.25),
    ("lv15_cut15",    1.5, 0.15),
    ("lv20_cut20",    2.0, 0.20),
    ("lv20_cut25",    2.0, 0.25),
    ("lv20_cut30",    2.0, 0.30),
]


def main():
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    cfg0 = load_config()
    t0 = time.time()
    ctx = prepare_context(cfg0, days)
    print(f"装载 {len(ctx['main_dates'])} 日, {time.time()-t0:.0f}s", flush=True)
    f = OUT / f"exp_levdd_203_{days}.json"
    res = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}

    for name, lev, cut in CASES:
        cfg = copy.deepcopy(cfg0)
        sat = {**(cfg0.get("satellite") or {})}
        sat.update({"mode": "lowvol", "hold_days": 10, "max_pos": 1, "sort_by": "mom20_ivol",
                    "leverage": lev})
        cfg["satellite"] = sat
        r = run(ctx, cfg, entry_ok=entry_ok_series(ctx.get("reg"), cfg),
                lev_cut=cut, lev_floor=1.0)
        m = r["metrics"]
        row = {k: m.get(k) for k in ("总收益率%", "年化收益率%", "最大回撤%", "卡玛比率",
                                     "交易次数", "平均单笔收益%")}
        row.update({"lever": lev, "cut": cut})
        res[name] = row
        flag = "OK" if (row["最大回撤%"] > -35 and row["年化收益率%"] > 40) else ""
        print(f"{name:12s} lev {lev:.1f} cut {str(cut):>4s}  "
              f"总 {row['总收益率%']:>11.1f}%  年化 {row['年化收益率%']:6.2f}%  "
              f"回撤 {row['最大回撤%']:7.2f}%  卡玛 {row['卡玛比率']:5.2f}  {flag}", flush=True)
        f.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("写出", f)


if __name__ == "__main__":
    main()
