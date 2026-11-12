"""第 207 轮: pbd5(挂单窗口 3->5 日)的分年分解 —— 判断它是不是"某一年特别好的产物".

全史总倍数 +24.8% 已超出 ab_test 的 ±3% 噪声护栏, 但 1500 日只 +4.2%(在护栏内)。
两个窗口方向一致、幅度差 6 倍, 必须看年度: 若增量集中在 2015/2021 这种单边年,
就不能采纳(第一百二十轮 bias 11.6 的教训: 全样本最优 ≠ 每年都好)。

输出 output/sat_207_year.json 与终端表格。
"""
from __future__ import annotations

import copy
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context                      # noqa: E402
from src.config import load_config                            # noqa: E402
from src.satellite import entry_ok_series                     # noqa: E402
from tools.satellite_backtest import run                      # noqa: E402

OUT = ROOT / "output"
os.environ["BT_FULL_TRADES"] = "1"

VARIANTS = {"base": {}, "pbd5": {"satellite": {"pullback_days": 5}}}


def merge(base: dict, ov: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in ov.items():
        out[k] = merge(out.get(k) or {}, v) if isinstance(v, dict) else v
    return out


def main() -> None:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 3645
    cfg = load_config()
    ctx = prepare_context(cfg, days)
    out = {}
    for name, ov in VARIANTS.items():
        c = merge(cfg, ov)
        r = run(ctx, c, entry_ok=entry_ok_series(ctx.get("reg"), c))
        tr = r.get("trades") or []
        by_year = defaultdict(list)
        for t in tr:
            y = str(t["exit_date"])[:4]
            by_year[y].append(float(t["ret_pct"]))
        rows = {}
        for y, v in sorted(by_year.items()):
            # 年度复利(该年内依次满仓单吊的乘积) —— 与引擎路径不完全等价, 只用于"增量来自哪年"
            comp = 1.0
            for x in v:
                comp *= (1 + x / 100.0)
            rows[y] = {"n": len(v), "mean": round(sum(v) / len(v), 3),
                       "comp_pct": round((comp - 1) * 100, 2)}
        out[name] = {"metrics": {k: float(r["metrics"].get(k, 0)) for k in
                                 ("总收益率%", "年化收益率%", "最大回撤%", "卡玛比率", "交易次数")},
                     "year": rows}
        print(f"\n{name}: 总 {out[name]['metrics']['总收益率%']:.1f}%  "
              f"笔 {len(tr)}")
        for y, v in rows.items():
            print(f"  {y}  n={v['n']:>3}  单笔 {v['mean']:+7.3f}%  年内复利 {v['comp_pct']:+8.2f}%")

    # 逐年对比
    yb, y5 = out["base"]["year"], out["pbd5"]["year"]
    years = sorted(set(yb) | set(y5))
    print("\n逐年对比(单笔均值):")
    win = 0
    for y in years:
        b = yb.get(y, {}).get("mean")
        s = y5.get(y, {}).get("mean")
        if b is None or s is None:
            print(f"  {y}  base {b}  pbd5 {s}  (缺失)")
            continue
        d = s - b
        win += 1 if d > 0 else 0
        print(f"  {y}  base {b:+7.3f}%  pbd5 {s:+7.3f}%  差 {d:+7.3f}pp  "
              f"n {yb[y]['n']}->{y5[y]['n']}")
    print(f"\n  pbd5 单笔更优的年份: {win}/{len(years)}")
    (OUT / f"sat_207_year_{days}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
