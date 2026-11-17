"""第250轮 B: 按 daily_run 的组装代码, 在真实面板上产出 latest.fillin 并落盘。

不复用 scan_250a 的中间结果(它不保存), 这里重算一遍但只跑当日一天。
产出 /tmp/site250/data/latest.json 供页面截图验证。
"""

from __future__ import annotations

import glob
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import fillin                      # noqa: E402
from src.config import load_config          # noqa: E402
from src.selector import build_indicators, split_by_date   # noqa: E402

WARM = 620


def main() -> None:
    cfg = load_config()
    fp = fillin.params(cfg)
    days = sorted(Path(f).name.replace(".csv.gz", "")
                  for f in glob.glob(str(ROOT / "data" / "panel" / "*.csv.gz")))[-WARM:]

    t0 = time.time()
    panel = pd.concat([pd.read_csv(ROOT / "data" / "panel" / f"{d}.csv.gz") for d in days],
                      ignore_index=True)
    fill_idx = fillin.market_dist(panel)          # 清洗前
    panel = build_indicators(panel)
    panel = fillin.add_indicators(panel)
    by_date = split_by_date(panel)
    print(f"面板 {len(panel):,} 行 {time.time()-t0:.1f}s  by_date keys 示例 {list(by_date)[-1]!r}")

    signal_date = str(days[-1])
    day = by_date[signal_date]
    print(f"signal_date={signal_date!r}  当日 {len(day)} 只")

    # ---- 以下是 src/daily_run.py 里 fillin 块的逐行复制 ----
    names = dict(zip(day["ts_code"].astype(str), day["ts_code"].astype(str)))
    _idxd = fillin.idx_dist_at(fill_idx, signal_date)
    t1 = time.time()
    _pl = fillin.plan(day, _idxd, fp)
    print(f"fillin.plan {time.time()-t1:.2f}s  idx_dist={_idxd:.2f}")
    if _pl is None:
        print("❌ None")
        return

    _idx = day.set_index(day["ts_code"].astype(str))
    _cands = []
    for c in _pl["cands"]:
        _fin = c["code"] in _idx.index
        _r = _idx.loc[c["code"]] if _fin else None
        if isinstance(_r, pd.DataFrame):
            _r = _r.iloc[0]
        _cands.append({
            "code": c["code"],
            "name": names.get(c["code"], ""),
            "close": round(float(_r["close"]), 2) if _fin else None,
            "pct": round(float(_r["pct"]), 2) if _fin else None,
            "dh60": round(c["dh60"], 1) if c["dh60"] == c["dh60"] else None,
            "rank": c["rk"],
        })
    fill_info = {
        "enabled": True, "path": _pl["path"],
        "idx_dist": round(float(_idxd), 2) if _idxd == _idxd else None,
        "broke": _pl["path"] == "破位",
        "pos": fillin.pos_of(_pl["path"], fp),
        "cands": _cands,
        "let_way": False,
        "blocked": True,        # 假设今天主策略有票
        "note": ("破位成立 → 明早从上往下，第一只低开的买入"
                 if _pl["path"] == "破位" else
                 f"破位不成立 → 只做深低开：前 {fp['deep_rank_max']} 名里谁低开在 "
                 f"{fp['deep_lo']}%~{fp['deep_hi']}% 就买谁"),
    }
    print(json.dumps({k: v for k, v in fill_info.items() if k != "cands"}, ensure_ascii=False))
    for c in _cands:
        print(f"  {c['rank']:>2} {c['code']} 收 {c['close']} 涨跌 {c['pct']}%  dh60 {c['dh60']}%")

    # ---- 写一份页面可直接吃的 latest.json ----
    base = json.loads((ROOT / "output" / "latest.json").read_text(encoding="utf-8"))
    base["signal_date"] = f"{signal_date[:4]}-{signal_date[4:6]}-{signal_date[6:]}"
    base["fillin"] = fill_info
    out = Path("/tmp/site250/data/latest.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(base, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"已写 {out}")


if __name__ == "__main__":
    main()
