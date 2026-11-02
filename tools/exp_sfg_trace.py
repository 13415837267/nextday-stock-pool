"""第一百二十七轮(续2): 定位"开盘浮盈15%全卖"实际命中的那笔交易。

消融显示关掉 sell_full_if_open_gain 会让 1500 日收益少 775pp、且少 1 笔,
但不知道具体是哪一天、哪只票、赚了多少。本脚本把 base 与关闭版逐笔对齐,
打印差异票的完整时间线与收益, 用于判断这条规则是"真实规律"还是"单笔运气"。
"""
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context, run_backtest  # noqa: E402
from src.config import load_config  # noqa: E402

DAYS = 1500


def main():
    base = load_config()
    ctx = prepare_context(base, DAYS)

    r_on = run_backtest(copy.deepcopy(base), DAYS, ctx)

    cfg_off = copy.deepcopy(base)
    cfg_off["execution"]["sell_full_if_open_gain"] = 999
    r_off = run_backtest(cfg_off, DAYS, ctx)

    t_on = r_on.get("trades") or r_on.get("trades_all") or []
    t_off = r_off.get("trades") or r_off.get("trades_all") or []
    print(f"成交笔数 on={len(t_on)} off={len(t_off)}")

    hit = [t for t in t_on if "全卖" in (t.get("note") or "")]
    print(f"\n=== base 中触发'开盘浮盈全卖'的 {len(hit)} 笔 ===")
    for t in hit:
        print(json.dumps({k: t.get(k) for k in
                          ["signal_date", "code", "name", "buy_date", "buy_px",
                           "sell_date", "sell_px", "sell_px_open", "sell_px_close",
                           "sell_frac_open", "ret_pct", "score", "gap_pct", "note"]},
                         ensure_ascii=False))

    # 对齐: 用 (signal_date, code) 作键
    key = lambda t: (t.get("signal_date"), t.get("code"))
    on_map = {key(t): t for t in t_on}
    off_map = {key(t): t for t in t_off}
    only_on = [k for k in on_map if k not in off_map]
    only_off = [k for k in off_map if k not in on_map]
    print(f"\n仅 base 有的票 {len(only_on)}: {only_on}")
    print(f"仅 关闭版 有的票 {len(only_off)}: {only_off}")

    print("\n=== 差异票在两版下的对比 ===")
    for k in only_on + only_off:
        a, b = on_map.get(k), off_map.get(k)
        print(f"  {k}")
        for tag, t in (("base", a), ("off ", b)):
            if t:
                print(f"    {tag}: 买{t.get('buy_date')}@{t.get('buy_px')} 卖{t.get('sell_date')}"
                      f"@{t.get('sell_px')} 收益{t.get('ret_pct')}% note={t.get('note')}")

    # 同键但收益不同(卖出时点被规则改变)
    diff = [(k, on_map[k], off_map[k]) for k in on_map if k in off_map
            and abs((on_map[k].get("ret_pct") or 0) - (off_map[k].get("ret_pct") or 0)) > 1e-6]
    print(f"\n=== 同票但收益不同 {len(diff)} 笔 ===")
    for k, a, b in diff:
        print(f"  {k}: base {a['ret_pct']}% (note={a.get('note')}) vs off {b['ret_pct']}% (note={b.get('note')})")


if __name__ == "__main__":
    main()
