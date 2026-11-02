"""可选规则留一消融(leave-one-out ablation) —— 第一百二十七轮。

为什么需要这个工具
------------------
参数扫描回答不了"这条规则到底有没有用": 同一参数在不同窗口会给出相反的
最优值(第一百一十二轮 12 组全证伪, 第一百二十一轮 Kendall W=0.058/0.008)。
消融换个问法: **把这条规则关掉, 结果变不变?** 三种可能各自对应明确结论 ——
  · 逐位相同        -> 该键在当前口径下空转(从未触发 / 被其它分支短路)
  · 变差            -> 规则有效, 但要看"影响了多少笔"才能判断是否可信
  · 变好            -> 规则是负贡献, 应当删除(这才叫真优化)

关键读法: **Δ收益必须和 Δ笔数一起看**。1500 日窗口末端净值已被放大几十倍,
末期 1 笔交易的微小差异可以放大成几百 pp —— 单看 Δ收益 会把"路径噪声"
误读成"规则很重要"。判定标准见 policy.yaml 与 CHANGELOG 第一百一十二轮:
单吊路径敏感, |Δ| < 500pp 一律视为噪声; 影响笔数 <= 2 的规则不得据以调参。

用法
----
  python3 tools/exp_ablate.py            # 全部变体
  python3 tools/exp_ablate.py --days 500 # 换窗口(结论可能与 1500 日不同)
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


def setv(**kv):
    """改值(不删键)。注意 skip_open_gain 在代码里是 ex["skip_open_gain"] 直接索引,
    删键会 KeyError, 只能改成 999 这类"等效关闭"的值。"""
    def f(cfg):
        cfg["execution"].update(kv)
    return f


def drop(*keys):
    def f(cfg):
        for k in keys:
            cfg["execution"].pop(k, None)
    return f


VARIANTS = [
    ("base", None),
    # 开盘浮盈 >= 15% 时开盘一次性清仓(只为验证"高浮盈兑现"是否真的赚钱)
    ("no_sfg", setv(sell_full_if_open_gain=999)),
    # T+1 高开 >= 10% 跳过买入(第九十二轮口径)
    ("no_sog", setv(skip_open_gain=999)),
    ("sog_7", setv(skip_open_gain=7)),
    # sell_open_frac=0 意味着全部按收盘卖, 这两条"延卖至收盘"的规则疑似被短路
    ("no_scd", setv(sell_close_if_open_drop=0, sell_limit_down_fallback=False)),
    # score_tiers 两档乘数均为 1.00, 疑似等价于无分档
    ("no_tiers", drop("score_tiers")),
]

FIELDS = ["总收益率%", "最大回撤%", "卡玛比率", "交易次数", "胜率%", "平均单笔收益%"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=None, help="窗口交易日数(默认取 config)")
    ap.add_argument("--out", default="", help="结果 JSON 输出路径")
    args = ap.parse_args()

    base = load_config()
    days = args.days or int(base["backtest"]["trading_days"])
    t0 = time.time()
    print(f"构建上下文(窗口 {days} 日)...", flush=True)
    ctx = prepare_context(base, days)
    print(f"上下文就绪 {time.time() - t0:.0f}s(后续各变体复用, 不重复拉数据)\n", flush=True)

    rows = []
    for name, mut in VARIANTS:
        cfg = copy.deepcopy(base)
        if mut:
            mut(cfg)
        t = time.time()
        try:
            res = run_backtest(cfg, days, ctx)
        except Exception as e:  # noqa: BLE001 单变体失败不中断整轮
            print(f"{name:9s} FAIL {e!r}", flush=True)
            rows.append({"name": name, "error": repr(e)[:200]})
            continue
        m = res.get("metrics") or {}
        row = {"name": name, **{k: m.get(k) for k in FIELDS}, "sec": round(time.time() - t)}
        rows.append(row)
        print("  " + json.dumps(row, ensure_ascii=False), flush=True)

    b = next((r for r in rows if r["name"] == "base" and "error" not in r), None)
    if b:
        print(f"\n=== 相对基线(窗口 {days} 日) ===", flush=True)
        print(f"  {'变体':9s} {'Δ收益pp':>10s} {'Δ卡玛':>8s} {'Δ笔数':>6s}  判定", flush=True)
        for r in rows:
            if r is b or "error" in r:
                continue
            dr = (r["总收益率%"] or 0) - (b["总收益率%"] or 0)
            dk = (r["卡玛比率"] or 0) - (b["卡玛比率"] or 0)
            dn = (r["交易次数"] or 0) - (b["交易次数"] or 0)
            if abs(dr) < 1e-9 and dn == 0:
                verdict = "空转(改不改都一样)"
            elif abs(dr) < 500:
                verdict = "在噪声带内(<500pp)"
            elif dn and abs(dn) <= 2:
                verdict = f"仅影响 {abs(dn)} 笔, 数值不可信"
            else:
                verdict = "显著"
            print(f"  {r['name']:9s} {dr:+10.2f} {dk:+8.3f} {dn:+6d}  {verdict}", flush=True)

    out = args.out or "/tmp/ablate.json"
    json.dump(rows, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n结果已写 {out}", flush=True)


if __name__ == "__main__":
    main()
