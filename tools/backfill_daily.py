"""补位信号（第 248 轮定稿：精确资金占用 + 重叠日让位）

★ 第 248 轮改了两处「资金调度」, 都不涉及信号本身:
  1) 占用口径从粗改精。旧版把主策略的 {signal, buy, sell} 三天全排, 但主策略是
     T+1 开盘买、T+2 收盘卖, sell 日收盘卖出后资金 T+1 开盘就能用 —— 而补位正好是
     T+1 开盘买。所以真正冲突的只有「补位的买入日落在主策略持仓区间内」,
     等价于「T 是主策略的 signal 日或 buy 日」。少排 sell 日, 多释放 27 天。
  2) 48 天两边都想要(主策略有信号、补位也有候选), 旧版一律让主策略。改成让补位:
     这 48 天主策略单笔 +0.52%, 补位 +2.84%; 其中深低开 41 天主策 +1.27% vs 补位 +4.01%,
     破位 7 天主策 -3.91% vs 补位 -4.06%(但补位只 30% 仓, 让位等于减仓避险)。
     执行含义: 这 48 天主策略停手, 只做补位。
  合计: 290 笔/5.41 → 317 笔/5.48(只改口径) → 375 笔/5.76(再加让位)。
  深低开样本 61→119 笔, t 值 1.69→2.70, 247 遗留的"不显著"问题被样本量解决。
  代价: 分年 5 胜 5 负(2018/2020/2023 改善, 2021/2022/2024 变差), 不是处处更好。

规则:
  1) 触发（满足其一就做, 两者都是 T 日收盘 / T+1 9:25 可知）:
     a. 全市场等权指数距 20 日均线 <= -3.51%  且  收盘距前 60 日最高 <= -24.46%
     b. 当天"最超跌的那只"(9 窗口排名第一) 低开在 -9.5% ~ -5% 之间
        ← 个股自己崩出缺口, 与大盘无关。上界 -9.5% 是为了排除"跌停开盘":
          那 19 笔里 11 笔是 gap <= -9.5%, 属跌停板博弈(买得到、卖不确定), 不是超跌反弹。
  2) 排序: 九个窗口 60/90/120/150/180/200/250/300/360 各自把当日候选按"距前 W 日最高
     最远"排名, 换成排名百分比(0~1), 九个求平均。平均越小 = 越多个窗口都认为它跌得透。
  3) 单票限次: 同一只票在整个历史上最多买 8 次（超出则换当天下一只）。
     ★ 第 246 轮把 243 的"冷却 20 日"换掉了: 冷却换票会把当天最好的票换掉, 卡玛掉 2.4;
       限次不换掉最好的票, 只是不再加, 卡玛还升一点(5.80 -> 5.93), 最长连买 20 -> 8。
  4) ★ 执行: T 日收盘只把候选排好序; T+1 集合竞价(9:25)看开盘价, 从排好的名单从上往下,
     第一只"开盘价 <= 昨收 x 1.0005"的就是今天的标的, 按开盘价买入, T+2 收盘卖出。
  5) 仓位 40%

★ 第 246 轮修掉的一个跨轮 bug（影响 243/244/245 全部结论）:
  build_features_234.py 是「按年分片」跑的, rolling 只看得到当年数据。于是每年 1 月
  的前 20~30 个交易日, dist_high60 / bias* / ret20 / vol20 全是 NaN。
  用 dist_high60 <= -24.46 建候选池, 每年 1 月恒为空 —— 245 的 198 笔里 1 月只有 2 笔。
  补上 1 月后是 232 笔, 而 1 月的破位日单笔只有 1.23%(其他月份 2.47%), 因为 1 月的下跌
  多由年报预告暴雷驱动, 是个股基本面事件, 不是流动性恐慌, 超跌反弹逻辑不成立。
  本脚本的 rolling 是在完整面板上算的, 没有这个问题。

★ 为什么是"T+1 开盘重选"而不是"T 日选好再祈祷低开"（第 245 轮）:
  244 轮的做法是 T 日选出 top1, T+1 看它是否低开, 高开就空仓 —— 结果 198 次里 82 次(41%)
  只能放弃。但实盘根本不是这么做的: T+1 9:25 开盘价是公开信息, 完全可以"在低开的票里
  重新挑一只"。同一批数据:
    244 (T日选/T+1放弃): 116 笔, 单笔 2.447%, 123.0% / -23.59% / 5.21
    245 (T+1 重选):      198 笔, 单笔 2.656%, 138.1% / -25.15% / 5.49
  被 244 放弃的 82 天, 245 做成后单笔 2.19%、胜率 65%、合计 +179pp —— 这 82 天不是没得做,
  只是不能买"全体最弱"的那只(它高开说明已经反弹), 要买"低开票里最弱"的那只。

★ 前情（第 244 轮）:
  ① 逐个窗口挑最优是过拟合。历史卡玛排名 -> 次年卡玛排名 的秩相关 = -0.158(9 年 3 正)。
     walk-forward 每年重选得 5.17, 而 45 个组合随机选的中位数就是 4.88 —— 是运气不是本事。
  ② 把 9 个窗口的排名平均, 等于取消"挑窗口"。单窗口中位 4.73 / 最好 5.13 / 集成 5.56;
     随机抽 2/3/5/7/9 个窗口中位 4.75/4.94/5.11/5.37/5.55 单调上升。
  ③ 候选池预过滤 n1_ok 是未来信息（提前知道明天低开）, 244 已剥离。

★ 已知局限:
  - 198 笔 / 11 年, 每年 18 笔。2020 年只有 12 次、单笔 0.21%, 基本没贡献;
    收益集中在 2022 和 2024。
  - 245 与 244 在"244 也成交的 116 天"里逐日配对: 平均 +0.542pp 但胜率只有 12%,
    说明是"少数几次大赢"撑起来的, 不是稳定小胜。
  - 加成交额 >= 0.5 亿 的流动性约束后: 130.8% / -24.82% / 5.27（单笔 2.32%）。

★ 已排除（不要再试）:
  - ETF 抄底: 破位日宽基 ETF 开->次日收 0.39%(胜率 57%), 股票 ETF 0.37%, 债券 0.06%,
    黄金 0.12%; 而个股单吊 2.45%。ETF 当日开->收 全是负的(-0.39%)。差 6 倍, 不用了。
  - 高开日买"最强"的票: 高开组内分五档, 最弱档 0.79% > 最强档 0.24%, 单调, 反着买更差。
  - 244 之前的: 放宽 idx_dist 到 -2.0 / 破位深度定仓 / 跌停家数定仓 / 每波限次 /
    连亏降仓 / 部分止盈止损 / T+1 收盘买（补位单笔掉到 0.059%, 收益 100% 来自低开买入）。

用法:
  python3 tools/backfill_daily.py            # 最新信号日 + 最近 250 日回放
  python3 tools/backfill_daily.py --days 480
  python3 tools/backfill_daily.py --json     # 只输出最新一日 JSON（供 daily_run 调用）
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data" / "panel"
OUT = ROOT / "output"

IDX_DIST_MAX = -5.0        # 等权指数距 20 日线（251: -3.51 -> -5.0）
DIST_HIGH_MAX = -42.0      # 距前 60 日最高（251: -35 -> -42）
WINS = (60, 90, 120, 150, 180, 200, 250, 300, 360)
MAX_WIN = 360
MAX_TIMES = 8              # 同一只票最多买几次
COOLDOWN_DAYS = 10         # 第251轮: 同一条腿成交后 N 个交易日不再做（0=不限）
POS_BREAK = 0.30           # 破位日仓位
POS_DEEP = 0.50            # 深低开日仓位（单笔期望是破位的 1.8 倍，按比例给）
DEEP_LO, DEEP_HI = -8.0, -5.0   # 个股自己崩的缺口区间（下界: 跌太深接不住）
DEEP_RANK_MAX = 3          # 深低开名单只数（252: 8 -> 3）
DEEP_SKIP = 3              # 252: 深低开从第 4 名起取（跳过距前高最远的 3 只）
MIN_LISTED_BARS = 250
N1_GAP_MAX = 0.05          # T+1 开盘 <= 昨收 x (1+0.05%) 才算低开
TOP_N = 3                  # 输出前 N 只排序候选（252: 10 -> 3）
PICK_BREAK = "lowest"      # 252: 破位在名单里挑开盘最低的（原: 排名第一只达标的）


def load_panel(dates: list[str]) -> pd.DataFrame:
    frames = []
    cols = ["ts_code", "trade_date", "open", "high", "low", "close", "pre_close",
            "pct", "amount", "factor", "turnover_rate", "is_paused", "is_st"]
    for d in dates:
        f = PANEL / f"{d}.csv.gz"
        if not f.exists():
            continue
        try:
            frames.append(pd.read_csv(f, usecols=cols, dtype={"ts_code": "str"}))
        except Exception:      # noqa: BLE001
            continue
    if not frames:
        raise SystemExit("没有可用的面板缓存 data/panel/")
    d = pd.concat(frames, ignore_index=True)
    del frames
    # ★ 内存: 1200 日面板约 900 万行, float64 + object 会吃 3GB 以上, 加上 9 个
    #   窗口的 dh 列必 OOM(实测 exit=137)。数值列一律降 float32, ts_code 驻留。
    for c in d.columns:
        if c != "ts_code":
            d[c] = pd.to_numeric(d[c], errors="coerce", downcast="float")
    d["ts_code"] = d["ts_code"].astype("str")
    return d


def load_main_signals() -> tuple[set[int], set[int]]:
    """主策略的 {signal 日} 与 {buy 日}。这两个集合 = 补位的真占用区间。

    补位是 T+1 开盘买; 主策略 buy 日 T+1 开盘时钱还没回来(sell 日收盘才卖), 所以
    T ∈ {signal, buy} 时补位做不了。T = sell 日时不冲突(资金当日回笼, 次日开盘可用)。
    """
    sig: set[int] = set()
    buy: set[int] = set()
    f = OUT / "w232_trades.csv"
    if f.exists():
        t = pd.read_csv(f, usecols=["signal", "buy"])
        sig |= set(int(x) for x in t.signal)
        buy |= set(int(x) for x in t.buy)
    lj = ROOT / "site" / "data" / "latest.json"
    if lj.exists():
        try:
            j = json.loads(lj.read_text(encoding="utf-8"))
            sd = str(j.get("signal_date") or "").replace("-", "")
            bd = str((j.get("plan") or {}).get("buy_date") or "").replace("-", "")
            if sd.isdigit():
                sig.add(int(sd))
            if bd.isdigit():
                buy.add(int(bd))
        except Exception:      # noqa: BLE001
            pass
    return sig, buy


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=250)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    all_dates = sorted(p.stem.replace(".csv", "") for p in PANEL.glob("*.csv.gz"))
    use = all_dates[-(a.days + MAX_WIN + 260):]     # 预热: 最长窗口 + 250(上市天数)
    d = load_panel(use)
    print(f"面板 {use[0]}~{use[-1]}  {len(d):,} 行")

    # ---- 市场: 等权指数距 20 日线（与 build_features_234 同口径） ----
    g = d.groupby("trade_date", sort=True)
    st = pd.DataFrame({"eq_ret": g["pct"].mean(), "med_pct": g["pct"].median()}).reset_index()
    st["eq_nav"] = (1 + st["eq_ret"] / 100).cumprod()
    st["ma20"] = st["eq_nav"].rolling(20, min_periods=10).mean()
    st["idx_dist"] = (st["eq_nav"] / st["ma20"] - 1) * 100
    mkt = st.set_index("trade_date").reindex(sorted(d["trade_date"].unique()))

    # ---- 个股: 必须先在未过滤序列上算滚动量, 再过滤股票池 ----
    #   若先删停牌行再 rolling, "前 W 个交易日"会变成"前 W 个有效行", 与回测口径不一致。
    d["adj"] = d["close"] * d["factor"]
    d["adj_h"] = d["high"] * d["factor"]
    d["adj_o"] = d["open"] * d["factor"]
    d = d.sort_values(["ts_code", "trade_date"])
    minp = {W: max(W // 3, 40) for W in WINS}
    for W in WINS:
        # 不含当日: 先 shift(1) 再 rolling（含当日会低估信号强度）
        hi = d.groupby("ts_code")["adj_h"].transform(
            lambda s, W=W: s.shift(1).rolling(W, min_periods=minp[W]).max())
        d[f"dh{W}"] = (d["adj"] / hi - 1) * 100
    d["n1_gap"] = (d.groupby("ts_code")["adj_o"].shift(-1) / d["adj"] - 1) * 100
    d["bars"] = d.groupby("ts_code").cumcount() + 1

    # ---- 过滤股票池: 主板 / 非停牌 / 非 ST / 涨跌幅正常 ----
    # 3 万本金开不了创业板(10万)/科创板(50万)/北交所(50万); 北交所 30% 涨跌幅会放大假收益
    d = d[(d["is_paused"] == 0) & (d["is_st"] == 0)]
    d = d[d["pct"].between(-11.5, 11.5)]
    d = d[d["ts_code"].str[:2].isin(["60", "00"])]

    win = d[(d["bars"] >= MIN_LISTED_BARS) & d["dh60"].notna()].copy()
    win["idx_dist"] = win["trade_date"].map(mkt["idx_dist"])
    win["circ_mv"] = win["amount"] / win["turnover_rate"].replace(0, np.nan) / 1e6
    # ★ 2024 修正: 先在全池上对每天算 9 窗口排名, 再判断触发。
    #   旧版先用 (idx_dist, dh60) 过滤出破位日, 只能做"破位"这一条路;
    #   246 要加"个股自己崩"这条路, 而它可能出现在任何一天。
    win = win.sort_values(["trade_date", "amount"], ascending=[True, False])
    days = sorted(win["trade_date"].unique())[-a.days:]
    print(f"\n近 {a.days} 个交易日，逐日算排名")

    dhcols = [f"dh{W}" for W in WINS]

    def ranked(g: pd.DataFrame, exclude: set | None = None) -> pd.DataFrame:
        """9 窗口排名百分比平均, 升序返回（名列前茅 = 9 个窗口都认为跌得透）"""
        g2 = g.dropna(subset=dhcols).copy()
        if not len(g2):
            return g2
        if exclude:
            g3 = g2[~g2["ts_code"].isin(exclude)]
            if len(g3):
                g2 = g3
        denom = {c: max(int(g2[c].notna().sum()), 1) for c in dhcols}
        rk = pd.DataFrame({c: g2[c].rank(ascending=True) / denom[c] for c in dhcols})
        g2["mr"] = rk.mean(axis=1, skipna=True)
        return g2.sort_values(["mr", "amount"], ascending=[True, False])

    di = {int(d): i for i, d in enumerate(all_dates)}
    main_sig, main_buy = load_main_signals()
    print(f"主策略占用: {len(main_sig)} 个信号日 / {len(main_buy)} 个买入日")
    ntimes: dict[str, int] = {}
    last_fill: dict[str, int] = {}
    rows = []
    # ★ 不要预先建 {date: DataFrame}, 那是把全量面板再复制一份(900 万行必 OOM)。
    #   直接迭代 groupby, 只保留当天的切片。
    want = set(int(x) for x in days)
    last_group = None
    for dt, g in win.groupby("trade_date", sort=True):
        dt = int(dt)
        if dt == int(all_dates[-1]):
            last_group = g
        # ★ 不要把窗口外的日子直接跳过: 单票限次(<=8)必须从面板最早一天开始累计,
        #   否则实盘从"0 次"起算会比回测宽松得多(实测 1200 日窗口多做出 54 次交易)。
        #   窗口外的日子只参与计数, 不写进 history。
        full = ranked(g)                       # 全池排名, 用来判断"今天最超跌的是谁"
        if not len(full):
            continue
        full["rk"] = np.arange(1, len(full) + 1)
        top0 = full.iloc[0]
        # ★ 破位判定只用 T 日可知的信息(指数 + 这只票自己的 dh60)。
        #   246 曾经在"明天会低开的票"里取第一名去判破位 —— 那是未来的信息, 已修。
        broke = (float(top0["idx_dist"]) <= IDX_DIST_MAX
                 and float(top0["dh60"]) <= DIST_HIGH_MAX)
        free = full["ts_code"].map(lambda c: ntimes.get(c, 0) < MAX_TIMES).to_numpy()
        low = (full["n1_gap"] <= N1_GAP_MAX).to_numpy()
        if broke:
            # 252: 同批低开票里挑开盘最低的那只（经济含义: 买最便宜的, 赚当日超卖反弹）
            sel_ = full[low & free].sort_values("n1_gap") if PICK_BREAK == "lowest" else full[low & free]
            why = "破位"
        else:
            # ★ 深低开: T+1 9:25 全池都能看到开盘价, 不必只盯 T 日第一名。
            #   252: 名单 = 第 DEEP_SKIP+1 ~ DEEP_SKIP+DEEP_RANK_MAX 名（跳过最崩的 3 只）——
            #   1-3 名单笔 0.43%, 4-8 名 2.29%, 距前高最远的是长期崩塌趋势。
            keep = (low & free
                    & full["n1_gap"].between(DEEP_LO, DEEP_HI, inclusive="right").to_numpy()
                    & (full["rk"] > DEEP_SKIP).to_numpy()
                    & (full["rk"] <= DEEP_SKIP + DEEP_RANK_MAX).to_numpy())
            sel_ = full[keep]
            why = "深低开"
        if not len(sel_):
            if broke and dt in want:   # 破位成立但一只低开的都没有 —— 值得留痕
                rows.append({"signal": dt, "idx_dist": round(float(top0["idx_dist"]), 2),
                             "n": len(g), "code": "-", "mr": np.nan, "gap": np.nan,
                             "why": why, "let": False})
            continue       # 深低开没候选 = 这天本来就没事, 不记
        # ★ 资金占用（249 定稿）: 主策略 signal 日 / buy 日的钱不在这, 补位不做。
        #   248 曾让补位在重叠日顶掉主策略（let_way），合入真实引擎后测出是负贡献
        #   （2500 日卡玛 4.64 → 4.27），已关闭。卖出日不排（资金当日回笼）。
        if dt in main_sig or dt in main_buy:
            continue
        let_way = False
        # ★ 冷却（251）: 刚补过就不再连做。
        #   破位 332 笔里 278 笔是同一波的第 2 笔及以后，最长一波连做 23 个交易日 ——
        #   那是"越跌越买"的网格摊平，不是单吊。窗口外的日子同样计入。
        #   ★ 按腿分开算（全局冷却实测更差: 61 笔、卡玛 4.63 vs 71 笔、5.37）。
        if COOLDOWN_DAYS > 0:
            _ci = di.get(dt)
            _lc = last_fill.get(why)
            if _ci is not None and _lc is not None and (_ci - _lc) <= COOLDOWN_DAYS:
                continue
        top = sel_.iloc[0]
        ntimes[top["ts_code"]] = ntimes.get(top["ts_code"], 0) + 1
        if di.get(dt) is not None:
            last_fill[why] = di[dt]
        if dt not in want:
            continue
        r = {"signal": dt, "idx_dist": round(float(top0["idx_dist"]), 2),
             "n": len(g), "code": top["ts_code"],
             "mr": round(float(top["mr"]), 3),
             "gap": round(float(top["n1_gap"]), 2), "why": why,
             "pos": POS_BREAK if broke else POS_DEEP, "let": let_way}
        for W in WINS:
            r[f"dh{W}"] = round(float(top[f"dh{W}"]), 1)
        rows.append(r)
    hist = pd.DataFrame(rows)
    if len(hist):
        ok = hist[hist["code"] != "-"]
        cols = ["signal", "idx_dist", "n", "why", "code", "mr", "gap", "pos", "let"]
        print(hist[[c for c in cols if c in hist.columns]].to_string(index=False))
        print(f"\n回放: 触发 {len(hist)} 次, 成交 {len(ok)} 次"
              + (f", 平均低开 {ok['gap'].mean():.2f}%" if len(ok) else ""))
        if len(ok):
            print(f"  其中仅深低开触发 {int((ok['why'] == '深低开').sum())} 次")
        hist.to_csv(OUT / "backfill_history.csv", index=False, encoding="utf-8-sig")

    # ---- 最新交易日: 只给排序名单, 开盘价明天才知道 ----
    last = all_dates[-1]
    li = float(mkt["idx_dist"].get(int(last), np.nan))
    gl = last_group
    print(f"\n最新交易日 {last}: 等权指数距20日线 {li:+.2f}%")
    if gl is not None and len(gl):
        cand = ranked(gl)
        top0 = cand.iloc[0]
        d60 = float(top0["dh60"])
        broke = (li <= IDX_DIST_MAX) and (d60 <= DIST_HIGH_MAX)
        print(f"  当天最超跌: {top0['ts_code']}  距60日高 {d60:.1f}%  "
              f"→ 破位条件 {'✅ 满足' if broke else '❌ 不满足'}")
        # 252: 破位列前 TOP_N 只（在里头挑开盘最低的），深低开列第 4~6 名 —— 表与文案一致
        show = (cand.head(TOP_N) if broke
                else cand.iloc[DEEP_SKIP:DEEP_SKIP + DEEP_RANK_MAX])
        print(show[["ts_code", "close", "pct", "mr", "dh60", "dh250", "dh360", "amount"]]
              .to_string(index=False))
        print("\n  ★ 明早 9:25 开盘后, 两条路二选一:")
        print(f"    ① 破位路径 {'✅ 今天成立' if broke else '❌ 今天不成立'}"
              f" (指数 {li:+.2f}% <= {IDX_DIST_MAX}% 且 第一名距60日高 {d60:.1f}% <= {DIST_HIGH_MAX}%)")
        print(f"       → 上表 {TOP_N} 只里, **开盘价最低**的那只买入(且 <= 昨收×1.0005), 仓位 {POS_BREAK:.0%}")
        print(f"    {'②' if not broke else '②(备选)'} 深低开路径: 上表这 {DEEP_RANK_MAX} 只里,"
              f" 谁的开盘缺口落在 {DEEP_LO}% ~ {DEEP_HI}% 就买谁(取排名最好的), 仓位 {POS_DEEP:.0%}")
        print(f"       名单是 9 窗口排名第 {DEEP_SKIP + 1}~{DEEP_SKIP + DEEP_RANK_MAX} 名 —— "
              f"最崩的 {DEEP_SKIP} 只不接(1-3 名单笔 0.43%, 4-8 名 2.29%)")
        print(f"       注意: 缺口 < {DEEP_LO}% 的不做(跌得更深, 次日反弹不确定);")
        print(f"             缺口 > {DEEP_HI}% 跌得不够, 也不做")
        print("    破位成立时走①(两条都成立也走①), 只有②成立时走②, 都不成立就空仓")
        print(f"    冷却(251): 同一条腿成交后 {COOLDOWN_DAYS} 个交易日内不再做 —— "
              "一次暴跌只抄一次底, 不是越跌越买")
        print("    持有: T+1 开盘买入 → T+2 收盘卖出, 不挂限价单(247 验证过, 挂单必更差)")
        print("    资金占用: 主策略卖出日不排(资金当日回笼), 只排 signal 日与买入日")
        ms = int(last) in main_sig
        print(f"    今天主策略{'✅ 有信号' if ms else '❌ 没信号'} —— "
              + ("补位不做（钱要留给主策略，249: 让位实测是负贡献）"
                 if ms else "补位可以做（主策略明天不占用资金）"))
        print(f"    历史(255 定稿, 2500日真实账户): 96.4% / -17.2% / 5.60  (补位 50 笔)")
        print("    注: 历史回放的\"主策略占用\"取自 w232_trades.csv(全期回测), 与任意单次回测窗口的")
        print("        主策略交易可能差几笔, 会连锁影响 10 日冷却窗口 —— 审计时个别日期对不上属正常。")
        print("    对照: 249 口径 108.1% / -23.3% / 4.64 (补位 306 笔) —— 笔数砍 77%, 年化 -15pp")


def _write_json() -> None:
    pass


if __name__ == "__main__":
    main()
