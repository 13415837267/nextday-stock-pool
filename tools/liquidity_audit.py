# -*- coding: utf-8 -*-
"""第六十八轮: 流动性/结构因子族诊断(文献驱动, 事前注册, 零参数变更)。

文献依据: 《中国A股的隔夜-日内反转效应》(经济学报2020)——该效应随非流动性
加剧、卖空限制增强、散户占比提高而增强。本策略吃"深低开->日内修复"即该效应
的交易化表达, 假设: 信号票非流动性越高(流动性越差), 承接修复越强。

事前注册四假设(157笔全样本, top5口径):
  H4 Amihud非流动性: 20日均值(|pct| / amount亿元), 越高越"非流动" -> 假设正向
  H5 换手率均值: 20日turnover_rate均值 -> 探索(双向)
  H6 信号日回调深度: T日pct(-6~-3%区间内越深越恐慌) -> 探索(双向)
  H7 规模(成交额): 20日amount均值(亿元), 小票弹性 -> 假设正向(小盘)

判据(同六十七轮): 五等分桶顶底差>=+1pp/笔且大体单调+era后同向 -> 观察级苗头;
接入须走前过滤双段同向且>=+0.5pp/笔。
"""
import json, math
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path('/workspace/longhu_picker')
bl = json.load(open(ROOT / 'output/backtest_long.json'))
trades = [t for t in bl['trades_all'] if t.get('ret_pct') is not None]
cal = sorted(p.stem.replace('.csv', '') for p in (ROOT / 'data' / 'panel').glob('*.csv.gz'))
WIN = 25
ERA = '20240901'

rows, miss = [], 0
for t in trades:
    S, C = t['signal_date'], t['code']
    try:
        i = cal.index(S)
    except ValueError:
        miss += 1; continue
    if i < WIN - 1:
        miss += 1; continue
    days = cal[i - WIN + 1:i + 1]
    amihud, tors, amts = [], [], []
    dip_pct = None
    for d in days:
        df = pd.read_csv(ROOT / 'data' / 'panel' / f'{d}.csv.gz', dtype={'trade_date': str})
        row = df[df['ts_code'] == C]
        if row.empty: continue
        r = row.iloc[0]
        if int(r['is_paused']) == 1: continue
        pct, amt, to = abs(float(r['pct'])), float(r['amount']) / 1e8, float(r['turnover_rate'])
        if amt > 0: amihud.append(pct / amt)
        tors.append(to); amts.append(amt)
        if d == S: dip_pct = float(r['pct'])
    if len(amihud) < 20 or dip_pct is None:
        miss += 1; continue
    rows.append(dict(ret=t['ret_pct'], sell=t['sell_date'],
                     amihud=float(np.mean(amihud[-20:])),
                     turnover=float(np.mean(tors[-20:])),
                     amount=float(np.mean(amts[-20:])),
                     dip=dip_pct))

OUT = ROOT / "output" / "liquidity_audit.json"
print(f'宇宙: {len(rows)}笔(缺{miss})')

def buckets(key, label, asc=True, n_b=5):
    print(f'\n== {label} 五等分桶 ==')
    s = sorted(rows, key=lambda r: r[key], reverse=not asc)
    n, sz = len(s), len(s) // 5
    avgs = []
    for b in range(n_b):
        seg = s[b * sz:(b + 1) * sz] if b < n_b - 1 else s[b * sz:]
        rets = [x['ret'] for x in seg]
        comp = math.prod(1 + x / 100 for x in rets)
        avgs.append(sum(rets) / len(rets))
        tag = f'桶{b+1}' + ('(低)' if b == 0 else '(高)' if b == n_b - 1 else '')
        print(f'  {tag:6s} n={len(rets):3d} 均值{sum(rets)/len(rets):+.3f}%/笔 复利{(comp-1)*100:+.1f}%')
    spread = avgs[-1] - avgs[0]
    e = [r for r in rows if r['sell'] >= ERA]
    se = sorted(e, key=lambda r: r[key], reverse=not asc)
    sz2 = len(se) // n_b
    lo = [x['ret'] for x in se[:sz2]]
    hi = [x['ret'] for x in se[(n_b-1)*sz2:]]
    print(f'  顶底差: {spread:+.3f}pp/笔 | era后 低桶{sum(lo)/len(lo):+.2f}%(n={len(lo)}) vs 高桶{sum(hi)/len(hi):+.2f}%(n={len(hi)})')
    return spread

buckets('amihud', 'H4: Amihud非流动性(低->高)')
buckets('turnover', 'H5: 换手率均值(低->高)')
buckets('dip', 'H6: 信号日回调深度(浅->深)')
buckets('amount', 'H7: 成交额规模(小->大)')

(OUT).write_text(json.dumps({'universe_n': len(rows), 'missing': miss,
    'note': 'H4-H7全部证伪或弱(2026-10-27): 顶底差 Amihud -0.587/换手 -0.846/回调深度 +0.445/成交额 +1.118pp/笔, 均非单调或era后不满足观察级判据; 详见CHANGELOG第六十八轮'}, ensure_ascii=False, indent=1))
print(f'已写出 {OUT}')
