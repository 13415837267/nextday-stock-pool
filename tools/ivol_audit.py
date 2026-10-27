# -*- coding: utf-8 -*-
"""IVOL特质波动率因子审计(第六十七轮, 文献驱动, 事前注册, 零参数变更)。

文献依据: 陆蓉/王麒《经济学(季刊)》2025"昼伏夜出的特质波动率之谜——股价
'低开高走'的经济学含义"——机构主导隔夜(压高低特质波动股->低开), 散户主导
日内(推高->高走)。本策略吃"深低开->日内修复"段, 假设: 信号票IVOL越高承接
修复越强(正向因子)。

口径:
  - 宇宙: backtest_long.json 全部执行成交(ret_pct 净口径, top5口径157笔)
  - IVOL20: 信号日往前20个交易日, (个股日收益pct - 市场等权日均pct)的标准差
    (市场调整模型; 市场等权日收益=当日panel全市场pct均值, 剔除停牌)
  - 判据(注册): ①五等分桶顶底差>=+1pp/笔且大体单调 -> 观察级苗头;
    ②"剔除IVOL最低五分位"过滤在era前/era后双段同向改善且>=+0.5pp/笔 -> 讨论接入
首轮结果(2026-10-27): ①成立(顶底差+2.90pp/笔, 桶1 -0.585% -> 桶5 +2.313%,
  era后同向); ②不成立(era前+0.05pp/era后+0.10pp, 双段同向但量级不足)。
  -> **维持观察级不接入**, 重验线=era后样本IVOL五分位桶单调复验(每桶n>=10,
  现n=7~11)或执行样本累计翻倍; 接入路径=评分权重 dip_weights 增加 ivol 因子
  (须先经 h1/h2 走前验证)。

用法: python tools/ivol_audit.py
输出: 打印 + output/ivol_audit.json
"""
import json, math
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ERA = '20240901'
WIN = 25
OUT = ROOT / 'output' / 'ivol_audit.json'


def main():
    bl = json.loads((ROOT / 'output' / 'backtest_long.json').read_text())
    trades = [t for t in bl['trades_all'] if t.get('ret_pct') is not None]
    cal = sorted(p.stem.replace('.csv', '') for p in (ROOT / 'data' / 'panel').glob('*.csv.gz'))

    mkt = {}
    def market_pct(d):
        if d in mkt: return mkt[d]
        p = ROOT / 'data' / 'panel' / f'{d}.csv.gz'
        if not p.exists(): return None
        df = pd.read_csv(p, usecols=['pct', 'is_paused'])
        v = float(df.loc[df['is_paused'] == 0, 'pct'].mean())
        mkt[d] = v
        return v

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
        pct = []
        for d in days:
            df = pd.read_csv(ROOT / 'data' / 'panel' / f'{d}.csv.gz', dtype={'trade_date': str})
            row = df[df['ts_code'] == C]
            if row.empty: continue
            r = row.iloc[0]
            if int(r['is_paused']) == 1: continue
            pct.append((d, float(r['pct'])))
        if len(pct) < 20:
            miss += 1; continue
        last = pct[-20:]
        mk = [market_pct(d) for d, _ in last]
        resid = [pi - mi for (_, pi), mi in zip(last, mk) if mi is not None]
        rows.append(dict(signal_date=S, sell_date=t['sell_date'], code=C,
                         rank=t.get('rank'), ret=t['ret_pct'],
                         ivol=float(np.std(resid, ddof=1))))

    def st(sel):
        if not sel: return {'n': 0}
        rets = [x['ret'] for x in sel]
        comp = math.prod(1 + x / 100 for x in rets)
        return {'n': len(rets), 'avg': round(sum(rets) / len(rets), 3),
                'wr': round(100 * sum(x > 0 for x in rets) / len(rets), 1),
                'comp': round((comp - 1) * 100, 1)}

    result = {'universe_n': len(rows), 'missing': miss, 'era_split': ERA,
              'bucket_quintile': {}, 'walkforward_filter': {}, 'corr': {}}

    # 五等分桶
    s = sorted(rows, key=lambda r: r['ivol'])
    n, sz = len(s), len(s) // 5
    for b in range(5):
        seg = s[b * sz:(b + 1) * sz] if b < 4 else s[b * sz:]
        result['bucket_quintile'][f'bucket{b+1}'] = st(seg)
    b1, b5 = result['bucket_quintile']['bucket1'], result['bucket_quintile']['bucket5']
    result['bucket_quintile']['top_bottom_spread'] = round(
        (b5.get('avg') or 0) - (b1.get('avg') or 0), 3)

    # 走前过滤(剔除最低五分位)
    q20 = sorted(r['ivol'] for r in rows)[len(rows) // 5]
    filt = [r for r in rows if r['ivol'] > q20]
    for seg_name, base_s, filt_s in (
            ('all', rows, filt),
            ('era前', [r for r in rows if r['sell_date'] < ERA], [r for r in filt if r['sell_date'] < ERA]),
            ('era后', [r for r in rows if r['sell_date'] >= ERA], [r for r in filt if r['sell_date'] >= ERA])):
        result['walkforward_filter'][seg_name] = {'base': st(base_s), 'filtered': st(filt_s)}

    iv = [r['ivol'] for r in rows]; rt = [r['ret'] for r in rows]
    result['corr']['all'] = round(float(np.corrcoef(iv, rt)[0, 1]), 3)
    iv_e = [r['ivol'] for r in rows if r['sell_date'] >= ERA]
    rt_e = [r['ret'] for r in rows if r['sell_date'] >= ERA]
    if len(iv_e) > 2:
        result['corr']['era后'] = round(float(np.corrcoef(iv_e, rt_e)[0, 1]), 3)

    print(f"宇宙: {len(rows)}笔(缺{miss})")
    print(f"五等分桶顶底差: {result['bucket_quintile']['top_bottom_spread']:+.3f}pp/笔 "
          f"(桶1 {b1.get('avg')}% -> 桶5 {b5.get('avg')}%)")
    wf = result['walkforward_filter']
    for seg in ('all', 'era前', 'era后'):
        print(f"  {seg}: BASE {wf[seg]['base'].get('avg')}% -> 过滤 {wf[seg]['filtered'].get('avg')}%")
    print(f"相关系数: 全窗 {result['corr']['all']}, era后 {result['corr'].get('era后')}")
    print("判据: 双段同向但<+0.5pp -> 维持观察级不接入(重验线见脚本头注)")
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f"已写出 {OUT}")


if __name__ == '__main__':
    main()
