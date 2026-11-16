"""第241轮G: 如果破位日选股 alpha 是零, 那么"随机抽10只"应该和"成交额前10"同分布"""
from pathlib import Path
import numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[1]; FEAT=ROOT/"output"/"feat234"; OUT=ROOT/"output"
COST=0.10; rng=np.random.default_rng(241)
COLS=["ts_code","trade_date","y","n1_ok","pct","circ_mv","idx_dist","dist_high60","amt_yi","is_paused","is_st","listed"]
parts=[]
for p in sorted(FEAT.glob("f*.parquet")):
    d=pd.read_parquet(p,columns=COLS)
    d=d[(d.is_paused==0)&(d.is_st==0)&(d.n1_ok==True)&d.pct.between(-11.5,11.5)&d.y.between(-40,60)&(d.listed>=250)]
    d=d[d.ts_code.str[:2].isin(["60","00"])]
    if len(d): parts.append(d)
d=pd.concat(parts,ignore_index=True)
def M(r):
    nav=(1+r/100).cumprod(); ann=nav.iloc[-1]**(244/len(r))-1; dd=(nav/nav.cummax()-1).min()
    return ann*100, dd*100, ann*100/abs(dd*100)
cal=np.sort(d.trade_date.unique()); tr=pd.read_csv(OUT/"w232_trades.csv")
cal=cal[cal>=tr.signal.min()]; tr=tr[tr.signal.isin(cal)]
busy=set()
for _,r in tr.iterrows(): busy.add(r.buy); busy.add(r.sell)
sig=set(tr.signal); free=set(x for x in cal if x not in busy and x not in sig)
base=pd.Series(0.0,index=cal)
for dt,g in tr.groupby("signal"): base[dt]=g["ret"].mean()
b=M(base)
nxt={cal[i]:cal[i+1] for i in range(len(cal)-1)}; nxt[cal[-1]]=cal[-1]
s=d[(d.idx_dist<=-3.51)&(d.dist_high60<=-24.46)]
days=[]
for dt in sorted(s.trade_date.unique()):
    if dt not in free: continue
    bd=nxt.get(dt); sd=nxt.get(bd,bd)
    if bd in busy or sd in busy: continue
    days.append(dt)
sub={dt:g for dt,g in d[d.trade_date.isin(days)].groupby("trade_date")}
def run(k, w):
    v={}
    for dt,g in sub.items():
        idx=rng.choice(len(g), size=min(k,len(g)), replace=False)
        v[dt]=g["y"].values[idx].mean()-COST
    v=pd.Series(v); sim=base.copy(); sim[v.index]+=v.values*w
    return M(sim)+(v.mean(),)
print(f"基准 年化{b[0]:.1f}% 回撤{b[1]:.2f}% 卡玛{b[2]:.2f}")
for k in (1,5,10,20):
    res=np.array([run(k,1.0) for _ in range(200)])
    q=np.percentile(res[:,2],[5,50,95])
    print(f" 随机{k:>2}只 满仓: 卡玛 中位{np.median(res[:,2]):.2f}  5~95% [{q[0]:.2f},{q[2]:.2f}]  "
          f"单笔中位 {np.median(res[:,3]):+.3f}%  回撤中位 {np.median(res[:,1]):.1f}%")
# 对照: 成交额前10
v={dt:g.nlargest(10,"amt_yi")["y"].mean()-COST for dt,g in sub.items()}
v=pd.Series(v); sim=base.copy(); sim[v.index]+=v.values*1.0
a,dd,cm=M(sim)
print(f" 成交额前10 满仓: 年化{a:.1f}% 回撤{dd:.2f}% 卡玛{cm:.2f}")
