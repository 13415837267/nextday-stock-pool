"""第241轮F: 同一波下跌里重复抄底是亏的 —— 加"每波限次"能否改善"""
from pathlib import Path
import numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[1]; FEAT=ROOT/"output"/"feat234"; OUT=ROOT/"output"
COST=0.10
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
    return round(ann*100,2),round(dd*100,2),round(ann*100/abs(dd*100),2)
cal=np.sort(d.trade_date.unique()); tr=pd.read_csv(OUT/"w232_trades.csv")
cal=cal[cal>=tr.signal.min()]; tr=tr[tr.signal.isin(cal)]
busy=set()
for _,r in tr.iterrows(): busy.add(r.buy); busy.add(r.sell)
sig=set(tr.signal); free=set(x for x in cal if x not in busy and x not in sig)
base=pd.Series(0.0,index=cal)
for dt,g in tr.groupby("signal"): base[dt]=g["ret"].mean()
b=M(base); print(f"基准 {b}")
nxt={cal[i]:cal[i+1] for i in range(len(cal)-1)}; nxt[cal[-1]]=cal[-1]
s=d[(d.idx_dist<=-3.51)&(d.dist_high60<=-24.46)]
days=[]
for dt in sorted(s.trade_date.unique()):
    if dt not in free: continue
    bd=nxt.get(dt); sd=nxt.get(bd,bd)
    if bd in busy or sd in busy: continue
    days.append(dt)
sub=d[d.trade_date.isin(days)]
y3={dt:g.nlargest(min(10,len(g)),"amt_yi")["y"].mean()-COST for dt,g in sub.groupby("trade_date")}
y4={dt:g["y"].mean()-COST for dt,g in sub.groupby("trade_date")}
print(f"严格口径 {len(days)} 次")
dt=pd.to_datetime(pd.Series(days).astype(str)); wave=(dt.diff().dt.days>10).cumsum().values
def variant(keep_first=None, keep_n=None, stop_after=0):
    """返回保留的天数列表"""
    out=[]; cnt_in_wave={}; losses={}
    for d0,w in zip(days,wave):
        c=cnt_in_wave.get(w,0)
        if keep_first is not None and c>=keep_first: continue
        if keep_n is not None and c>=keep_n: continue
        if stop_after and losses.get(w,0)>=stop_after: continue
        out.append(d0)
        cnt_in_wave[w]=c+1
    return out
for tag,yv in (("S3 流动性10只",y3),("S4 全等权",y4)):
    print(f"\n=== {tag}")
    for name,args in (("每次都做",{}),("每波只做第1次",{"keep_first":1}),
                      ("每波最多3次",{"keep_n":3}),("每波最多2次",{"keep_n":2}),
                      ("每波连亏2次即停",{"stop_after":2})):
        ks=variant(**args)
        v=pd.Series({k:yv[k] for k in ks})
        for w in (0.3,1.0):
            if tag.startswith("S4") and w==0.3: continue
            sim=base.copy(); sim[v.index]+=v.values*w; a,dd,cm=M(sim)
            print(f"  {name:<16} {len(v):>3}次 {w:>4.0%}仓  单笔{v.mean():+.3f}% 最差{v.min():.1f}% 年化{a:6.2f} 回撤{dd:6.2f} 卡玛{cm:.2f} (基准{b[2]})")
