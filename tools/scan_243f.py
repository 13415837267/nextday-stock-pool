"""第243轮 F: 验证"超跌越深、后面越差"是否单调 —— 决定分档仓位是规律还是拟合

243E 唯一三段全正的方案是"越深越轻"(idx<=-6 上 40% 仓),
但三档阈值(4.5 / 6.0)是随手定的。本轮先看原始事实: 按破位深度分档, 单笔收益怎么走。
  单调下降 => 规律, 可以用连续仓位函数
  锯齿     => 拟合, 放弃
"""
from pathlib import Path
import numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[1]; FEAT=ROOT/"output"/"feat234"; OUT=ROOT/"output"
COST=0.10
COLS=["ts_code","trade_date","y","n1_ok","pct","idx_dist","dist_high60","amt_yi","is_paused","is_st","listed"]
parts=[]
for p in sorted(FEAT.glob("f*.parquet")):
    d=pd.read_parquet(p,columns=COLS)
    d=d[(d.is_paused==0)&(d.is_st==0)&(d.listed>=250)]
    d=d[d.pct.between(-11.5,11.5)&d.y.between(-40,60)]
    d=d[d.ts_code.str[:2].isin(["60","00"])]
    d=d[d.n1_ok==True]
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
b=M(base); print(f"基准 年化{b[0]} 回撤{b[1]} 卡玛{b[2]}")
nxt={cal[i]:cal[i+1] for i in range(len(cal)-1)}; nxt[cal[-1]]=cal[-1]
s=d[(d.idx_dist<=-3.51)&(d.dist_high60<=-24.46)]
days=[]
for dt in sorted(s.trade_date.unique()):
    if dt not in free: continue
    bd=nxt.get(dt); sd=nxt.get(bd,bd)
    if bd in busy or sd in busy: continue
    days.append(dt)
rec=[]
for dt in days:
    g=d[(d.trade_date==dt)&d.dist_high60.notna()]
    if not len(g): continue
    g=g.sort_values(["dist_high60","amt_yi"],ascending=[True,False])
    r=g.iloc[0]
    rec.append({"signal":dt,"y":r.y-COST,"idx":r.idx_dist,"code":r.ts_code})
Q=pd.DataFrame(rec)
print(f"补位 {len(Q)} 次  idx_dist 分布: {Q.idx.describe().round(2).to_dict()}\n")
print("① 按破位深度分档 (看单笔是否单调)")
Q["b"]=pd.cut(Q.idx,[-100,-8,-6.5,-5.5,-4.5,-3.51],labels=["<=-8","-8~-6.5","-6.5~-5.5","-5.5~-4.5","-4.5~-3.5"])
g=Q.groupby("b",observed=True).agg(n=("y","size"),mean=("y","mean"),med=("y","median"),
                                   worst=("y","min"),best=("y","max"))
g["win%"]=Q.groupby("b",observed=True).y.apply(lambda x:(x>0).mean()*100).round(0)
g["std"]=Q.groupby("b",observed=True).y.std().round(2)
print(g.round(2).to_string())
print("\n  分年 各档单笔:")
t=Q.copy(); t["yr"]=t.signal//10000
print(t.pivot_table(index="yr",columns="b",values="y",aggfunc="mean",observed=True).round(2).to_string())

print("\n② 分段 (看单调性是否稳定)")
for y0,y1 in ((2016,2019),(2020,2023),(2024,2026)):
    tt=Q[(Q.signal//10000>=y0)&(Q.signal//10000<=y1)]
    gg=tt.groupby("b",observed=True).y.agg(["size","mean"]).round(2)
    print(f"  {y0}-{y1}: {gg['mean'].to_dict()}  笔数{gg['size'].to_dict()}")

print("\n③ 连续仓位函数 (避免三档阈值拟合): 仓位 = clip(1 - k*(-3.51-idx)/10, lo, 1)")
def ev(wmap,tag):
    v=pd.Series(dict(zip(Q.signal,Q.y)))
    w=pd.Series(wmap).reindex(v.index)
    sim=base.copy(); sim[v.index]+=(v.values*w.values)
    a,dd,cm=M(sim)
    segs=[]
    for y0,y1 in ((2016,2019),(2020,2023),(2024,2026)):
        m=(sim.index//10000>=y0)&(sim.index//10000<=y1)
        _,_,c=M(sim[m]); _,_,cb=M(base[m]); segs.append(round(c-cb,2))
    return {"方案":tag,"年化":a,"回撤":dd,"卡玛":cm,"Δ":round(cm-b[2],2),"三段":segs,
            "平均仓%":round(w.mean()*100,0)}
rows=[]
for k in (0.05,0.08,0.10,0.15,0.30):
    w={r.signal:max(0.3,min(1.0,1-k*(-3.51-r.idx)/10)) for r in Q.itertuples()}
    rows.append(ev(w,f"线性 k={k}"))
for k in (0.08,0.10,0.15):
    w={r.signal:max(0.3,min(1.0,1-k*(-3.51-r.idx)/10)) for r in Q.itertuples()}
    rows.append(ev({s:ww*0.5 for s,ww in w.items()},f"线性 k={k} 再砍半"))
print(pd.DataFrame(rows).to_string(index=False))
Q.to_csv(OUT/"w243f_picks.csv",index=False,encoding="utf-8-sig")
