"""第243轮 E: 单吊满仓的回撤 -45% —— 用"波次"和"分档仓位"压

242 轮结论: 回撤结构性来自单只 -16.7%, 换选股键解决不了, 只能降仓位。
但还有两个没测的旋钮, 都直接作用在"连续抄底"这件事上:
  ① 波次限制: 同一波下跌(间隔<=N日)只做前 K 次 —— 避免在下跌中继反复满仓
  ② 分档仓位: 超跌越深仓位越高(反向于直觉, 但要数据说话)
  ③ 连亏降仓: 连续亏损后仓位减半

判断标准(事先定, 不许事后改):
  - 年化不低于基准 88%
  - 回撤从 -45.2% 明显收窄(至少 5pp)
  - 三段(16-19/20-23/24-26)卡玛都不低于现行补位
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
b=M(base); print(f"基准 {b}")
nxt={cal[i]:cal[i+1] for i in range(len(cal)-1)}; nxt[cal[-1]]=cal[-1]
def pick(lo,hi_high=-24.46):
    s=d[(d.idx_dist<=lo)&(d.dist_high60<=hi_high)]
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
        rec.append({"signal":dt,"code":r.ts_code,"y":r.y-COST,"idx":r.idx_dist})
    return pd.DataFrame(rec)
Q=pick(-3.51)
print(f"补位 {len(Q)} 次\n")
SEG=[(2016,2019),(2020,2023),(2024,2026)]
def evalv(tag,v_by_day,w_by_day=None):
    v=pd.Series(v_by_day); sim=base.copy()
    if w_by_day is None: sim[v.index]+=v.values
    else:
        w=pd.Series(w_by_day); sim[v.index]+=(v.values*w.reindex(v.index).fillna(0).values)
    a,dd,cm=M(sim)
    segs=[]
    for y0,y1 in SEG:
        m=(sim.index//10000>=y0)&(sim.index//10000<=y1)
        _,_,c=M(sim[m]); _,_,cb=M(base[m]); segs.append(round(c-cb,2))
    return {"方案":tag,"次数":len(v),"单笔":round(v.mean(),3),"年化":a,"回撤":dd,"卡玛":cm,
            "Δ卡玛":round(cm-b[2],2),"三段Δ":segs}
rows=[evalv("全做 满仓",dict(zip(Q.signal,Q.y)))]
# ① 波次限制
Q=Q.sort_values("signal").reset_index(drop=True)
Q["dt"]=pd.to_datetime(Q.signal.astype(str))
Q["gap"]=Q.dt.diff().dt.days
for N in (5,10,20):
    Q["wave"]=(Q.gap>N).cumsum()
    for K in (1,2,3):
        tot={};cnt={}
        for dt,w in zip(Q.signal,Q.wave):
            c=cnt.get(w,0)
            if c<K: tot[dt]=Q.loc[Q.signal==dt,"y"].iloc[0]; cnt[w]=c+1
        rows.append(evalv(f"每波(间隔>{N}日)最多{K}次",tot))
# ② 分档仓位
dep=dict(zip(Q.signal,Q.idx))
for name,f in (("深度分档 越深越重",lambda x:1.0 if x<=-6 else (0.7 if x<=-4.5 else 0.4)),
               ("深度分档 越深越轻",lambda x:0.4 if x<=-6 else (0.7 if x<=-4.5 else 1.0)),
               ("浅档减半",lambda x:1.0 if x<=-4.5 else 0.5)):
    w={k:f(v) for k,v in dep.items()}
    rows.append(evalv(name,dict(zip(Q.signal,Q.y)),w))
# ③ 连亏降仓
v=dict(zip(Q.signal,Q.y)); w={}; ls=0
for dt in Q.signal:
    w[dt]=1.0 if ls==0 else (0.5 if ls==1 else 0.25)
    ls=ls+1 if v[dt]<0 else 0
rows.append(evalv("连亏降仓 1/2/3次 -> 100/50/25%",v,w))
r=pd.DataFrame(rows)
pd.set_option("display.width",200)
print(r.to_string(index=False))
r.to_csv(OUT/"w243e_variants.csv",index=False,encoding="utf-8-sig")
