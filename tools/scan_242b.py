"""第242轮B: 单吊的尾部控制 —— 入场价过滤 (次日竞价缺口 n1_gap) + 仓位"""
from pathlib import Path
import numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[1]; FEAT=ROOT/"output"/"feat234"; OUT=ROOT/"output"
COST=0.10
COLS=["ts_code","trade_date","y","n1_ok","n1_gap","pct","circ_mv","idx_dist","dist_high60",
      "amt_yi","vol20","amp","is_paused","is_st","listed"]
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
sub={dt:g for dt,g in d[d.trade_date.isin(days)].groupby("trade_date")}
# 选股: 距60日高最远(跌最透), 同分取成交额大的保证能买
pick={}
for dt,g in sub.items():
    g2=g.dropna(subset=["dist_high60"]).sort_values(["dist_high60","amt_yi"],ascending=[True,False])
    if len(g2): pick[dt]=g2.iloc[0]
P=pd.DataFrame(pick.values(),index=list(pick.keys()))
print(f"单吊候选 {len(P)} 次  gap 分布: {P.n1_gap.describe().round(2).to_dict()}")
FILT={"全买":(None,None),"gap>=-3%":(-3,None),"gap>=-5%":(-5,None),"gap>=-8%":(-8,None),
      "gap<=-3%(深低开)":(None,-3),"gap<=-5%(极深低开)":(None,-5),
      "gap -5~-1%":(-5,-1),"gap -3~0%":(-3,0),"gap>=-1%(不深低开)":(-1,None)}
print("\n%s"%("="*94))
print("① 入场缺口过滤 (单吊满仓, 跌得最透那只)")
rows=[]
for lab,(lo,hi) in FILT.items():
    m=pd.Series(True,index=P.index)
    if lo is not None: m&=P.n1_gap>=lo
    if hi is not None: m&=P.n1_gap<=hi
    q=P[m]
    if len(q)<20: continue
    v=pd.Series((q.y-COST).values,index=q.index)
    sim=base.copy(); sim[v.index]+=v.values; a,dd,cm=M(sim)
    rows.append({"过滤":lab,"次数":len(v),"单笔":round(v.mean(),3),"胜率%":round((v>0).mean()*100,1),
                 "最差":round(v.min(),1),"年化":a,"回撤":dd,"卡玛":cm,"Δ":round(cm-b[2],2)})
r=pd.DataFrame(rows).sort_values("卡玛",ascending=False)
print(r.to_string(index=False))
r.to_csv(OUT/"w242b_gap.csv",index=False,encoding="utf-8-sig")

print("\n② 仓位敏感性 (全买 / gap>=-5%)")
for lab,(lo,hi) in (("全买",(None,None)),("gap>=-5%",(-5,None))):
    m=pd.Series(True,index=P.index)
    if lo is not None: m&=P.n1_gap>=lo
    q=P[m]; v=pd.Series((q.y-COST).values,index=q.index)
    print(f"  {lab}:")
    for w in (0.3,0.5,0.7,1.0):
        sim=base.copy(); sim[v.index]+=v.values*w; a,dd,cm=M(sim)
        print(f"    {w:>4.0%}: 年化{a:6.2f} 回撤{dd:6.2f} 卡玛{cm:.2f} Δ{cm-b[2]:+.2f}")
print("\n③ 分年 (全买 满仓)")
v=pd.Series((P.y-COST).values,index=P.index); p=v.reset_index(); p.columns=["signal","y"]
p["yr"]=p.signal//10000
g=p.groupby("yr").y.agg(["size","sum","mean"]); g["win"]=p.groupby("yr").y.apply(lambda x:(x>0).mean()*100).round(0)
print(g.round(2).to_string())
print(f"合计 {len(p)} 次 单笔{p.y.mean():+.3f}% 胜率{(p.y>0).mean()*100:.1f}%")
run=mx=0
for y in p.y:
    run=run+1 if y<0 else 0; mx=max(mx,run)
print(f"最长连亏 {mx} 次")
p.to_csv(OUT/"w242b_picks.csv",index=False,encoding="utf-8-sig")
