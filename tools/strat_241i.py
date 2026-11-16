"""第241轮I: 确定性替代"随机20只" —— 市值分层抽样, 每层取1只"""
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
    return ann*100, dd*100, ann*100/abs(dd*100)
cal=np.sort(d.trade_date.unique()); tr=pd.read_csv(OUT/"w232_trades.csv")
cal=cal[cal>=tr.signal.min()]; tr=tr[tr.signal.isin(cal)]
busy=set()
for _,r in tr.iterrows(): busy.add(r.buy); busy.add(r.sell)
sig=set(tr.signal); free=set(x for x in cal if x not in busy and x not in sig)
base=pd.Series(0.0,index=cal)
for dt,g in tr.groupby("signal"): base[dt]=g["ret"].mean()
b=M(base); print(f"基准 {b[0]:.1f}%/{b[1]:.2f}%/{b[2]:.2f}")
nxt={cal[i]:cal[i+1] for i in range(len(cal)-1)}; nxt[cal[-1]]=cal[-1]
s=d[(d.idx_dist<=-3.51)&(d.dist_high60<=-24.46)]
days=[]
for dt in sorted(s.trade_date.unique()):
    if dt not in free: continue
    bd=nxt.get(dt); sd=nxt.get(bd,bd)
    if bd in busy or sd in busy: continue
    days.append(dt)
sub={dt:g for dt,g in d[d.trade_date.isin(days)].groupby("trade_date")}
print(f"补位日 {len(days)}")
def strat(n_groups, pick_per=1, key="circ_mv"):
    """按 key 等频分 n_groups 层, 每层取 pick_per 只(层内取成交额最大的, 保证可买)"""
    v={}
    for dt,g in sub.items():
        g2=g.dropna(subset=[key]).sort_values(key)
        if len(g2)<n_groups*pick_per: 
            v[dt]=g2["y"].mean()-COST; continue
        sel=[]
        for _, gg in g2.groupby(pd.qcut(g2[key].rank(method="first"), n_groups, labels=False)):
            sel.append(gg.nlargest(pick_per,"amt_yi")["y"].values)
        v[dt]=np.concatenate(sel).mean()-COST
    v=pd.Series(v); sim=base.copy(); sim[v.index]+=v.values
    a,dd,cm=M(sim); return a,dd,cm,v.mean(),v.min(),v
print("\n市值分层抽样 (每层取成交额最大的1只):")
for ng in (5,10,15,20,30,40,60):
    a,dd,cm,mn,mi,_=strat(ng)
    print(f"  {ng:>2}层({ng}只): 单笔{mn:+.3f}% 最差{mi:.1f}% 年化{a:6.1f} 回撤{dd:6.2f} 卡玛{cm:.2f} Δ{cm-b[2]:+.2f}")
a,dd,cm,mn,mi,v20=strat(20,2)
print(f"\n20层 每层2只(40只): ", end="")
a2,dd2,cm2,mn2,mi2,_=strat(20,2)
print(f"单笔{mn2:+.3f}% 年化{a2:.1f} 回撤{dd2:.2f} 卡玛{cm2:.2f} Δ{cm2-b[2]:+.2f}")
print("\n仓位敏感性 (20层20只):")
for w in (0.3,0.5,0.7,1.0):
    sim=base.copy(); sim[v20.index]+=v20.values*w; a3,dd3,cm3=M(sim)
    print(f"  {w:>4.0%}: 年化{a3:6.1f} 回撤{dd3:6.2f} 卡玛{cm3:.2f} Δ{cm3-b[2]:+.2f}")
p=v20.reset_index(); p.columns=["signal","y"]; p["yr"]=p.signal//10000
print("\n分年(20层满仓):")
g=p.groupby("yr").y.agg(["size","sum","mean"]); g["win"]=p.groupby("yr").y.apply(lambda x:(x>0).mean()*100).round(0)
print(g.round(2).to_string())
p.to_csv(OUT/"w241i_picks.csv",index=False,encoding="utf-8-sig")
