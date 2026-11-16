"""第241轮H: 破位日的选股键 —— 成交额(大盘)最差, 波动/振幅才是正解"""
from pathlib import Path
import numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[1]; FEAT=ROOT/"output"/"feat234"; OUT=ROOT/"output"
COST=0.10
COLS=["ts_code","trade_date","y","n1_ok","pct","circ_mv","idx_dist","dist_high60","amt_yi",
      "vol20","amp","turnover_rate","close_pos","is_paused","is_st","listed"]
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
def ev(key, asc, k, w=1.0):
    v={}
    for dt,g in sub.items():
        g2=g.dropna(subset=[key])
        if not len(g2): continue
        sel=g2.nsmallest(min(k,len(g2)),key) if asc else g2.nlargest(min(k,len(g2)),key)
        v[dt]=sel["y"].mean()-COST
    v=pd.Series(v); sim=base.copy(); sim[v.index]+=v.values*w
    a,dd,cm=M(sim); return a,dd,cm,v.mean(),v.min()
print("\n单键 x 只数 (满仓):")
rows=[]
for key,asc,lab in [("vol20",False,"20日波动最高"),("vol20",True,"20日波动最低"),
                    ("amp",False,"振幅最高"),("amp",True,"振幅最低"),
                    ("circ_mv",True,"市值最小"),("circ_mv",False,"市值最大"),
                    ("amt_yi",False,"成交额最大"),("amt_yi",True,"成交额最小"),
                    ("turnover_rate",False,"换手最高")]:
    for k in (5,10,20):
        a,dd,cm,mn,mi=ev(key,asc,k)
        rows.append({"选股键":lab,"只数":k,"单笔":round(mn,3),"最差":round(mi,1),
                     "年化":round(a,1),"回撤":round(dd,2),"卡玛":round(cm,2),"Δ卡玛":round(cm-b[2],2)})
r=pd.DataFrame(rows); print(r.to_string(index=False))
r.to_csv(OUT/"w241h_keys.csv",index=False,encoding="utf-8-sig")
print("\n只数敏感性 (20日波动最高):")
for k in (3,5,8,10,15,20,30):
    a,dd,cm,mn,mi=ev("vol20",False,k)
    print(f"  {k:>2}只: 单笔{mn:+.3f}% 最差{mi:.1f}% 年化{a:6.1f} 回撤{dd:6.2f} 卡玛{cm:.2f}")
