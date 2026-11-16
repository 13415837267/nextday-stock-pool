"""第243轮 G: 恐慌的"广度"而不是"幅度" —— 跌停家数是否决定补位成败

已排除: 破位深度(-3.5~-8)分档是倒U型且段间翻转, 不能用来定仓位。
新维度: 同一天指数跌幅相近, "跌停 20 家"和"跌停 200 家"是两种完全不同的日子。
       广度(limit_dn / breadth)反映流动性危机程度, 有机制解释:
       跌停潮 = 卖盘无法成交 = 次日反弹可能更强, 但也可能连续跌停(尾部)。
本脚本测: 跌停家数分档 x 补位单笔, 看是否单调; 若是, 用它做仓位或干脆排除。
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
mk=pd.read_csv(FEAT/"market.csv").set_index("trade_date")
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
    m=mk.loc[dt]
    rec.append({"signal":dt,"y":r.y-COST,"idx":r.idx_dist,"code":r.ts_code,
                "dn":float(m["limit_dn"]),"br":float(m["breadth"]),"med":float(m["med_pct"])})
Q=pd.DataFrame(rec)
print(f"补位 {len(Q)} 次  跌停家数: {Q.dn.describe().round(0).to_dict()}\n")
print("① 跌停家数分档")
Q["b"]=pd.cut(Q.dn,[-1,5,15,40,100,10000],labels=["<=5","6-15","16-40","41-100",">100"])
g=Q.groupby("b",observed=True).agg(n=("y","size"),mean=("y","mean"),med=("y","median"),
                                   worst=("y","min"),best=("y","max"))
g["win%"]=Q.groupby("b",observed=True).y.apply(lambda x:(x>0).mean()*100).round(0)
g["std"]=Q.groupby("b",observed=True).y.std().round(2)
print(g.round(2).to_string())
t=Q.copy(); t["yr"]=t.signal//10000
print("\n  分年 各档单笔:")
print(t.pivot_table(index="yr",columns="b",values="y",aggfunc="mean",observed=True).round(2).to_string())
print("\n  分段:")
for y0,y1 in ((2016,2019),(2020,2023),(2024,2026)):
    tt=Q[(Q.signal//10000>=y0)&(Q.signal//10000<=y1)]
    gg=tt.groupby("b",observed=True).y.agg(["size","mean"]).round(2)
    print(f"  {y0}-{y1}: {gg['mean'].to_dict()}")
print("\n② breadth 分档 (上涨家数占比)")
Q["b2"]=pd.cut(Q.br,[-1,0.05,0.12,0.22,0.35,1],labels=["<=5%","5-12%","12-22%","22-35%",">35%"])
g2=Q.groupby("b2",observed=True).agg(n=("y","size"),mean=("y","mean"),worst=("y","min"))
g2["win%"]=Q.groupby("b2",observed=True).y.apply(lambda x:(x>0).mean()*100).round(0)
print(g2.round(2).to_string())
print("\n③ 先看相关性")
print(Q[["y","idx","dn","br","med"]].corr()["y"].round(3).to_string())
Q.to_csv(OUT/"w243g_picks.csv",index=False,encoding="utf-8-sig")
