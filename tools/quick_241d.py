"""第241轮D: 把补位仓位做成超跌深度的函数 + 与主策略风控日的重叠度"""
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
busy=set(); [ (busy.add(r.buy),busy.add(r.sell)) for _,r in tr.iterrows() ]
sig=set(tr.signal); free=[x for x in cal if x not in busy and x not in sig]
base=pd.Series(0.0,index=cal)
for dt,g in tr.groupby("signal"): base[dt]=g["ret"].mean()
b=M(base); print(f"基准 {b}")
nxt={cal[i]:cal[i+1] for i in range(len(cal)-1)}; nxt[cal[-1]]=cal[-1]
s=d[(d.idx_dist<=-3.51)&(d.dist_high60<=-24.46)]
sub=s[s.trade_date.isin(free)]
v3={dt:g.nlargest(min(10,len(g)),"amt_yi")["y"].mean()-COST for dt,g in sub.groupby("trade_date")}
v4={dt:g["y"].mean()-COST for dt,g in sub.groupby("trade_date")}
depth=sub.groupby("trade_date")["idx_dist"].first()
print(f"补位日 {len(v3)}")
print("\n=== 固定仓位 (S3 流动性10只 / S4 全等权)")
for tag,v in (("S3",v3),("S4",v4)):
    v=pd.Series(v)
    for w in (0.3,0.5,0.7,1.0):
        sim=base.copy(); sim[v.index]+=v.values*w; a,dd,cm=M(sim)
        print(f"  {tag} {w:>4.0%}: 年化{a} 回撤{dd} 卡玛{cm}")
print("\n=== 深度分档仓位")
def wf(x):  # 超跌越深仓位越高
    return 1.0 if x<=-8 else (0.6 if x<=-6 else (0.4 if x<=-4.5 else 0.25))
for tag,v in (("S3",v3),("S4",v4)):
    v=pd.Series(v); w=pd.Series({k:wf(depth[k]) for k in v.index})
    sim=base.copy(); sim[v.index]+=v.values*w.values; a,dd,cm=M(sim)
    print(f"  {tag} 深度分档: 年化{a} 回撤{dd} 卡玛{cm}  平均仓位{w.mean():.0%}")
    print("     仓位分布", w.round(2).value_counts().sort_index().to_dict())
print("\n=== 补位日 vs 主策略风控日重叠")
mk=pd.read_csv(OUT/"feat234"/"market.csv")
mk=mk.set_index("trade_date") if "trade_date" in mk.columns else mk
mk.columns=[c.lower() for c in mk.columns]
cal_s=set(sub.trade_date.unique())
if "idx_ma20_dist" in mk.columns:
    bad=set(mk[(mk.idx_ma20_dist<=0)].index)
    print(f"补位日中 指数在20日线下方 {(len(cal_s&bad))}/{len(cal_s)} = {len(cal_s&bad)/len(cal_s)*100:.0f}%")
print("补位日最集中的月份:", pd.Series([str(x)[:6] for x in sub.trade_date.unique()]).value_counts().head(5).to_dict())
