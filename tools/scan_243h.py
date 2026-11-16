"""第243轮 H: 补位日能否直接用主策略评分选股 —— 让两套规则合成一套

主策略评分 = 0.20*z(mom20) + 0.70*z(-bias20) + 0.05*z(-amount) + 0.03*z(low_tail)
核心是 -bias20(70%) = 买 20 日乖离最负的票。
补位用的是 dist_high60(距 60 日高最远), 窗口不同。
若两者等价 -> 可以统一成一套逻辑, 少维护一份规则;
若 dist_high60 明显更好 -> 说明 60 日窗口捕捉到了 20 日看不到的东西, 保留。
"""
from pathlib import Path
import numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[1]; FEAT=ROOT/"output"/"feat234"; OUT=ROOT/"output"
COST=0.10
COLS=["ts_code","trade_date","y","n1_ok","pct","idx_dist","dist_high60","dist_high20","bias20",
      "bias5","bias10","bias60","mom20" if False else "ret20","amt_yi","vol20","low_tail",
      "is_paused","is_st","listed"]
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
sub={dt:g for dt,g in d[d.trade_date.isin(days)].groupby("trade_date")}
print(f"补位日 {len(days)} 次, 中位票数 {int(np.median([len(g) for g in sub.values()]))}\n")
def zrank(g,col,asc):
    """日内 zscore 排序值(与 selector 同口径: 截面 z); NaN 记 0(等于截面均值)"""
    v=g[col].astype(float)
    z=(v-v.mean())/(v.std(ddof=0)+1e-9)
    z=z.fillna(0.0)
    return -z if asc else z
KEYS=[("dist_high60",True,"距60日高最远 (242现行)"),
      ("dist_high20",True,"距20日高最远"),
      ("bias20",True,"20日乖离最负 (评分核心70%)"),
      ("bias5",True,"5日乖离最负"),("bias10",True,"10日乖离最负"),("bias60",True,"60日乖离最负"),
      ("ret20",True,"20日跌幅最大"),
      ("__score__",True,"主策略完整评分(4因子加权)")]
rows=[]
for col,asc,lab in KEYS:
    v={}
    for dt,g in sub.items():
        if col=="__score__":
            sc=(0.20*zrank(g,"ret20",False)+0.70*zrank(g,"bias20",True)
                +0.05*zrank(g,"amt_yi",True)+0.03*zrank(g,"low_tail",False))
            i=sc.idxmax()
        else:
            g2=g.dropna(subset=[col]).sort_values([col,"amt_yi"],ascending=[asc,False])
            if not len(g2): continue
            i=g2.index[0]
        v[dt]=g.loc[i,"y"]-COST
    v=pd.Series(v); sim=base.copy(); sim[v.index]+=v.values
    a,dd,cm=M(sim)
    segs=[]
    for y0,y1 in ((2016,2019),(2020,2023),(2024,2026)):
        m=(sim.index//10000>=y0)&(sim.index//10000<=y1)
        _,_,c=M(sim[m]); _,_,cb=M(base[m]); segs.append(round(c-cb,2))
    rows.append({"排序键":lab,"次数":len(v),"单笔":round(v.mean(),3),
                 "胜率%":round((v>0).mean()*100,1),"最差":round(v.min(),1),
                 "年化":a,"回撤":dd,"卡玛":cm,"Δ":round(cm-b[2],2),"三段":segs})
r=pd.DataFrame(rows).sort_values("卡玛",ascending=False)
pd.set_option("display.width",220)
print(r.to_string(index=False))
r.to_csv(OUT/"w243h_keys.csv",index=False,encoding="utf-8-sig")

# 重叠度: 现行 vs 主策略评分, 选到同一只的比例
same=0; tot=0
for dt,g in sub.items():
    a1=g.dropna(subset=["dist_high60"]).sort_values(["dist_high60","amt_yi"],ascending=[True,False])
    if not len(a1): continue
    c1=a1.iloc[0].ts_code
    sc=(0.20*zrank(g,"ret20",False)+0.70*zrank(g,"bias20",True)
        +0.05*zrank(g,"amt_yi",True)+0.03*zrank(g,"low_tail",False))
    c2=g.loc[sc.idxmax(),"ts_code"]
    tot+=1; same+= (c1==c2)
print(f"\n两套规则选到同一只: {same}/{tot} = {same/tot*100:.0f}%")
