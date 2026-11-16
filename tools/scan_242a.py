"""第242轮: 单吊约束下重做补位 —— 选哪一只 + 什么价才买

241 轮的分散方案被否, 但 241 发现了一个被埋掉的线索:
破位日 7 个选股键里, 只有"距60日高"的 y 与 y_ex 同时单调向好
  Q1(跌得最透) y +1.44 / y_ex +0.29   Q5(最抗跌) y +0.49 / y_ex -0.17
而 240 轮选的是"市值最小", 那是另一个方向。
本脚本: 单吊满仓, 逐个选股键取 1 只, 再用 n1_gap(次日竞价缺口)做入场过滤。
"""
from pathlib import Path
import numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[1]; FEAT=ROOT/"output"/"feat234"; OUT=ROOT/"output"
COST=0.10
COLS=["ts_code","trade_date","y","n1_ok","n1_gap","pct","circ_mv","idx_dist","dist_high60",
      "amt_yi","vol20","amp","turnover_rate","close_pos","rel_pct","is_paused","is_st","listed"]
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
b=M(base); print(f"基准 年化{b[0]}% 回撤{b[1]}% 卡玛{b[2]}")
nxt={cal[i]:cal[i+1] for i in range(len(cal)-1)}; nxt[cal[-1]]=cal[-1]
s=d[(d.idx_dist<=-3.51)&(d.dist_high60<=-24.46)]
days=[]
for dt in sorted(s.trade_date.unique()):
    if dt not in free: continue
    bd=nxt.get(dt); sd=nxt.get(bd,bd)
    if bd in busy or sd in busy: continue
    days.append(dt)
sub={dt:g for dt,g in d[d.trade_date.isin(days)].groupby("trade_date")}
print(f"补位日 {len(days)}\n")
KEYS=[("dist_high60",True,"距60日高最远(跌最透)"),("dist_high60",False,"距60日高最近(最抗跌)"),
      ("circ_mv",True,"市值最小"),("circ_mv",False,"市值最大"),
      ("amp",False,"振幅最大"),("vol20",False,"波动最大"),
      ("close_pos",False,"收盘位置最高"),("close_pos",True,"收盘位置最低"),
      ("rel_pct",False,"最抗跌(超额最高)"),("rel_pct",True,"跌最狠(超额最低)"),
      ("amt_yi",True,"成交额最小"),("amt_yi",False,"成交额最大")]
rows=[]
for key,asc,lab in KEYS:
    v={}
    for dt,g in sub.items():
        g2=g.dropna(subset=[key])
        if not len(g2): continue
        v[dt]=(g2.nsmallest(1,key) if asc else g2.nlargest(1,key))["y"].iloc[0]-COST
    v=pd.Series(v); sim=base.copy(); sim[v.index]+=v.values
    a,dd,cm=M(sim)
    rows.append({"选股键":lab,"单笔":round(v.mean(),3),"胜率%":round((v>0).mean()*100,1),
                 "最差":round(v.min(),1),"年化":a,"回撤":dd,"卡玛":cm,"Δ卡玛":round(cm-b[2],2)})
r=pd.DataFrame(rows).sort_values("卡玛",ascending=False)
print("① 单吊满仓, 逐个选股键取 1 只")
print(r.to_string(index=False))
r.to_csv(OUT/"w242a_keys.csv",index=False,encoding="utf-8-sig")
