"""第二百四十轮 C: 把"大盘破位日补位"做成一条可落地的规则

240B 的结论: 宽覆盖条件(W1/W3, 覆盖 40% 空仓日)年化翻倍但回撤也翻倍, 卡玛反降
—— 那是变相加杠杆。真正提升卡玛的是"大盘破位日"这一类窄条件:
  W6 20日弱+大盘破位   卡玛 8.43  回撤 -23.59
  W7 距60日高最远+破位 卡玛 9.62  回撤 -23.59   (基准 6.63)
本轮把它们并成一条规则, 查: ① 去重后多少次 ② 分年 ③ 与 B1 合并 ④ 仓位敏感性
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "output" / "feat234"
OUT = ROOT / "output"
COST = 0.10
COLS = ["ts_code","trade_date","y","n1_ok","pct","rel_pct","vol20","is_paused","is_st",
        "listed","circ_mv","dist_high60","rs20","idx_dist","dn_streak","amp"]

def load():
    parts=[]
    for p in sorted(FEAT.glob("f*.parquet")):
        d=pd.read_parquet(p,columns=COLS)
        d=d[(d.is_paused==0)&(d.is_st==0)&(d.n1_ok==True)&d.pct.between(-11.5,11.5)&d.y.between(-40,60)&(d.listed>=250)]
        d=d[d.ts_code.str[:2].isin(["60","00"])]   # 3万本金只能买主板: 创业板/科创板/北交所都有门槛
        if len(d): parts.append(d)
    return pd.concat(parts,ignore_index=True)

def M(r):
    nav=(1+r/100).cumprod(); yrs=len(r)/244
    ann=nav.iloc[-1]**(1/yrs)-1; dd=(nav/nav.cummax()-1).min()
    return round(ann*100,2),round(dd*100,2),round(ann*100/abs(dd*100),2)

def main():
    d=load()
    d["vol_x_pct"]=d.vol20*d.pct
    d["dn_x_amp"]=d.dn_streak*d.amp
    cal=np.sort(d.trade_date.unique()); cal=cal[cal>=20160101]
    tr=pd.read_csv(OUT/"w232_trades.csv"); tr=tr[tr.signal.isin(cal)]
    busy,sig=set(),set(tr.signal)
    for _,r in tr.iterrows(): busy.add(r.buy); busy.add(r.sell)
    free=[x for x in cal if x not in busy and x not in sig]
    d=d[d.trade_date.isin(free)].copy()

    RULES={
      "W6 20日弱+破位":   lambda x:(x.rs20<-7.41)&(x.idx_dist<-3.51),
      "W7 距60高最远+破位":lambda x:(x.dist_high60<-24.46)&(x.idx_dist<-3.51),
      "W5 抗跌+跌停带量":  lambda x:(x.rel_pct>1.24)&(x.dn_x_amp>5.47),
      "B1 抗跌+高波动杀":  lambda x:(x.rel_pct>1.26)&(x.vol_x_pct<-4.37),
    }
    U={k:np.zeros(len(d),bool) for k in RULES}
    for k,f in RULES.items(): U[k]=f(d)

    tcal=[x for x in cal if x>=20210104]
    base=pd.Series(0.0,index=tcal)
    for dt,g in tr[tr.signal>=20210104].groupby("signal"): base[dt]=g["ret"].mean()
    b=M(base); print(f"测试段 {tcal[0]}~{tcal[-1]} {len(tcal)}日  基准 {b}")

    COMBOS={"W6":["W6 20日弱+破位"],"W7":["W7 距60高最远+破位"],
            "W6+W7":["W6 20日弱+破位","W7 距60高最远+破位"],
            "W6+W7+W5":["W6 20日弱+破位","W7 距60高最远+破位","W5 抗跌+跌停带量"],
            "全部4条":list(RULES)}
    print("\n%s"%"="*78)
    rows=[]
    for cn,ks in COMBOS.items():
        m=np.zeros(len(d),bool)
        for k in ks: m|=U[k]
        s=d[m]; s=s[s.trade_date>=20210104]
        for k,tag in ((1,"1只"),(3,"3只")):
            v={}
            for dt,g in s.groupby("trade_date"):
                v[dt]=g.nsmallest(min(k,len(g)),"circ_mv")["y"].mean()-COST
            v=pd.Series(v); sim=base.copy(); sim[v.index]=v.values
            a,dd,cm=M(sim)
            rows.append({"组合":cn,"选股":tag,"笔数":len(v),"覆盖率%":round(len(v)/len(tcal)*100,0),
                         "单笔":round(v.mean(),3),"年化":a,"回撤":dd,"卡玛":cm,
                         "比基准":round(cm-b[2],2)})
    r=pd.DataFrame(rows); print(r.to_string(index=False))

    # 分年: 用 W6+W7 3只
    m=U["W6 20日弱+破位"]|U["W7 距60高最远+破位"]
    s=d[m]; s=s[s.trade_date>=20210104]
    v={dt:g.nsmallest(min(3,len(g)),"circ_mv")["y"].mean()-COST for dt,g in s.groupby("trade_date")}
    v=pd.Series(v); p=v.reset_index(); p.columns=["signal","y"]; p["yr"]=p.signal//10000
    print("\n分年(W6+W7, 3只):")
    g=p.groupby("yr").y.agg(["size","sum","mean","min"])
    g["win"]=p.groupby("yr").y.apply(lambda x:(x>0).mean()*100).round(0)
    print(g.round(2).to_string())
    print(f"\n合计 {len(p)} 笔 单笔 {p.y.mean():+.3f}% 胜率 {(p.y>0).mean()*100:.1f}% 最差 {p.y.min():.2f}%")
    n=p.groupby("yr").size()
    print("每年次数:",dict(n))

    print("\n仓位敏感性(W6+W7 3只):")
    for w in (0.3,0.5,0.7,1.0):
        sim=base.copy(); sim[v.index]+=v.values*w
        a,dd,cm=M(sim); print(f"  {w:.0%}: 年化{a}% 回撤{dd}% 卡玛{cm}")
    p.to_csv(OUT/"w240c_picks.csv",index=False,encoding="utf-8-sig")

if __name__=="__main__": main()
