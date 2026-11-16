"""第243轮 K: 定稿 —— "距前250日最高(不含当日)"的仓位表与稳健性

243J 的窗口曲线给出平台: 60~300 日卡玛 4.2~5.4, 三段全正。取 250 日, 因为它
① 回撤最低(-29.65, 与 300 同) ② 第三段最强(+7.08) ③ 窗口长, 不受短期噪声影响。
本脚本: ① 仓位表 ② 分年 ③ 换选股键的稳健性 (市值/成交额同分时谁优先) ④ 每日名单导出
"""
from pathlib import Path
import gc
import numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[1]; PANEL=ROOT/"data"/"panel"; FEAT=ROOT/"output"/"feat234"; OUT=ROOT/"output"
COST=0.10
COLS=["ts_code","trade_date","y","n1_ok","pct","idx_dist","dist_high60","amt_yi","circ_mv",
      "is_paused","is_st","listed"]

def M(r):
    nav=(1+r/100).cumprod(); ann=nav.iloc[-1]**(244/len(r))-1; dd=(nav/nav.cummax()-1).min()
    return round(ann*100,2),round(dd*100,2),round(ann*100/abs(dd*100),2)

def main():
    parts=[]
    for p in sorted(FEAT.glob("f*.parquet")):
        d=pd.read_parquet(p,columns=COLS)
        d=d[(d.is_paused==0)&(d.is_st==0)&(d.listed>=250)]
        d=d[d.pct.between(-11.5,11.5)&d.y.between(-40,60)]
        d=d[d.ts_code.str[:2].isin(["60","00"])]
        d=d[d.n1_ok==True]
        if len(d):
            d["trade_date"]=d["trade_date"].astype("int64"); parts.append(d)
    d=pd.concat(parts,ignore_index=True); del parts; gc.collect()
    tr=pd.read_csv(OUT/"w232_trades.csv")
    cal=np.sort(d.trade_date.unique()); cal=cal[cal>=tr.signal.min()]
    tr=tr[tr.signal.isin(cal)]
    busy=set()
    for _,r in tr.iterrows(): busy.add(r.buy); busy.add(r.sell)
    sig=set(tr.signal); free=set(x for x in cal if x not in busy and x not in sig)
    base=pd.Series(0.0,index=cal)
    for dt,g in tr.groupby("signal"): base[dt]=g["ret"].mean()
    b=M(base); print(f"基准 年化{b[0]} 回撤{b[1]} 卡玛{b[2]}")
    nxt={cal[i]:cal[i+1] for i in range(len(cal)-1)}; nxt[cal[-1]]=cal[-1]
    days=[]
    for dt in sorted(d[(d.idx_dist<=-3.51)&(d.dist_high60<=-24.46)].trade_date.unique()):
        if dt not in free: continue
        bd=nxt.get(dt); sd=nxt.get(bd,bd)
        if bd in busy or sd in busy: continue
        days.append(dt)
    keep=d[d.trade_date.isin(days)][["ts_code","trade_date","y","amt_yi","circ_mv"]].copy()
    tkey=set(zip(keep.ts_code,keep.trade_date)); del d; gc.collect()
    print(f"补位日 {len(days)} 候选 {len(keep):,}")

    # 只算 250 日窗口
    px_parts=[]
    for p in sorted(PANEL.glob("*.csv.gz")):
        try: x=pd.read_csv(p,usecols=["ts_code","trade_date","high","close","factor"])
        except Exception: continue
        if not len(x): continue
        f=x["factor"].fillna(1.0)
        px_parts.append(pd.DataFrame({"ts_code":x["ts_code"].astype("category"),
            "trade_date":x["trade_date"].astype("int64"),
            "ah":(x["high"]*f).astype("float32"),"ac":(x["close"]*f).astype("float32")}))
    px=pd.concat(px_parts,ignore_index=True); del px_parts; gc.collect()
    px=px.sort_values(["ts_code","trade_date"],kind="mergesort").reset_index(drop=True)
    hm=px.groupby("ts_code",sort=False)["ah"].transform(
        lambda s: s.shift(1).rolling(250,min_periods=80).max())
    px["dh"]=((px["ac"]/hm-1)*100).astype("float32"); del hm; gc.collect()
    px=px[["ts_code","trade_date","dh"]].dropna()
    px["ts_code"]=px["ts_code"].astype(str)
    px=px[[k in tkey for k in zip(px.ts_code,px.trade_date)]]
    print(f"250日窗口可用 {len(px):,} 行")
    m=keep.merge(px,on=["ts_code","trade_date"],how="left"); del px; gc.collect()

    def pick(sec_key, asc_small=True):
        v={}
        for dt,g in m.groupby("trade_date"):
            g=g.dropna(subset=["dh"])
            if not len(g): continue
            # 主键 dh 最远; 同分/近似取成交额大的保证买得到
            g=g.assign(k=g["dh"]*1000-g["amt_yi"]/1e4)
            g=g.sort_values("k")
            v[dt]=g.iloc[0]["y"]-COST
        return pd.Series(v)
    v=pick(None)
    print(f"\n① 250日 满仓: {len(v)} 次 单笔{v.mean():+.3f}% 胜率{(v>0).mean()*100:.1f}%")
    print("\n仓位表:")
    rows=[]
    for w in (0.2,0.3,0.4,0.5,0.7,0.85,1.0):
        sim=base.copy(); sim[v.index]+=v.values*w
        a,dd,cm=M(sim)
        rows.append({"仓位":f"{w:.0%}","年化":a,"回撤":dd,"卡玛":cm,"Δ":round(cm-b[2],2)})
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n② 分年 (满仓)")
    p=pd.DataFrame({"yr":v.index//10000,"y":v.values})
    g=p.groupby("yr").y.agg(["size","sum","mean"]); g["win%"]=p.groupby("yr").y.apply(lambda x:(x>0).mean()*100).round(0)
    print(g.round(2).to_string())
    run=mx=0
    for x in v.values:
        run=run+1 if x<0 else 0; mx=max(mx,run)
    vv=v.sort_values(ascending=False); tot=vv.sum()
    print(f"合计 {len(v)} 次 单笔{v.mean():+.3f}% 最长连亏{mx} 最差{v.min():.1f}% 前5笔占总{vv[:5].sum()/tot*100:.0f}%")

    print("\n③ 同分优先键的稳健性 (主键 dh250 不变, 只换破同分的次序)")
    rows=[]
    for tag,col,sgn in (("成交额最大","amt_yi",-1),("成交额最小","amt_yi",1),
                        ("市值最小","circ_mv",1),("市值最大","circ_mv",-1)):
        vv2={}
        for dt,gg in m.groupby("trade_date"):
            gg=gg.dropna(subset=["dh"])
            if not len(gg): continue
            gg=gg.assign(k=gg["dh"]*1000+sgn*gg[col]/1e4)
            vv2[dt]=gg.sort_values("k").iloc[0]["y"]-COST
        s2=pd.Series(vv2); sim=base.copy(); sim[s2.index]+=s2.values
        a,dd,cm=M(sim)
        rows.append({"优先":tag,"单笔":round(s2.mean(),3),"年化":a,"回撤":dd,"卡玛":cm,"Δ":round(cm-b[2],2)})
    print(pd.DataFrame(rows).to_string(index=False))

    # 导出名单
    rec=[]
    for dt,gg in m.groupby("trade_date"):
        gg=gg.dropna(subset=["dh"])
        if not len(gg): continue
        gg=gg.assign(k=gg["dh"]*1000-gg["amt_yi"]/1e4).sort_values("k")
        r=gg.iloc[0]
        rec.append({"signal":dt,"code":r.ts_code,"dh250":round(float(r.dh),2),
                    "y":round(float(r.y-COST),3)})
    P=pd.DataFrame(rec).sort_values("signal")
    P.to_csv(OUT/"w243k_picks.csv",index=False,encoding="utf-8-sig")
    print(f"\n名单已存 {len(P)} 行")

if __name__=="__main__": main()
