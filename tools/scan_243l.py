"""第243轮 L: 同一只票的冷却期 —— 243K 暴露 192 次只用了 75 只票, 最长连续 12 次

"距 250 日高最远"天然会反复选中同一只持续下跌的票。002676.SZ 出现 13 次合计 -26pp。
对 3 万本金实盘, 连续 12 天买同一只在跌的票是不可接受的。
测: 该票在最近 N 个交易日内买过则跳过, 取次远的票。看卡玛损失多少。
"""
from pathlib import Path
import gc
import numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[1]; PANEL=ROOT/"data"/"panel"; FEAT=ROOT/"output"/"feat234"; OUT=ROOT/"output"
COST=0.10
COLS=["ts_code","trade_date","y","n1_ok","pct","idx_dist","dist_high60","amt_yi","is_paused","is_st","listed"]

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
    keep=d[d.trade_date.isin(days)][["ts_code","trade_date","y","amt_yi"]].copy()
    tkey=set(zip(keep.ts_code,keep.trade_date)); del d; gc.collect()

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
    px=px[["ts_code","trade_date","dh"]].dropna(); px["ts_code"]=px["ts_code"].astype(str)
    px=px[[k in tkey for k in zip(px.ts_code,px.trade_date)]]
    m=keep.merge(px,on=["ts_code","trade_date"],how="left"); del px; gc.collect()
    m["k"]=m["dh"]*1000-m["amt_yi"]/1e4
    m=m.sort_values(["trade_date","k"])
    byday={dt:g.dropna(subset=["dh"]) for dt,g in m.groupby("trade_date")}
    order=sorted(byday)
    pos={dt:i for i,dt in enumerate(order)}

    print(f"\n{'冷却':>6}{'次数':>6}{'单笔':>8}{'年化':>9}{'回撤':>9}{'卡玛':>7}{'Δ':>7}{'票数':>6}{'最长连续':>9}{'前5票占比':>10}")
    rows=[]
    for cd in (0,5,10,20,40,60):
        last={}   # code -> 上次买入的交易日序号
        v={}; pick={}
        for dt in order:
            g=byday[dt]
            if not len(g): continue
            cand=g
            if cd>0:
                ok=[(code not in last) or (pos[dt]-last[code]>=cd) for code in g.ts_code]
                cand=g[np.array(ok)]
                if not len(cand): cand=g    # 全被冷却则退回, 宁可重复也不能空仓
            r=cand.iloc[0]
            v[dt]=r["y"]-COST; pick[dt]=r.ts_code
            last[r.ts_code]=pos[dt]
        v=pd.Series(v); sim=base.copy(); sim[v.index]+=v.values
        a,dd,cm=M(sim)
        pk=pd.Series(pick); vc=pk.value_counts()
        run=mx=0; prev=None
        for c in pk.sort_index().values:
            run=run+1 if c==prev else 1; prev=c; mx=max(mx,run)
        top5=vc.head(5).sum()/len(pk)*100
        rows.append({"冷却":cd,"次数":len(v),"单笔":round(v.mean(),3),"年化":a,"回撤":dd,
                     "卡玛":cm,"Δ":round(cm-b[2],2),"票数":len(vc),"最长连续":mx,"前5票%":round(top5,0)})
        print(f"{cd:>6}{len(v):>6}{v.mean():>8.3f}{a:>9.2f}{dd:>9.2f}{cm:>7.2f}{cm-b[2]:>+7.2f}{len(vc):>6}{mx:>9}{top5:>9.0f}%")
        if cd==20:
            v20=v
    pd.DataFrame(rows).to_csv(OUT/"w243l_cool.csv",index=False,encoding="utf-8-sig")

    print("\n冷却 20 日 x 仓位:")
    for w0 in (0.3,0.5,0.7,0.85,1.0):
        sim=base.copy(); sim[v20.index]+=v20.values*w0
        a,dd,cm=M(sim)
        print(f"  {w0:>4.0%}: 年化{a:7.2f} 回撤{dd:7.2f} 卡玛{cm:.2f}  Δ{cm-b[2]:+.2f}")
    pd.DataFrame({"signal":v20.index,"y":v20.values}).to_csv(OUT/"w243l_picks20.csv",index=False,encoding="utf-8-sig")

    print("\n冷却 20 日 分年:")
    p=pd.DataFrame({"yr":v20.index//10000,"y":v20.values})
    g=p.groupby("yr").y.agg(["size","sum","mean"]); g["win%"]=p.groupby("yr").y.apply(lambda x:(x>0).mean()*100).round(0)
    print(g.round(2).to_string())

if __name__=="__main__": main()
