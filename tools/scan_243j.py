"""第243轮 J: 窗口曲线 —— 90/120/150/180/200/250/300/360 日, 看是否平滑高原

243I: 排序窗口 60 -> 120/250 日后 单笔 1.768 -> 1.861~2.089, 回撤 -45.2 -> -29.7,
卡玛 3.38 -> 5.08~5.39, 三段全正。但 60 -> 120 -> 250 -> ATH 是倒U, 必须确认平台够宽。
逐窗口扫, 每次只保留补位日需要的行(全史 rolling 但只落地子集), 避免 OOM。
"""
from pathlib import Path
import gc
import numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[1]; PANEL=ROOT/"data"/"panel"; FEAT=ROOT/"output"/"feat234"; OUT=ROOT/"output"
COST=0.10
WS=[60,90,120,150,180,200,250,300,360]
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
            d["trade_date"]=d["trade_date"].astype("int64")
            parts.append(d)
    d=pd.concat(parts,ignore_index=True); del parts; gc.collect()

    tr=pd.read_csv(OUT/"w232_trades.csv")
    cal=np.sort(d.trade_date.unique()); cal=cal[cal>=tr.signal.min()]
    tr=tr[tr.signal.isin(cal)]
    busy=set()
    for _,r in tr.iterrows(): busy.add(r.buy); busy.add(r.sell)
    sig=set(tr.signal); free=set(x for x in cal if x not in busy and x not in sig)
    base=pd.Series(0.0,index=cal)
    for dt,g in tr.groupby("signal"): base[dt]=g["ret"].mean()
    b=M(base); print(f"基准 年化{b[0]} 回撤{b[1]} 卡玛{b[2]}",flush=True)
    nxt={cal[i]:cal[i+1] for i in range(len(cal)-1)}; nxt[cal[-1]]=cal[-1]
    days=[]
    for dt in sorted(d[(d.idx_dist<=-3.51)&(d.dist_high60<=-24.46)].trade_date.unique()):
        if dt not in free: continue
        bd=nxt.get(dt); sd=nxt.get(bd,bd)
        if bd in busy or sd in busy: continue
        days.append(dt)
    keep=d[d.trade_date.isin(days)][["ts_code","trade_date","y","amt_yi","idx_dist","dist_high60"]].copy()
    print(f"补位日 {len(days)}  候选 {len(keep):,} 行",flush=True)
    tkey=set(zip(keep.ts_code,keep.trade_date))
    del d; gc.collect()

    # 只读一次 panel, 逐窗口算完立即压缩成 target 子集
    px_parts=[]
    for p in sorted(PANEL.glob("*.csv.gz")):
        try: x=pd.read_csv(p,usecols=["ts_code","trade_date","high","close","factor"])
        except Exception: continue
        if not len(x): continue
        f=x["factor"].fillna(1.0)
        px_parts.append(pd.DataFrame({
            "ts_code":x["ts_code"].astype("category"),
            "trade_date":x["trade_date"].astype("int64"),
            "ah":(x["high"]*f).astype("float32"),
            "ac":(x["close"]*f).astype("float32")}))
    px=pd.concat(px_parts,ignore_index=True); del px_parts; gc.collect()
    px=px.sort_values(["ts_code","trade_date"],kind="mergesort").reset_index(drop=True)
    print(f"  panel {len(px):,} 行已载入",flush=True)
    win={}
    for w in WS:
        hm=px.groupby("ts_code",sort=False)["ah"].transform(
            lambda s,w=w: s.shift(1).rolling(w,min_periods=max(20,w//3)).max())
        sub=pd.DataFrame({"ts_code":px["ts_code"].astype(str),"trade_date":px["trade_date"],
                          "dh":((px["ac"]/hm-1)*100).astype("float32")})
        del hm; gc.collect()
        sub=sub.dropna()
        sub=sub[[k in tkey for k in zip(sub.ts_code,sub.trade_date)]]
        win[w]=sub
        print(f"  窗口 {w}: {len(sub):,} 行",flush=True)
        del sub; gc.collect()
    del px; gc.collect()

    print(f"\n{'窗口':>6}{'单笔':>8}{'胜率%':>7}{'最差':>8}{'年化':>9}{'回撤':>9}{'卡玛':>7}{'Δ':>7}   三段Δ")
    rows=[]
    for w in WS:
        m=keep.merge(win[w],on=["ts_code","trade_date"],how="left")
        v={}
        for dt,g in m.groupby("trade_date"):
            g=g.dropna(subset=["dh"]).sort_values(["dh","amt_yi"],ascending=[True,False])
            if len(g): v[dt]=g.iloc[0]["y"]-COST
        v=pd.Series(v); sim=base.copy(); sim[v.index]+=v.values
        a,dd,cm=M(sim)
        segs=[]
        for y0,y1 in ((2016,2019),(2020,2023),(2024,2026)):
            msk=(sim.index//10000>=y0)&(sim.index//10000<=y1)
            _,_,c=M(sim[msk]); _,_,cb=M(base[msk]); segs.append(round(c-cb,2))
        rows.append({"窗口":w,"次数":len(v),"单笔":round(v.mean(),3),"胜率%":round((v>0).mean()*100,1),
                     "最差":round(v.min(),1),"年化":a,"回撤":dd,"卡玛":cm,"Δ":round(cm-b[2],2),"三段":segs})
        print(f"{w:>6}{v.mean():>8.3f}{(v>0).mean()*100:>7.1f}{v.min():>8.1f}{a:>9.2f}{dd:>9.2f}{cm:>7.2f}{cm-b[2]:>+7.2f}   {segs}",flush=True)
        if w==250:
            v250=v
    pd.DataFrame(rows).to_csv(OUT/"w243j_win.csv",index=False,encoding="utf-8-sig")

    print("\n250日窗口分年:")
    p=pd.DataFrame({"yr":v250.index//10000,"y":v250.values})
    g=p.groupby("yr").y.agg(["size","sum","mean"]); g["win%"]=p.groupby("yr").y.apply(lambda x:(x>0).mean()*100).round(0)
    print(g.round(2).to_string())
    run=mx=0
    for x in v250.values:
        run=run+1 if x<0 else 0; mx=max(mx,run)
    print(f"合计 {len(v250)} 次 单笔{v250.mean():+.3f}% 胜率{(v250>0).mean()*100:.1f}% 最长连亏{mx}")

if __name__=="__main__": main()
