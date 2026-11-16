"""第243轮 I: 沿"窗口越长越好"的单调趋势外推 —— 120日 / 250日 / 上市以来最高

243H 的证据链(全部同一批 192 次, 只换排序键):
  距60日高 1.768 > 距20日高 1.752 > 20日乖离 1.323 > 60日乖离 1.223 > 10日乖离 0.850 > 5日乖离 0.067
两条独立趋势: ① 窗口越长越好 ② "距高点" 优于 "距均线"
若趋势为真, 继续延长窗口应继续变好。测 dist_high120 / 250 / 上市以来最高(ATH)。
同时测"距 60 日低"(反向) 作为安慰剂 —— 若它也变好, 说明只是噪声在动。
"""
from pathlib import Path
import numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[1]; PANEL=ROOT/"data"/"panel"; FEAT=ROOT/"output"/"feat234"; OUT=ROOT/"output"
COST=0.10
CACHE=OUT/"w243i_dh.parquet"
COLS=["ts_code","trade_date","y","n1_ok","pct","idx_dist","dist_high60","dist_high20","bias20",
      "amt_yi","low_tail","dist_low60","is_paused","is_st","listed","ret20"]

def dh_table():
    if CACHE.exists(): return pd.read_parquet(CACHE)
    parts=[]
    for p in sorted(PANEL.glob("*.csv.gz")):
        try:
            d=pd.read_csv(p,usecols=["ts_code","trade_date","high","low","close","factor"])
        except Exception: continue
        if not len(d): continue
        f=d["factor"].fillna(1.0)
        parts.append(pd.DataFrame({"ts_code":d["ts_code"],"trade_date":d["trade_date"].astype("int64"),
                                   "ah":(d["high"]*f).astype("float32"),"al":(d["low"]*f).astype("float32"),
                                   "ac":(d["close"]*f).astype("float32")}))
    px=pd.concat(parts,ignore_index=True)
    px=px.sort_values(["ts_code","trade_date"],kind="mergesort").reset_index(drop=True)
    g=px.groupby("ts_code",sort=False)
    # 不含当日: shift(1) 后再 rolling max
    for w in (120,250):
        col=f"ah{w}"
        px[col]=g["ah"].transform(lambda s,w=w: s.shift(1).rolling(w,min_periods=max(20,w//3)).max())
    # 上市以来最高(不含当日)
    px["ahATH"]=g["ah"].transform(lambda s: s.shift(1).cummax())
    px["al60"]=g["al"].transform(lambda s: s.shift(1).rolling(60,min_periods=20).min())
    out=px[["ts_code","trade_date","ac","ah120","ah250","ahATH","al60"]].copy()
    out["dh120"]=(out["ac"]/out["ah120"]-1)*100
    out["dh250"]=(out["ac"]/out["ah250"]-1)*100
    out["dhATH"]=(out["ac"]/out["ahATH"]-1)*100
    out["dl60"]=(out["ac"]/out["al60"]-1)*100
    out=out[["ts_code","trade_date","dh120","dh250","dhATH","dl60"]]
    out.to_parquet(CACHE,index=False,compression="zstd")
    print(f"缓存 {CACHE} {len(out):,} 行")
    return out

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
        if len(d): parts.append(d)
    d=pd.concat(parts,ignore_index=True)
    dh=dh_table()
    d=d.merge(dh,on=["ts_code","trade_date"],how="left")
    print(f"合并后 {len(d):,} 行  dh250 非空 {d.dh250.notna().sum():,}")
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
    print(f"补位日 {len(days)}\n")
    KEYS=[("dist_high20",True,"距20日高"),("dist_high60",True,"距60日高 (现行)"),
          ("dh120",True,"距120日高"),("dh250",True,"距250日高"),("dhATH",True,"距上市以来最高"),
          ("dl60",False,"距60日低 (安慰剂: 反向)")]
    rows=[]
    for col,asc,lab in KEYS:
        v={}
        for dt,g in sub.items():
            g2=g.dropna(subset=[col])
            if not len(g2): continue
            g2=g2.sort_values([col,"amt_yi"],ascending=[asc,False])
            v[dt]=g2.iloc[0]["y"]-COST
        if len(v)<50: continue
        v=pd.Series(v); sim=base.copy(); sim[v.index]+=v.values
        a,dd,cm=M(sim)
        segs=[]
        for y0,y1 in ((2016,2019),(2020,2023),(2024,2026)):
            m=(sim.index//10000>=y0)&(sim.index//10000<=y1)
            _,_,c=M(sim[m]); _,_,cb=M(base[m]); segs.append(round(c-cb,2))
        rows.append({"排序键":lab,"次数":len(v),"单笔":round(v.mean(),3),
                     "胜率%":round((v>0).mean()*100,1),"最差":round(v.min(),1),
                     "年化":a,"回撤":dd,"卡玛":cm,"Δ":round(cm-b[2],2),"三段":segs})
    r=pd.DataFrame(rows)
    pd.set_option("display.width",220)
    print(r.to_string(index=False))
    r.to_csv(OUT/"w243i_keys.csv",index=False,encoding="utf-8-sig")

if __name__=="__main__": main()
