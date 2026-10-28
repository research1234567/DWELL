"""Real outflow series (Aave V3, 12 months, 5 pools) with optional injected reflexivity: y'_t = y_t * exp(-kappa*(q_{t-d}-q0)), kappa=rho/q0.
rho=0 => passive backtest on the real data."""
import sys, json, numpy as np, warnings; warnings.filterwarnings("ignore")
from collections import deque
from real_panel import build
ALPHA=0.1; TAU=0.9; W=168; GAMS=np.array([0.005,0.01,0.02,0.05,0.1,0.2]); ETA=60.; SHARE=1e-3
def series():
    p=build(); out={}
    for a,g in p.groupby('asset'):
        g=g.sort_values('hour'); y=g.y.fillna(0.0).values; out[a]=y
    return out
def qlev(arr,al):
    al=np.asarray(al,float); out=np.empty_like(al); hi=al<0; lo=al>1; mid=~(hi|lo)
    out[hi]=1e3*(arr.max()+1e-9); out[lo]=-1.0
    if mid.any(): out[mid]=np.quantile(arr,1-al[mid])
    return out
def replay(y,rho,d,cap=None,fixed=None,common=False,exact=True,stats=None,zero=False):
    s=10*y[:W].mean()+1e-12; z=y/s; q0=float(np.quantile(z[:W],TAU))+1e-9; kappa=rho/q0
    buf=deque(list(z[:W]),maxlen=W); a=np.full(len(GAMS),ALPHA); w=np.ones(len(GAMS))/len(GAMS); at=ALPHA; dep=[q0]*(d+2)
    n=len(z)-W; cov=np.zeros(n); pin=np.zeros(n)
    for i in range(n):
        arr=np.fromiter(buf,float)
        if fixed is None:
            if exact: qi=qlev(arr,a); q=float(qlev(arr,np.array([float(w@a)]))[0])
            else:
                qi=np.quantile(arr,1-np.clip(a,0,1)); al_=float(w@a); forced=False
                if zero and al_>1: q=0.0; forced=True          # empty set -> zero buffer, error counted as a miss (outcomes >= 0)
                else: q=float(np.quantile(arr,1-min(1,max(0,al_))))
            if stats is not None: stats['neg']=stats.get('neg',0)+int(float(w@a)<0); stats['n']=stats.get('n',0)+1
        else: q=float(np.quantile(arr,1-min(1,max(0,at))))
        dep.append(q); yt=z[W+i]*np.exp(np.clip(-kappa*(dep[-1-d]-q0),-5,5)) if rho>0 else z[W+i]
        cov[i]=float(yt<=q); pin[i]=(yt-q)*(TAU-(yt<q))
        if fixed is None:
            loss=(yt-qi)*(TAU-(yt<qi)); loss=loss-loss.min(); w=w*np.exp(-ETA*loss); w/=w.sum(); w=(1-SHARE)*w+SHARE/len(GAMS)
            ge=np.minimum(GAMS,cap if cap is not None else np.inf); errd=1.0 if (zero and fixed is None and not exact and forced) else float(yt>q); a=a+ge*(ALPHA-((errd*np.ones(len(GAMS))) if common else (yt>qi).astype(float)))
        else: at+=fixed*(ALPHA-float(yt>q))
        buf.append(yt)
    hours=np.arange(W,W+n); svb=(hours>=216)&(hours<384)
    return cov.mean(),(cov[svb].mean() if svb.any() else np.nan),pin.mean()/ (y.mean()/s+1e-12)
if __name__=="__main__":
    rho=float(sys.argv[1]); d=int(sys.argv[2]); Y=series()
    from core import gt_crit
    rb=2*rho if rho>0 else 0.2; capEG=0.5*min(gt_crit(max(rb,1e-3),k,gmax=2.0,n=500) for k in range(1,max(d*2,2)+1)) if rb>0 else None
    meths=[('Fixed 0.005',dict(fixed=0.005)),('Fixed 0.02',dict(fixed=0.02)),('Fixed 0.05',dict(fixed=0.05)),('DtACI',dict(cap=None,exact=False)),('DtACI (deployed error)',dict(cap=None,common=True,exact=False,zero=True)),('EG-DtACI (envelope 2*rho,2*d)',dict(cap=capEG,common=True,exact=False,zero=True))]
    R={}
    for lab,kw in meths:
        r=np.array([replay(Y[a],rho,d,**kw) for a in sorted(Y)]); R[lab]=dict(cov=r[:,0].tolist(),svb=r[:,1].tolist(),pin=r[:,2].tolist())
        print(f"rho={rho} d={d} {lab:30s} cov {r[:,0].mean():.3f} (|dev| {abs(r[:,0]-.9).mean()*100:.2f}pp) svb-cov {np.nanmean(r[:,1]):.3f} pin(norm) {r[:,2].mean():.3f}",flush=True)
    json.dump(dict(rho=rho,d=d,cap=capEG,res=R,assets=sorted(Y)),open(f'out/replay_r{rho}_d{d}.json','w'))
