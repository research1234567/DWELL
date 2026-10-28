"""Envelope-guarded DtACI (EG-DtACI): cap each expert's level-step at c * min_{1<=d<=dbar} gt_crit(rhobar, d).
   No online probing.  Truth regimes: calm (d=1,rho=.64) alternating with a reactive regime; L=2000, T=8000."""
import os, sys, json, numpy as np, warnings; warnings.filterwarnings("ignore")
from collections import deque
from core import *
T=8000; BLK=2000; W=500; TAU=1-ALPHA; C=0.5
GAMS=np.array([0.005,0.01,0.02,0.05,0.1,0.2]); ETA=60.; SHARE=1e-3
TRUTH={'extreme':(-2.0,120.0,60),'mild':(-1.0,24.0,6),'severe':(-1.0,48.0,12),'vsevere':(-1.5,48.0,24)}
calm=Env(a=-0.5,b=3.0,d=1)
isp=np.array([((t-1)//BLK)%2==1 for t in range(1,T+1)])
_cache={}
def robust_cap(rhobar,dbar,c=C):
    k=(round(rhobar,3),dbar)
    if k not in _cache:
        ds=list(range(1,dbar+1)) if dbar<=30 else sorted(set(list(range(1,10))+list(range(10,dbar+1,5))+list(range(max(1,dbar-6),dbar+1))))
        _cache[k]=c*min(gt_crit(rhobar,d,gmax=2.0,n=700 if d<=30 else 1200) for d in ds)
    return _cache[k]
def qlev(arr,alpha_level):
    """threshold for miscoverage level alpha_level; exact conventions: alpha>1 -> empty set (always miss), alpha<0 -> full space (never miss)"""
    al=np.asarray(alpha_level,float); out=np.empty_like(al)
    hi=al<0; lo=al>1; mid=~(hi|lo)
    out[hi]=1.0; out[lo]=-1.0
    if mid.any(): out[mid]=np.quantile(arr,1-al[mid])
    return out
def run(panic,cap,seed,fixed=None,common=False,exact=True,blk=None):
    bl=blk or BLK; isp=np.array([((t-1)//bl)%2==1 for t in range(1,T+1)])
    envf=lambda t: panic if ((t-1)//bl)%2==1 else calm
    rng=np.random.default_rng(1000+seed); e0=calm
    buf=deque(list(1/(1+np.exp(-(e0.a-e0.b*e0.qstar+e0.sig*rng.standard_normal(W))))),maxlen=W)
    a=np.full(len(GAMS),ALPHA); w=np.ones(len(GAMS))/len(GAMS); dep=[calm.qstar]*400; at=ALPHA
    cov=np.zeros(T);pin=np.zeros(T);err2=np.zeros(T);gm=np.zeros(T)
    for t in range(1,T+1):
        env=envf(t); arr=np.fromiter(buf,float)
        if fixed is None:
            qi=qlev(arr,a) if exact else np.quantile(arr,1-np.clip(a,0,1)); q=float(qlev(arr,np.array([float(w@a)]))[0]) if exact else float(np.quantile(arr,1-np.clip(float(w@a),0,1)))
        else: q=float(np.quantile(arr,1-min(1,max(0,at))))
        dep.append(q); y=env.draw(dep[-1-env.d],rng)
        cov[t-1]=float(y<=q); pin[t-1]=(y-q)*(TAU-(y<q)); err2[t-1]=(q-env.qstar)**2
        if fixed is None:
            loss=(y-qi)*(TAU-(y<qi)); w=w*np.exp(-ETA*loss); w/=w.sum(); w=(1-SHARE)*w+SHARE/len(GAMS)
            ge=np.minimum(GAMS,cap(t) if callable(cap) else cap); errv=(float(y>q)*np.ones(len(GAMS))) if common else (y>qi).astype(float); a=a+ge*(ALPHA-errv); gm[t-1]=float(w@ge)
        else: at+=fixed*(ALPHA-float(y>q)); gm[t-1]=fixed
        buf.append(y)
    return [abs(cov.mean()-.9)*100,abs(cov[isp].mean()-.9)*100,pin.mean()*100,err2[~isp].mean()*1e4,err2[isp].mean()*1e4,gm[~isp].mean(),gm[isp].mean()]
def agg(rs): r=np.array(rs); return r.mean(0).tolist(),(r.std(0,ddof=1)/np.sqrt(len(r))).tolist()
if __name__=="__main__":
    name=sys.argv[1]; NS=int(sys.argv[2]); a_,b_,d_=TRUTH[name]; panic=Env(a=a_,b=b_,d=d_)
    gtc=gt_crit(panic.rho,d_,gmax=2.0,n=700); R={'truth':dict(rho=panic.rho,d=d_,gtc=gtc)}
    def go(label,fn): R[label]=dict(zip(('mean','se'),agg([fn(s) for s in range(NS)]))); m=R[label]['mean']; print(f"{label:34s} pin {m[2]:.3f} | panic MSE {m[4]:7.1f} | cov|dev| {m[0]:.2f}/{m[1]:.2f} | gamma panic {m[6]:.3f}",flush=True)
    go('Fixed g=0.02',lambda s:run(panic,None,s,fixed=0.02))
    go('Fixed at oracle cap',lambda s:run(panic,None,s,fixed=C*gtc))
    go('DtACI (uncapped)',lambda s:run(panic,np.inf,s))
    go('DtACI + oracle cap',lambda s:run(panic,C*gtc,s))
    for kr in [0.5,1.0,2.0]:
        for kd in [0.5,1.0,2.0]:
            rb=kr*panic.rho; db=max(1,int(round(kd*d_))); cap=robust_cap(rb,db)
            go(f'EG-DtACI rho x{kr}, dbar x{kd}',lambda s,cap=cap:run(panic,cap,s)); R[f'EG-DtACI rho x{kr}, dbar x{kd}']['cap']=cap
    rb,db=3.0,24; cap=robust_cap(rb,db); go('EG-DtACI universal (rho<=3,d<=24)',lambda s:run(panic,cap,s)); R['EG-DtACI universal (rho<=3,d<=24)']['cap']=cap
    for lag in [0,100,300]:
        flag=lambda t,lag=lag: ((max(1,t-lag)-1)//BLK)%2==1
        go(f'EG-DtACI regime-aware (flag lag {lag})',lambda s,flag=flag:run(panic,(lambda t,flag=flag: cap if flag(t) else np.inf),s))
    json.dump(R,open(f'out/env_{name}.json','w'),indent=1)
