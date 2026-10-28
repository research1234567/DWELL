"""Drifting-baseline benchmark: logit-mean a_t = a_regime + D_t, D_t a reflected random walk (sd 0.012/step, |D|<=0.9).
   Compares fixed steps, uncapped DtACI, and envelope-guarded DtACI (universal / regime-aware).  T=8000, L=2000."""
import sys, json, numpy as np, warnings; warnings.filterwarnings("ignore")
from collections import deque
from core import *
import env_bench as eb
T=8000; BLK=2000; W=500; TAU=1-ALPHA; GAMS=eb.GAMS; ETA=eb.ETA; SHARE=eb.SHARE
cache={}
def getenv(a,b,d):
    k=(round(a,2),b,d)
    if k not in cache: cache[k]=Env(a=k[0],b=b,d=d)
    return cache[k]
def run(truth,cap,seed,fixed=None,sd=float(__import__("os").environ.get("DRIFT",0.012))):
    a_p,b_p,d_p=truth; rng=np.random.default_rng(5000+seed)
    D=np.zeros(T+1)
    for t in range(1,T+1): D[t]=np.clip(D[t-1]+sd*rng.standard_normal(),-0.9,0.9)
    isp=lambda t:((t-1)//BLK)%2==1
    envf=lambda t: getenv(a_p+D[t],b_p,d_p) if isp(t) else getenv(-0.5+D[t],3.0,1)
    e0=envf(1); buf=deque(list(1/(1+np.exp(-(e0.a-e0.b*e0.qstar+e0.sig*rng.standard_normal(W))))),maxlen=W)
    a=np.full(len(GAMS),ALPHA); w=np.ones(len(GAMS))/len(GAMS); dep=[e0.qstar]*80; at=ALPHA
    cov=np.zeros(T);pin=np.zeros(T);err2=np.zeros(T)
    for t in range(1,T+1):
        env=envf(t); arr=np.fromiter(buf,float)
        if fixed is None: qi=np.quantile(arr,1-np.clip(a,0,1)); q=float(np.quantile(arr,1-np.clip(float(w@a),0,1)))
        else: q=float(np.quantile(arr,1-min(1,max(0,at))))
        dep.append(q); y=env.draw(dep[-1-env.d],rng)
        cov[t-1]=float(y<=q); pin[t-1]=(y-q)*(TAU-(y<q)); err2[t-1]=(q-env.qstar)**2
        if fixed is None:
            loss=(y-qi)*(TAU-(y<qi)); w=w*np.exp(-ETA*loss); w/=w.sum(); w=(1-SHARE)*w+SHARE/len(GAMS)
            ge=np.minimum(GAMS,cap(t) if callable(cap) else cap); a=a+ge*(ALPHA-(y>qi).astype(float))
        else: at+=fixed*(ALPHA-float(y>q))
        buf.append(y)
    p=np.array([isp(t) for t in range(1,T+1)])
    return [abs(cov.mean()-.9)*100,abs(cov[p].mean()-.9)*100,pin.mean()*100,err2[~p].mean()*1e4,err2[p].mean()*1e4]
if __name__=="__main__":
    name=sys.argv[1]; NS=int(sys.argv[2]); truth=eb.TRUTH[name]; panic=Env(a=truth[0],b=truth[1],d=truth[2])
    ucap=eb.robust_cap(3.0,24); ocap=0.5*gt_crit(panic.rho,truth[2],gmax=2.0,n=700)
    flag=lambda t:((t-1)//BLK)%2==1
    M=[('Fixed g=0.005',dict(cap=None,fixed=0.005)),('Fixed g=0.01',dict(cap=None,fixed=0.01)),('Fixed g=0.02',dict(cap=None,fixed=0.02)),('Fixed g=0.05',dict(cap=None,fixed=0.05)),
       ('DtACI (uncapped)',dict(cap=np.inf)),('DtACI + oracle cap (regime-aware)',dict(cap=lambda t:ocap if flag(t) else np.inf)),
       ('EG-DtACI universal',dict(cap=ucap)),('EG-DtACI regime-aware',dict(cap=lambda t:ucap if flag(t) else np.inf))]
    R={}
    for lab,kw in M:
        rs=np.array([run(truth,kw['cap'],s,fixed=kw.get('fixed')) for s in range(NS)])
        R[lab]=dict(mean=rs.mean(0).tolist(),se=(rs.std(0,ddof=1)/np.sqrt(NS)).tolist()); m=R[lab]['mean']
        print(f"{lab:36s} pin {m[2]:.3f}±{R[lab]['se'][2]:.3f} | MSE calm {m[3]:6.1f} panic {m[4]:7.1f} | cov|dev| {m[0]:.2f}/{m[1]:.2f}",flush=True)
    json.dump(R,open(f'out/env2_{name}.json','w'),indent=1)
