import os, numpy as np, json, sys, warnings; warnings.filterwarnings("ignore")
from collections import deque
from statsmodels.regression.quantile_regression import QuantReg
from core import *
import os
BLK=int(os.environ.get('BLK',800)); T=int(os.environ.get('TT',6400)); WN=int(os.environ.get('WN',800)); NS=int(sys.argv[1]) if len(sys.argv)>1 else 10; W=500
PA=float(os.environ.get('PA',-1.0)); PB=float(os.environ.get('PB',24.0)); PD=int(os.environ.get('PD',6))
calm=Env(a=-0.5,b=3.0,d=1); panic=Env(a=PA,b=PB,d=PD)
envf=lambda t: panic if ((t-1)//BLK)%2==1 else calm
isp=np.array([((t-1)//BLK)%2==1 for t in range(1,T+1)])
GT={id(calm):gt_crit(calm.rho,1),id(panic):gt_crit(panic.rho,PD)}
GAMS=np.array([0.005,0.01,0.02,0.05,0.1,0.2]); ETA=60.; SHARE=1e-3; TAU=1-ALPHA

class Spike:
    """single-spike probing estimator: dithered distributed-lag quantile regression -> (d_hat, rho_hat)"""
    def __init__(s,D=10,Wn=800,every=100,s_dither=0.01,warm=200): s.D,s.Wn,s.every,s.sd,s.warm=D,Wn,every,s_dither,warm; s.e=[];s.y=[];s.q=[];s.cap=0.1;s.rho=0.;s.d=1
    def update(s,t,eps,y,q):
        s.e.append(eps);s.y.append(y);s.q.append(q)
        if t>=s.warm and t%s.every==0:
            n=len(s.y);lo=max(s.D,n-s.Wn);idx=np.arange(lo,n)
            e=np.array(s.e);X=np.column_stack([np.ones(len(idx)),np.array(s.q)[idx]]+[e[idx-k] for k in range(s.D+1)])
            try: r=QuantReg(np.array(s.y)[idx],X).fit(q=TAU,max_iter=400,p_tol=1e-5)
            except Exception: return
            rho=-r.params[3:]; k=int(np.argmax(rho))+1; rh=max(rho[k-1],0.0)
            s.rho=0.5*s.rho+0.5*rh if s.rho>0 else rh; s.d=k
            s.cap=1.0 if s.rho<0.05 else float(min(1.0,gt_crit(s.rho,s.d,gmax=2.0,n=800)))
def run_one(variant,seed):
    rng=np.random.default_rng(1000+seed); env0=calm
    buf=deque(list(1/(1+np.exp(-(env0.a-env0.b*env0.qstar+env0.sig*rng.standard_normal(W))))),maxlen=W)
    a=np.full(len(GAMS),ALPHA); w=np.ones(len(GAMS))/len(GAMS); dep=[calm.qstar]*80
    sp=Spike(D=int(os.environ.get('SD',10)),Wn=WN) if variant=='identified' else None; c=0.5
    cov=np.zeros(T);pin=np.zeros(T);err2=np.zeros(T);gm=np.zeros(T)
    for t in range(1,T+1):
        env=envf(t); arr=np.fromiter(buf,float)
        qi=np.quantile(arr,1-np.clip(a,0,1)); alpha_t=float(w@a); q=float(np.quantile(arr,1-np.clip(alpha_t,0,1)))
        eps=sp.sd*rng.standard_normal() if sp else 0.0
        qd=min(1-1e-6,max(1e-6,q+eps)); dep.append(qd); y=env.draw(dep[-1-env.d],rng)
        cov[t-1]=float(y<=qd); pin[t-1]=(y-qd)*(TAU-(y<qd)); err2[t-1]=(q-env.qstar)**2
        loss=(y-qi)*(TAU-(y<qi)); w=w*np.exp(-ETA*loss); w/=w.sum(); w=(1-SHARE)*w+SHARE/len(GAMS)
        errs=(y>qi).astype(float)
        if variant=='oracle': cap=c*GT[id(env)]
        elif variant=='identified': sp.update(t,eps,y,q); cap=c*sp.cap
        else: cap=np.inf
        ge=np.minimum(GAMS,cap); a=a+ge*(ALPHA-errs); gm[t-1]=float(w@ge); buf.append(y)
    return cov,pin,err2,gm
def run_fixed(g,seed):
    rng=np.random.default_rng(1000+seed); env0=calm
    buf=deque(list(1/(1+np.exp(-(env0.a-env0.b*env0.qstar+env0.sig*rng.standard_normal(W))))),maxlen=W); at=ALPHA; dep=[calm.qstar]*80
    cov=np.zeros(T);pin=np.zeros(T);err2=np.zeros(T)
    for t in range(1,T+1):
        env=envf(t); q=float(np.quantile(np.fromiter(buf,float),1-min(1,max(0,at)))); dep.append(q); y=env.draw(dep[-1-env.d],rng)
        cov[t-1]=float(y<=q); pin[t-1]=(y-q)*(TAU-(y<q)); err2[t-1]=(q-env.qstar)**2; at+=g*(ALPHA-float(y>q)); buf.append(y)
    return cov,pin,err2,np.full(T,g)
def summarize(res):
    r=[]
    for cov,pin,err2,gm in res: r.append([abs(cov.mean()-.9)*100,abs(cov[isp].mean()-.9)*100,pin.mean()*100,err2[~isp].mean()*1e4,err2[isp].mean()*1e4,gm[~isp].mean(),gm[isp].mean()])
    r=np.array(r); return r.mean(0).tolist(),(r.std(0,ddof=1)/np.sqrt(len(r))).tolist()
if __name__=="__main__":
    R={}
    jobs=[("Fixed (level) g=0.02",lambda s:run_fixed(0.02,s)),
          ("DtACI (level-update)",lambda s:run_one('plain',s)),("DtACI + cap (oracle rho,d)",lambda s:run_one('oracle',s)),("DtACI + cap (identified, single-spike)",lambda s:run_one('identified',s))]
    for name,fn in jobs:
        m,se=summarize([fn(s) for s in range(NS)]); R[name]=dict(mean=m,se=se)
        print(f"{name:42s} cov|dev| all {m[0]:.2f} panic {m[1]:.2f} | pinball {m[2]:.3f}±{se[2]:.3f} | MSE calm {m[3]:.1f} panic {m[4]:.1f}±{se[4]:.1f} | gamma calm {m[5]:.3f} panic {m[6]:.3f}",flush=True)
    json.dump(R,open(f'out/alpha_sev_b{int(PB)}_d{PD}_L{BLK}.json' if os.environ.get('SEV') else f'out/alpha_bench_L{BLK}_W{WN}.json','w'),indent=1)
