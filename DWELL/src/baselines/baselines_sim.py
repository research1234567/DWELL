"""Competitor baselines (decaying-step online conformal, quantile tracking, conformal PID) in the SAME simulator
as env_bench.run (same regime blocks, same T, same RNG stream 1000+seed incl. the 500 initial-window draws).
Added 2026-10-28; never modifies stored arrays. Metrics order identical to env_bench.run:
 [cov_dev_overall, cov_dev_panic, pinball_x100, mse_calm_x1e4, mse_panic_x1e4, step_calm, step_panic]
(last two: mean *threshold-space* step for thr methods, level step for level methods -- not comparable across spaces)."""
import sys, numpy as np, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core import *
import env_bench as eb
T,BLK,W,TAU=eb.T,eb.BLK,eb.W,eb.TAU

def run(method, panic, seed, calm=None, blk=None, T_=None):
    """method: dict(kind=..., params). kinds:
       decay_thr : q_{t+1}=q_t + c t^-(1/2+eps) (err-alpha)     [Angelopoulos, Barber, Bates 2024 update, score=y]
       decay_lvl : a_{t+1}=a_t + c t^-(1/2+eps) (alpha-err), q = window quantile(1-a)  (matched to level-form arms)
       qt_thr    : q_{t+1}=q_t + eta (err-alpha), eta fixed or 'eta_frac' * trailing max (Angelopoulos-Candes-Tibshirani 2023)
       pid_thr   : q_{t+1}= Qwin(1-alpha) + K_I tan( S_t log t /(t Csat) ),  S_t=sum(err-alpha)  (tan integrator, trailing-quantile scorecaster, no D term)"""
    calm=calm or eb.calm; bl=blk or BLK; Tn=T_ or T
    isp=np.array([((t-1)//bl)%2==1 for t in range(1,Tn+1)])
    envf=lambda t: panic if ((t-1)//bl)%2==1 else calm
    rng=np.random.default_rng(1000+seed); e0=calm
    buf=list(1/(1+np.exp(-(e0.a-e0.b*e0.qstar+e0.sig*rng.standard_normal(W)))))
    from collections import deque
    buf=deque(buf,maxlen=W)
    k=method['kind']; p=method
    arr=np.fromiter(buf,float); q=float(np.quantile(arr,1-ALPHA)); a=ALPHA; S=0.0
    dep=[calm.qstar]*400
    cov=np.zeros(T_ or T);pin=np.zeros(Tn);err2=np.zeros(Tn);gm=np.zeros(Tn)
    for t in range(1,Tn+1):
        env=envf(t)
        if k=='decay_lvl':
            qd=float(eb.qlev(np.fromiter(buf,float),np.array([a]))[0])
        elif k=='pid_thr':
            qb=float(np.quantile(np.fromiter(buf,float),1-ALPHA))
            x=S*np.log(t)/(t*p['Csat']) if t>1 else 0.0
            r=p['KI']*np.tan(x) if abs(x)<np.pi/2 else np.sign(x)*np.inf
            qd=qb+r if np.isfinite(r) else (1.0 if r>0 else -1.0)
            qd=min(1.0,max(-1.0,qd))
        else:
            qd=min(1.0,max(-1.0,q))
        dep.append(qd); y=env.draw(dep[-1-env.d],rng); miss=float(y>qd)
        cov[t-1]=1-miss; pin[t-1]=(y-qd)*(TAU-(y<qd)); err2[t-1]=(qd-env.qstar)**2
        if k=='decay_thr':
            g=p['c']*t**(-(0.5+p.get('eps',0.1))); q=q+g*(miss-ALPHA); gm[t-1]=g
        elif k=='decay_lvl':
            g=p['c']*t**(-(0.5+p.get('eps',0.1))); a=a+g*(ALPHA-miss); gm[t-1]=g
        elif k=='qt_thr':
            if 'eta_frac' in p:
                Bh=max(buf)  # trailing max of scores over window (paper heuristic eta=0.1*B_t, B_t trailing max over Delta)
                g=p['eta_frac']*Bh
            else: g=p['eta']
            q=q+g*(miss-ALPHA); gm[t-1]=g
        elif k=='pid_thr':
            S+=miss-ALPHA; gm[t-1]=0.0
        buf.append(y)
    return [abs(cov.mean()-.9)*100,abs(cov[isp].mean()-.9)*100,pin.mean()*100,err2[~isp].mean()*1e4,err2[isp].mean()*1e4,gm[~isp].mean(),gm[isp].mean()]

def regimes(family):
    """returns dict name->(calm,panic). 'f2' logit-normal (seeds 0..49 in the paper), 'g2' heavy-tailed (0..19)"""
    if family=='f2':
        calm=Env(a=-0.5,b=3.0,d=1); return {n:(calm,Env(a=a_,b=b_,d=d_)) for n,(a_,b_,d_) in eb.TRUTH.items()}
    calm=EnvT(qstar=0.30,rho=0.64,d=1); TR={'mild':(0.10,1.85,6),'severe':(0.10,2.40,12),'vsevere':(0.10,2.08,24)}
    return {n:(calm,EnvT(qstar=q0,rho=rho,d=d)) for n,(q0,rho,d) in TR.items()}

def csat(Tn=T,delta=0.01):
    # Appendix B heuristic of Angelopoulos-Candes-Tibshirani 2023: Csat=(2/pi)(ceil(log(T)*delta)-1/log(T))
    return (2/np.pi)*(np.ceil(np.log(Tn)*delta)-1/np.log(Tn))
if __name__=="__main__":
    import time
    cs=regimes('f2')
    for name,(c,pn) in cs.items():
        for m in [dict(kind='decay_thr',c=1.0),dict(kind='qt_thr',eta_frac=0.1),dict(kind='pid_thr',KI=1.0,Csat=csat()),dict(kind='decay_lvl',c=0.5)]:
            t0=time.time(); r=run(m,pn,0,calm=c); print(name,m['kind'],'%.2fs'%(time.time()-t0),[round(float(x),3) for x in r])
