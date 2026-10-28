"""Non-oracle cap variants for EG-DtACI. Copy of env_bench.run (same dynamics, RNG, metrics) with a hook so the
level-step cap can depend on observable history only (deployed thresholds, window, miss indicators).
  cap=float          constant cap  (used for leave-one-regime-out cap selection)
  cap=Backoff(...)   online oscillation backoff, no knowledge of rho or d (see class)
Metrics identical to env_bench.run, plus mean cap in panic/calm blocks."""
import sys, numpy as np, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
from collections import deque
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core import *
import env_bench as eb
T,BLK,W,TAU,GAMS,ETA,SHARE=eb.T,eb.BLK,eb.W,eb.TAU,eb.GAMS,eb.ETA,eb.SHARE

class Backoff:
    """Observable-only step cap. Every `every` steps compare the recent mean absolute threshold increment D with the
    increment expected from the steps actually used, 2*alpha*(1-alpha)*eta_bar/f_hat (f_hat = window density estimate at the
    quantile). R=D/expected ~ 1 when the tracker moves only because of its own steps; R>>1 means the response is feeding back.
    R>kappa: cap*=down ; R<kappa_lo: cap*=up (to cmax). No rho, no d, no regime flag."""
    def __init__(s,kappa=2.0,kappa_lo=1.3,down=0.7,up=1.05,every=50,hor=100,cmax=0.2,cmin=0.002):
        s.p=dict(kappa=kappa,kappa_lo=kappa_lo,down=down,up=up,every=every,hor=hor,cmax=cmax,cmin=cmin); s.reset()
    def reset(s):
        s.cap=s.p['cmax']; s.D=None; s.prevq=None; s.E=None; s.n=0
    def observe(s,t,qd,buf,eta_bar):
        p=s.p
        if s.prevq is not None:
            inc=abs(qd-s.prevq); w=1.0/p['hor']
            arr=np.fromiter(buf,float)
            if t%p['every']==0 or s.E is None:
                lo,hi=np.quantile(arr,[0.85,0.95]); fh=0.1/max(hi-lo,1e-3)
            else: fh=s._fh
            s._fh=fh; exp_inc=2*ALPHA*(1-ALPHA)*eta_bar/fh
            s.D=inc if s.D is None else (1-w)*s.D+w*inc
            s.E=exp_inc if s.E is None else (1-w)*s.E+w*exp_inc
        s.prevq=qd
        if s.D is not None and t%p['every']==0 and t>p['hor']:
            R=s.D/max(s.E,1e-9)
            if R>p['kappa']: s.cap=max(p['cmin'],s.cap*p['down'])
            elif R<p['kappa_lo']: s.cap=min(p['cmax'],s.cap*p['up'])
        return s.cap

def run(panic,cap,seed,calm=None,blk=None,common=True):
    calm=calm or eb.calm; bl=blk or BLK
    isp=np.array([((t-1)//bl)%2==1 for t in range(1,T+1)])
    envf=lambda t: panic if ((t-1)//bl)%2==1 else calm
    rng=np.random.default_rng(1000+seed); e0=calm
    buf=deque(list(1/(1+np.exp(-(e0.a-e0.b*e0.qstar+e0.sig*rng.standard_normal(W))))),maxlen=W)
    a=np.full(len(GAMS),ALPHA); w=np.ones(len(GAMS))/len(GAMS); dep=[calm.qstar]*400
    cov=np.zeros(T);pin=np.zeros(T);err2=np.zeros(T);gm=np.zeros(T);cp=np.zeros(T)
    isb=isinstance(cap,Backoff)
    if isb: cap.reset(); curcap=cap.cap
    else: curcap=cap
    for t in range(1,T+1):
        env=envf(t); arr=np.fromiter(buf,float)
        qi=eb.qlev(arr,a); q=float(eb.qlev(arr,np.array([float(w@a)]))[0])
        dep.append(q); y=env.draw(dep[-1-env.d],rng)
        cov[t-1]=float(y<=q); pin[t-1]=(y-q)*(TAU-(y<q)); err2[t-1]=(q-env.qstar)**2
        loss=(y-qi)*(TAU-(y<qi)); w=w*np.exp(-ETA*loss); w/=w.sum(); w=(1-SHARE)*w+SHARE/len(GAMS)
        ge=np.minimum(GAMS,curcap); errv=(float(y>q)*np.ones(len(GAMS))) if common else (y>qi).astype(float)
        a=a+ge*(ALPHA-errv); gm[t-1]=float(w@ge); cp[t-1]=curcap if np.isfinite(curcap) else GAMS.max()
        buf.append(y)
        if isb: curcap=cap.observe(t,q,buf,gm[t-1])
    return [abs(cov.mean()-.9)*100,abs(cov[isp].mean()-.9)*100,pin.mean()*100,err2[~isp].mean()*1e4,err2[isp].mean()*1e4,gm[~isp].mean(),gm[isp].mean(),cp[~isp].mean(),cp[isp].mean()]
