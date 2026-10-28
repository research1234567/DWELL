import numpy as np, warnings
from core import *
from statsmodels.regression.quantile_regression import QuantReg
warnings.filterwarnings("ignore")
class SMACI:
    """Stability-Margin ACI.  (i) randomized probing: deploy q_t+eps_t, eps_t~N(0,(cd/f_hat)^2)
    (ii) distributed-lag *quantile* regression of y_t on eps_t..eps_{t-D} at level tau=1-alpha (+ control q_t):
         coefficient on eps_{t-k} = -rho_k  (rho_k = d Q_tau / d q_{t-k}),  f_hat from residual sparsity
    (iii) gain scheduling: gamma_t = c * gt_crit(rho_hat)/f_hat  with gt_crit from the exact stability map."""
    def __init__(s,D=10,W=800,every=50,c=0.5,cd=0.015,gt_max=1.0,gt_min=0.01,f0=2.0,warm=150,h=0.05):
        s.D,s.W,s.every,s.c,s.cd,s.gt_max,s.gt_min,s.f,s.warm,s.h=D,W,every,c,cd,gt_max,gt_min,f0,warm,h
        s.rho=np.zeros(D); s.eps=[];s.y=[];s.q=[]; s.gt=0.05; s.hist=[]
    def __call__(s,q,y,miss,t,st):
        s.eps.append(st['eps']); s.y.append(y); s.q.append(st['qd']-st['eps'])
        if t>=s.warm and t%s.every==0: s._identify()
        g=s.gt/s.f; st['g']=g; st['dither']=s.cd
        s.hist.append((s.f,s.rho.copy(),s.gt))
        return q+g*(miss-ALPHA)
    def _identify(s):
        W,D=s.W,s.D; n=len(s.y); lo=max(D,n-W); idx=np.arange(lo,n)
        if len(idx)<120: return
        e=np.array(s.eps); yv=np.array(s.y)[idx]; q=np.array(s.q)[idx]
        X=np.column_stack([np.ones(len(idx)),q]+[e[idx-k] for k in range(D+1)])
        try: r=QuantReg(yv,X).fit(q=1-ALPHA,max_iter=500,p_tol=1e-5)
        except Exception: return
        b=r.params; u=yv-X@b; tau=1-ALPHA; h=s.h
        spars=np.quantile(u,min(.999,tau+h))-np.quantile(u,tau-h)
        if spars<=1e-9: return
        fh=2*h/spars
        # robust se by bootstrap-free approx: Powell sandwich replaced by sqrt(tau(1-tau))/(f*s*sqrt(n)) per lag
        sdk=np.array([np.std(e[idx-k]) for k in range(D+1)])
        se=np.sqrt(tau*(1-tau))/(fh*sdk*np.sqrt(len(idx)))
        rho=-b[3:]                       # lags 1..D (index 2 is lag 0)
        rho=np.where(np.abs(rho)<1.0*se[1:],0.0,rho)
        s.f=0.5*s.f+0.5*fh; s.rho=np.maximum(rho,0.0)
        cap=s.gt_max/s.c
        gc=gt_crit_general(list(s.rho),gmax=cap,n=500) if s.rho.sum()>0 else cap
        s.gt=float(np.clip(s.c*min(gc,cap),s.gt_min,s.gt_max))
if __name__=="__main__":
    for (a,b,d) in [(-1.0,24.0,6),(-0.5,3.0,1),(-1.0,24.0,12)]:
        e=Env(a=a,b=b,d=d); c=SMACI(); o=run(lambda t:e,c,6000,1)
        fh,rh,gt=c.hist[-1]; print(f"b={b} d={d} true f={e.f:.2f} rho={e.rho:.2f}| est f={fh:.2f} rho_k={np.round(rh,2)} sum={rh.sum():.2f}")
        print("   true gt_c=%.3f est gt_c=%.3f  used gt=%.3f"%(gt_crit(e.rho,d),gt_crit_general(list(rh),n=500),gt))
