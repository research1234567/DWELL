import numpy as np
from collections import deque
from core import *
from scipy.linalg import solve_discrete_lyapunov as dlyap
def run_alpha(env,gamma,T,seed,W=500,burn=0):
    """Gibbs-Candes level-update ACI: alpha_{t+1}=alpha_t+gamma(alpha-err_t); q_t=(1-alpha_t)-quantile of last W scores (scores=y)."""
    rng=np.random.default_rng(seed); d=env.d; a_t=ALPHA; buf=deque([float(env.qstar)]*W,maxlen=W)  # warm start at equilibrium law
    qh=[env.qstar]*(d+60); qs=[]
    for t in range(T):
        lvl=min(1.0,max(0.0,1-a_t)); q=float(np.quantile(np.fromiter(buf,float),lvl)) if lvl<1 else max(buf)*1.0
        qs.append(q); qh.append(q); y=env.draw(qh[-1-d],rng); err=float(y>q)
        buf.append(y); a_t=a_t+gamma*(ALPHA-err)
    return np.array(qs)
def lin_var_q(env,gamma_qt,d):
    gt=gamma_qt*env.f; A=companion(gt,env.rho,d); B=np.zeros((d+1,1)); B[0,0]=gamma_qt*np.sqrt(env.alpha*(1-env.alpha))
    return np.inf if max(abs(np.linalg.eigvals(A)))>=1 else dlyap(A,B@B.T)[0,0]
if __name__=="__main__":
    import json; R=[]
    for (a_,b_,d) in [(-0.5,3.0,0),(-1.0,24.0,6)]:
        e0=Env(a=a_,b=b_); gtc=gt_crit(e0.rho,d) if d>0 else 2/(1+e0.rho); e=Env(a=a_,b=b_,d=d)
        print(f"b={b_} d={d} rho={e.rho:.2f} predicted alpha-form gamma_c={gtc:.4f}")
        for W in [200,1000]:
            for r in [0.05,0.2,0.5,0.9,1.5]:
                g=r*gtc; v=np.mean([np.mean((run_alpha(e,g,12000,s,W=W)[2000:]-e.qstar)**2) for s in range(4)])
                th=lin_var_q(e,g/e.f,d)  # QT-equivalent step g/f
                R.append((b_,d,W,r,g,v,th)); print(f"   W={W:5d} g/gc={r:.2f} g={g:.4f} sim Var={v:.3e}  QT-equivalent linear theory={th:.3e} ratio={v/th if np.isfinite(th) else float('nan'):.2f}",flush=True)
    json.dump(R,open('out/alpha_form.json','w'))
