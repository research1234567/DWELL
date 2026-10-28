import numpy as np
from scipy.linalg import solve_discrete_lyapunov as dl
from core import *
def lin_var(e,g,d):
    gt=g*e.f; A=companion(gt,e.rho,d); B=np.zeros((d+1,1)); B[0,0]=g*np.sqrt(e.alpha*(1-e.alpha))
    if max(abs(np.linalg.eigvals(A)))>=1: return np.inf
    return dl(A,B@B.T)[0,0]
for (a,b,d) in [(-0.5,3.0,0),(-1.0,24.0,0),(-1.0,24.0,6),(-1.0,24.0,12)]:
    e0=Env(a=a,b=b); gc=gt_crit(e0.rho,d)/e0.f; e=Env(a=a,b=b,d=d)
    print(f"b={b} d={d} gQT_c={gc:.4f}")
    for r in [0.1,0.3,0.5,0.7,0.9]:
        g=r*gc; sim=np.mean([np.mean((run(lambda t:e,make_fixed(g),40000,s)['q'][4000:]-e.qstar)**2) for s in range(6)])
        th=lin_var(e,g,d); print(f"   r={r:.1f} sim={sim:.3e} linear-theory={th:.3e} ratio={sim/th:.2f}")
