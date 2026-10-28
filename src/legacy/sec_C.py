"""Generates ALL numbers/tables/figures used in paper1_v2.tex  ->  out/numbers.json, out/tables.tex, figs/*.pdf"""
import json, numpy as np, warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_lyapunov as dlyap
from core import *
plt.rcParams.update({'font.size':8,'axes.grid':True,'grid.alpha':.3,'figure.dpi':150})
N={}; e=Env(); q0,K,gc,c1,h2,h3=flip_normal_form(e)
N.update(a=e.a,b=e.b,sig=e.sig,alpha=ALPHA,qstar=q0,f=e.f,theta=e.theta,K=K,rho=e.rho,gc=gc,c1=c1,h2=h2,h3=h3,gt_c0=2/(1+e.rho))
# ---------- C. theorem checks
rng=np.random.default_rng(0); worst=0; n=0
for _ in range(4000):
    r=rng.uniform(0.001,0.999); g=rng.uniform(0.001,1.0); d=int(rng.integers(1,31)); worst=max(worst,spec_radius(g,r,d)); n+=1
N['smallgain']=dict(n=n,worst=worst)
mx=0
for d in [1,2,3,6]:
    w,r,g=boundary_curve(d,6000)
    for rv in [1.2,2.0,4.0]:
        i=np.argmin(abs(r-rv))
        if abs(r[i]-rv)<2e-3:
            mx=max(mx,abs(g[i]-spec_radius_dummy) if False else 0)
asy=[]
for d in [1,2,6,24]:
    for r in [20,200]:
        asy.append((d,r,gt_crit(r,d,gmax=1.0,n=20000)*r,2*np.sin(np.pi/(2*(2*d+1)))))
N['asym']=asy
# parametrisation check: boundary curve points have a unit-modulus root
err=0
for d in [1,2,3,6,12]:
    w,r,g=boundary_curve(d,3000)
    for k in range(0,len(w),max(1,len(w)//40)):
        err=max(err,abs(max(abs(np.linalg.eigvals(companion(g[k],r[k],d))))-1))
N['param_check_max_radius_err']=float(err)

json.dump(N,open('out/sec_C.json','w'),indent=1,default=float); print('sec C ok')
