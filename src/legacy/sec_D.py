"""Generates ALL numbers/tables/figures used in paper1_v2.tex  ->  out/numbers.json, out/tables.tex, figs/*.pdf"""
import json, numpy as np, warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_lyapunov as dlyap
from core import *
plt.rcParams.update({'font.size':8,'axes.grid':True,'grid.alpha':.3,'figure.dpi':150})
N={}; e=Env(); q0,K,gc,c1,h2,h3=flip_normal_form(e)
N.update(a=e.a,b=e.b,sig=e.sig,alpha=ALPHA,qstar=q0,f=e.f,theta=e.theta,K=K,rho=e.rho,gc=gc,c1=c1,h2=h2,h3=h3,gt_c0=2/(1+e.rho))
# ---------- D. linear Lyapunov variance vs simulation
def lin_var(env,g,d):
    gt=g*env.f; A=companion(gt,env.rho,d); B=np.zeros((d+1,1)); B[0,0]=g*np.sqrt(env.alpha*(1-env.alpha))
    return np.inf if max(abs(np.linalg.eigvals(A)))>=1 else dlyap(A,B@B.T)[0,0]
vrows=[]
for (a_,b_,d) in [(-0.5,3.0,0),(-1.0,24.0,6)]:
    e0=Env(a=a_,b=b_); gcr=gt_crit(e0.rho,d)/e0.f if d>0 else 2/e0.K; ee=Env(a=a_,b=b_,d=d)
    for r in [0.1,0.5,0.9,2.0]:
        g=r*gcr; sim=np.mean([np.mean((run(lambda t:ee,make_fixed(g),20000,s)["q"][2000:]-ee.qstar)**2) for s in range(5)])
        vrows.append((b_,d,gcr,r,g,sim,lin_var(ee,g,d)))
N['var']=vrows

json.dump(N,open('out/sec_D.json','w'),indent=1,default=float); print('sec D ok')
