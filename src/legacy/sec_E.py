"""Generates ALL numbers/tables/figures used in paper1_v2.tex  ->  out/numbers.json, out/tables.tex, figs/*.pdf"""
import json, numpy as np, warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_lyapunov as dlyap
from core import *
plt.rcParams.update({'font.size':8,'axes.grid':True,'grid.alpha':.3,'figure.dpi':150})
N={}; e=Env(); q0,K,gc,c1,h2,h3=flip_normal_form(e)
N.update(a=e.a,b=e.b,sig=e.sig,alpha=ALPHA,qstar=q0,f=e.f,theta=e.theta,K=K,rho=e.rho,gc=gc,c1=c1,h2=h2,h3=h3,gt_c0=2/(1+e.rho))
# ---------- E. stationary frontier at paper calibration
def evalstat(ctrl_fn,seeds=12,T=4000):
    r=[]
    for s in range(seeds):
        o=run(lambda t:e,ctrl_fn(),T,s); r.append([abs(o['cov'].mean()-.9)*100,o['pin'].mean()*100,np.mean((o['q']-q0)**2)*1e4])
    r=np.array(r); return r.mean(0),r.std(0,ddof=1)/np.sqrt(seeds)
fr={}
for g in [0.005,0.01,0.02,0.03,0.05,0.08,0.15,0.30]: fr[f"Fixed $\\gamma$={g}"]=evalstat(lambda g=g:make_fixed(g))
fr["Decaying $\\gamma_0$=0.5"]=evalstat(lambda:make_decay(0.5)); fr["Decaying $\\gamma_0$=1.0"]=evalstat(lambda:make_decay(1.0))
fr["Damped, oracle $K$, $\\gamma_0$=0.15"]=evalstat(lambda:(lambda q,y,m,t,st:q+0.15/K*(m-ALPHA)))
fr["Fixed $\\gamma=0.15/K$ (=0.0558)"]=evalstat(lambda:make_fixed(0.15/K))
N['frontier']={k:[v[0].tolist(),v[1].tolist()] for k,v in fr.items()}
N['lin_mse_theory']={str(g):g*0.09/(2*K-g*K*K) *1e4 for g in [0.03,0.08,0.0558]}

json.dump(N,open('out/sec_E.json','w'),indent=1,default=float); print('sec E ok')
