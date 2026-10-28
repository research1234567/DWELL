"""Generates ALL numbers/tables/figures used in paper1_v2.tex  ->  out/numbers.json, out/tables.tex, figs/*.pdf"""
import json, numpy as np, warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_lyapunov as dlyap
from core import *
plt.rcParams.update({'font.size':8,'axes.grid':True,'grid.alpha':.3,'figure.dpi':150})
N={}; e=Env(); q0,K,gc,c1,h2,h3=flip_normal_form(e)
N.update(a=e.a,b=e.b,sig=e.sig,alpha=ALPHA,qstar=q0,f=e.f,theta=e.theta,K=K,rho=e.rho,gc=gc,c1=c1,h2=h2,h3=h3,gt_c0=2/(1+e.rho))
# ---------- F. coverage bound check (projection => one-sided)
def cov_run(yseq,g,q0=0.5):
    q=q0;miss=0
    for y in yseq:
        m=float(y>q); miss+=m; q=min(1,max(0,q+g*(m-ALPHA)))
    return miss/len(yseq)
T=2000; g=0.05; B=1/(g*T); rows=[]
rr_=np.random.default_rng(5)
for name,ys in [("y=0 (clipping)",np.zeros(T)),("y=1",np.ones(T)),("alternating 0/1",np.tile([0.,1.],T//2)),("uniform",rr_.uniform(size=T)),("beta(5,1)",rr_.beta(5,1,T)),("regime shift",np.r_[rr_.uniform(0,.3,T//2),rr_.uniform(.7,1,T//2)])]:
    m=cov_run(ys,g); rows.append((name,m,m-ALPHA,B))
N['covbound']=rows

json.dump(N,open('out/sec_F.json','w'),indent=1,default=float); print('sec F ok')
