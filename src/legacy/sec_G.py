"""Generates ALL numbers/tables/figures used in paper1_v2.tex  ->  out/numbers.json, out/tables.tex, figs/*.pdf"""
import json, numpy as np, warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_lyapunov as dlyap
from core import *
plt.rcParams.update({'font.size':8,'axes.grid':True,'grid.alpha':.3,'figure.dpi':150})
N={}; e=Env(); q0,K,gc,c1,h2,h3=flip_normal_form(e)
N.update(a=e.a,b=e.b,sig=e.sig,alpha=ALPHA,qstar=q0,f=e.f,theta=e.theta,K=K,rho=e.rho,gc=gc,c1=c1,h2=h2,h3=h3,gt_c0=2/(1+e.rho))
# ---------- G. identification accuracy
from statsmodels.regression.quantile_regression import QuantReg
idrows=[]
for (a_,b_,d) in [(-1.0,24.0,6),(-0.5,3.0,1)]:
    ee=Env(a=a_,b=b_,d=d)
    for s_ in [0.01,0.02]:
        for W in [500,2000]:
            ests=[]
            for sd in range(6):
                rng=np.random.default_rng(10+sd); q=ee.qstar; D=10; T=W+D+200; dep=[q]*50; eps=np.zeros(T); ys=np.zeros(T); qs=np.zeros(T)
                for t in range(T):
                    eps[t]=s_*rng.standard_normal(); qd=q+eps[t]; dep.append(qd); y=ee.draw(dep[-1-d],rng); ys[t]=y; qs[t]=q
                    q=min(1,max(0,q+0.003*(float(y>qd)-ALPHA)))
                idx=np.arange(T-W,T); X=np.column_stack([np.ones(W),qs[idx]]+[eps[idx-k] for k in range(D+1)])
                r=QuantReg(ys[idx],X).fit(q=0.9,max_iter=500); rho=-r.params[3:]; ests.append((rho.sum(),int(np.argmax(rho))+1))
            ests=np.array(ests); idrows.append((b_,d,ee.rho,s_,W,ests[:,0].mean(),ests[:,0].std(ddof=1),float(np.mean(ests[:,1]==d))))
N['ident']=idrows

json.dump(N,open('out/sec_G.json','w'),indent=1,default=float); print('sec G ok')
