"""Generates ALL numbers/tables/figures used in paper1_v2.tex  ->  out/numbers.json, out/tables.tex, figs/*.pdf"""
import json, numpy as np, warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_lyapunov as dlyap
from core import *
plt.rcParams.update({'font.size':8,'axes.grid':True,'grid.alpha':.3,'figure.dpi':150})
N={}; e=Env(); q0,K,gc,c1,h2,h3=flip_normal_form(e)
N.update(a=e.a,b=e.b,sig=e.sig,alpha=ALPHA,qstar=q0,f=e.f,theta=e.theta,K=K,rho=e.rho,gc=gc,c1=c1,h2=h2,h3=h3,gt_c0=2/(1+e.rho))
# ---------- A. bifurcation
gs=np.linspace(0.30,1.10,161); orb=[skeleton_orbit(e,g,n_burn=2500,n_keep=40) for g in gs]
lam=[lyapunov(e,g,n_burn=800,n=1500) for g in gs]
amp=[]
for g in [0.76,0.78,0.80,0.85,0.90]:
    o=skeleton_orbit(e,g); A=(o.max()-o.min())/2; th=np.sqrt(4*(g-gc)/(gc*c1)); amp.append((g,A,th,100*abs(A-th)/A))
N['amp']=amp
fig,ax=plt.subplots(1,2,figsize=(7.2,2.6))
for g,o in zip(gs,orb): ax[0].plot([g]*len(o),o,'k.',ms=.6)
gg=np.linspace(gc,1.1,100); ax[0].plot(gg,q0+np.sqrt(4*(gg-gc)/(gc*c1)),'r-',lw=.8); ax[0].plot(gg,q0-np.sqrt(4*(gg-gc)/(gc*c1)),'r-',lw=.8,label='normal form $q^*\\pm\\sqrt{4\\mu/(\\gamma_c c_1)}$')
ax[0].axvline(gc,c='b',ls='--',lw=.8); ax[0].set_xlabel('step size $\\gamma$'); ax[0].set_ylabel('$q$'); ax[0].legend(loc='upper left',fontsize=6); ax[0].set_title('(a) orbit diagram of mean skeleton')
ax[1].plot(gs,lam,'k-',lw=.8); ax[1].axhline(0,c='gray'); ax[1].axvline(gc,c='b',ls='--',lw=.8); ax[1].set_xlabel('step size $\\gamma$'); ax[1].set_ylabel('Lyapunov exponent'); ax[1].set_title('(b) exponent touches 0 only at $\\gamma_c$')
plt.tight_layout(); plt.savefig('figs/fig_bifurcation.pdf'); plt.close()
N['lyap_at']={str(g):lyapunov(e,g) for g in [0.60,0.7435,0.80,0.90,1.00]}
N['lyap_max_above']=float(max(l for g,l in zip(gs,lam) if g>gc+0.01)); N['lyap_min_below_gc']=float(max(l for g,l in zip(gs,lam) if g<gc-0.02))

json.dump(N,open('out/sec_A.json','w'),indent=1,default=float); print('sec A ok')
