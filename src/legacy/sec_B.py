"""Generates ALL numbers/tables/figures used in paper1_v2.tex  ->  out/numbers.json, out/tables.tex, figs/*.pdf"""
import json, numpy as np, warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_lyapunov as dlyap
from core import *
plt.rcParams.update({'font.size':8,'axes.grid':True,'grid.alpha':.3,'figure.dpi':150})
N={}; e=Env(); q0,K,gc,c1,h2,h3=flip_normal_form(e)
N.update(a=e.a,b=e.b,sig=e.sig,alpha=ALPHA,qstar=q0,f=e.f,theta=e.theta,K=K,rho=e.rho,gc=gc,c1=c1,h2=h2,h3=h3,gt_c0=2/(1+e.rho))
# ---------- B. delay structure (calibrated) + stability map
dl={}
for d in range(0,9):
    g=gt_crit(e.rho,d); A=companion(g*1.0005,e.rho,d); ev=np.linalg.eigvals(A); lm=ev[np.argmax(abs(ev))]
    dl[d]=dict(gt=g,gQT=g/e.f,ang=float(abs(np.angle(lm))/np.pi))
N['delay_cal']=dl; N['d1_closed_QT']=1/e.theta
fig,ax=plt.subplots(1,2,figsize=(7.2,2.8))
rr=np.concatenate([np.linspace(0.02,1.0,25),np.linspace(1.05,6,60)])
for d,c in [(0,'k'),(1,'C0'),(2,'C1'),(3,'C2'),(6,'C3'),(24,'C4')]:
    ax[0].plot(rr,[gt_crit(r,d,gmax=3,n=1500) if d>0 else 2/(1+r) for r in rr],c=c,label=f'd={d}')
ax[0].axvline(1,c='gray',ls=':'); ax[0].set_ylim(0,2.2); ax[0].set_xlabel('performative ratio $\\rho=\\theta/f$'); ax[0].set_ylabel('critical normalised gain $\\tilde\\gamma_c$'); ax[0].legend(fontsize=6,ncol=2); ax[0].set_title('(a) exact stability boundary')
ax[0].plot([e.rho],[2/(1+e.rho)],'r*',ms=7)
ax[1].set_yscale('log'); ax[1].set_xscale('log')
dd=np.arange(1,41)
for r,c in [(0.5,'C0'),(2,'C1'),(5,'C2'),(20,'C3')]:
    ax[1].plot(dd,[gt_crit(r,int(d),gmax=3,n=1500) for d in dd],'o-',ms=2,c=c,label=f'$\\rho$={r}')
ax[1].plot(dd,[2*np.sin(np.pi/(2*(2*d+1)))/20 for d in dd],'k--',lw=.7,label='$2\\sin(\\pi/(4d{+}2))/\\rho$, $\\rho$=20')
ax[1].set_xlabel('delay $d$'); ax[1].set_ylabel('$\\tilde\\gamma_c$'); ax[1].legend(fontsize=6); ax[1].set_title('(b) delay erodes the margin only if $\\rho>1$')
plt.tight_layout(); plt.savefig('figs/fig_stability_map.pdf'); plt.close()

json.dump(N,open('out/sec_B.json','w'),indent=1,default=float); print('sec B ok')
