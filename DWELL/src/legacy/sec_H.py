import json, numpy as np
from core import *
e=Env(); q=np.linspace(1e-4,1-1e-4,200001); q=q[abs(q-e.qstar)>1e-6]
h=(1-ALPHA)-e.F(q,q); r=-h/(q-e.qstar)
N=dict(kappa=float(r.min()),L=float(np.max(abs(h)/abs(q-e.qstar))),kappa_argmin=float(q[np.argmin(r)]),L_argmax=float(q[np.argmax(abs(h)/abs(q-e.qstar))]))
N['gamma_max']=2*N['kappa']/N['L']**2
for g in [0.005,0.01,0.02,0.05]:
    N[f'bound_{g}']=g*0.25/(2*N['kappa']-g*N['L']**2); N[f'bound09_{g}']=g*0.09/(2*N['kappa']-g*N['L']**2)
N['local_K']=e.K
json.dump(N,open('out/sec_H.json','w'),indent=1); print(N)
