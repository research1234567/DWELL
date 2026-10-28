import json, numpy as np
from core import *
from alpha_theory import alpha_system
N={'thr':[]}
for (a_,b_,d) in [(-0.5,3.0,0),(-1.0,24.0,6)]:
    e=Env(a=a_,b=b_,d=d); gtc=gt_crit(e.rho,d) if d>0 else 2/(1+e.rho)
    for W in [50,200,1000]:
        gs=np.linspace(0.3,1.6,53)*gtc; crit=next((g for g in gs if max(abs(np.linalg.eigvals(alpha_system(e,g,W)[0])))>=1),np.inf)
        N['thr'].append((b_,d,W,float(crit),float(gtc),float(crit/gtc)))
N['alpha_var']=json.load(open('out/alpha_theory.json'))
json.dump(N,open('out/sec_I.json','w')); print(N['thr'])
