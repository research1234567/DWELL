import sys, json, numpy as np, warnings; warnings.filterwarnings("ignore")
from core import *
import env_bench as eb
L=int(sys.argv[1]); NS=int(sys.argv[2]); a_,b_,d_=eb.TRUTH['severe']; panic=Env(a=a_,b=b_,d=d_); capU=eb.robust_cap(3.0,24)
M={'Fixed g=0.02':lambda s:eb.run(panic,None,s,fixed=0.02,blk=L),'DtACI (published)':lambda s:eb.run(panic,np.inf,s,common=False,blk=L),'EG-DtACI (universal)':lambda s:eb.run(panic,capU,s,common=True,blk=L)}
R={}
for k,f in M.items():
    r=np.array([f(s) for s in range(NS)]); R[k]=r.tolist(); print(f"L={L} {k:22s} pinball {r[:,2].mean():.3f}+/-{r[:,2].std(ddof=1)/np.sqrt(NS):.3f} | panic MSE {r[:,4].mean():.0f} | cov|dev| {r[:,0].mean():.2f}/{r[:,1].mean():.2f}",flush=True)
json.dump(R,open(f'out/persist_L{L}.json','w'))
