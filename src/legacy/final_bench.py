import sys, json, numpy as np, warnings; warnings.filterwarnings("ignore")
from core import *
import env_bench as eb
name=sys.argv[1]; NS=int(sys.argv[2]); a_,b_,d_=eb.TRUTH[name]; panic=Env(a=a_,b=b_,d=d_)
gtc=gt_crit(panic.rho,d_,gmax=0.6,n=1500 if d_>30 else 800); dbar=int(np.ceil(1.5*d_)); capEG=eb.robust_cap(1.5*panic.rho,dbar); capU=eb.robust_cap(3.0,24)
R={'truth':dict(rho=panic.rho,d=d_,gtc=gtc,dbar=dbar,capEG=capEG,capU=capU,oracle_cap=0.5*gtc)}; print(R['truth'],flush=True)
def go(label,fn):
    rs=np.array([fn(s) for s in range(NS)]); R[label]=dict(mean=rs.mean(0).tolist(),se=(rs.std(0,ddof=1)/np.sqrt(NS)).tolist(),rows=rs.tolist()); m=R[label]['mean']
    print(f"{label:34s} pin {m[2]:.3f} | panic MSE {m[4]:7.1f} | cov|dev| {m[0]:.2f}/{m[1]:.2f}",flush=True)
for g in [0.005,0.01,0.02,0.05]: go(f'Fixed g={g}',lambda s,g=g:eb.run(panic,None,s,fixed=g))
go('DtACI (uncapped)',lambda s:eb.run(panic,np.inf,s))
go('EG-DtACI (scenario envelope 1.5x)',lambda s:eb.run(panic,capEG,s))
go('EG-DtACI (universal envelope)',lambda s:eb.run(panic,capU,s))
json.dump(R,open(f'out/final_{name}.json','w'),indent=1)
