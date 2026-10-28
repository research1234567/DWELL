import sys, json, numpy as np, warnings; warnings.filterwarnings("ignore")
from core import *
import env_bench as eb
name=sys.argv[1]; s0=int(sys.argv[2]); n=int(sys.argv[3]); a_,b_,d_=eb.TRUTH[name]; panic=Env(a=a_,b=b_,d=d_)
gtc=gt_crit(panic.rho,d_,gmax=0.6,n=1500 if d_>30 else 800); dbar=int(np.ceil(1.5*d_)); capEG=eb.robust_cap(1.5*panic.rho,dbar); capU=eb.robust_cap(3.0,24)
M={'truth':dict(rho=panic.rho,d=d_,gtc=gtc,dbar=dbar,capEG=capEG,capU=capU,oracle_cap=0.5*gtc)}
seeds=range(s0,s0+n)
for g in [0.005,0.01,0.02,0.05]: M[f'Fixed g={g}']=[eb.run(panic,None,s,fixed=g) for s in seeds]
M['DtACI (uncapped, own error)']=[eb.run(panic,np.inf,s,common=False) for s in seeds]
M['DtACI (uncapped, deployed error)']=[eb.run(panic,np.inf,s,common=True) for s in seeds]
M['EG-DtACI (scenario envelope 1.5x)']=[eb.run(panic,capEG,s,common=True) for s in seeds]
M['EG-DtACI (universal envelope)']=[eb.run(panic,capU,s,common=True) for s in seeds]
json.dump(M,open(f'out/f2_{name}_{s0}.json','w'),default=float); print(name,s0,"done",flush=True)
