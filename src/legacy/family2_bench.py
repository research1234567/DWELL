"""Headline benchmark on a second, structurally different family (heavy-tailed multiplicative response; see core.EnvT)."""
import sys, json, numpy as np, warnings; warnings.filterwarnings("ignore")
from core import *
import env_bench as eb
eb.calm=EnvT(qstar=0.30,rho=0.64,d=1)
TR={'mild':(0.10,1.85,6),'severe':(0.10,2.40,12),'vsevere':(0.10,2.08,24)}
name=sys.argv[1]; s0=int(sys.argv[2]); n=int(sys.argv[3]); q0,rho,d=TR[name]; panic=EnvT(qstar=q0,rho=rho,d=d)
gtc=gt_crit(panic.rho,d,gmax=0.6,n=800); dbar=int(np.ceil(1.5*d)); capEG=eb.robust_cap(1.5*panic.rho,dbar); capU=eb.robust_cap(3.0,24)
M={'truth':dict(rho=panic.rho,d=d,gtc=gtc,dbar=dbar,capEG=capEG,capU=capU)}; seeds=range(s0,s0+n)
for g in [0.005,0.01,0.02,0.05]: M[f'Fixed g={g}']=[eb.run(panic,None,s,fixed=g) for s in seeds]
M['DtACI (published)']=[eb.run(panic,np.inf,s,common=False) for s in seeds]
M['DtACI (deployed error)']=[eb.run(panic,np.inf,s,common=True) for s in seeds]
M['EG-DtACI (scenario envelope)']=[eb.run(panic,capEG,s,common=True) for s in seeds]
M['EG-DtACI (universal envelope)']=[eb.run(panic,capU,s,common=True) for s in seeds]
json.dump(M,open(f'out/g2_{name}_{s0}.json','w'),default=float)
for k,v in M.items():
    if k!='truth': print(f"{name:8s} {k:30s} pinball {np.mean(np.array(v)[:,2]):.3f} | panic MSE {np.mean(np.array(v)[:,4]):7.1f} | cov|dev| {np.mean(np.array(v)[:,0]):.2f}/{np.mean(np.array(v)[:,1]):.2f}",flush=True)
