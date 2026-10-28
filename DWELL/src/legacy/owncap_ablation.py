"""Own-error + scenario-cap ablation arm, added 2026-10-27 to complete the 2x2 crossing.
Mirrors final_bench2.py (family f2, 50 seeds) and family2_bench.py (family g2, 20 seeds) setup exactly:
same TRUTH parameters, same gt_crit/robust_cap cap construction, same seed ranges (0..49 / 0..19).
Writes out/owncap_{family}_{regime}.json; never modifies the stored benchmark arrays.
Usage: python code/owncap_ablation.py {f2|g2} {mild|severe|vsevere|extreme(f2 only)}"""
import sys, json, numpy as np, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'code'))
from core import *
import env_bench as eb

family, name = sys.argv[1], sys.argv[2]
nseeds = 50 if family=='f2' else 20
if family=='f2':
    a_,b_,d_=eb.TRUTH[name]; panic=Env(a=a_,b=b_,d=d_)
else:
    eb.calm=EnvT(qstar=0.30,rho=0.64,d=1)
    TR={'mild':(0.10,1.85,6),'severe':(0.10,2.40,12),'vsevere':(0.10,2.08,24)}
    q0,rho,d=TR[name]; panic=EnvT(qstar=q0,rho=rho,d=d)
gtc=gt_crit(panic.rho,panic.d,gmax=0.6,n=1500 if panic.d>30 else 800)
dbar=int(np.ceil(1.5*panic.d)); capEG=eb.robust_cap(1.5*panic.rho,dbar)
res=[eb.run(panic,capEG,s,common=False) for s in range(nseeds)]
arr=np.array(res)
out=dict(family=family,regime=name,nseeds=nseeds,capEG=capEG,gtc=gtc,dbar=dbar,
         rho=panic.rho,d=panic.d,
         metrics=['cov_dev_overall','cov_dev_panic','pinball_x100','mse_calm_x1e4','mse_panic_x1e4','gamma_calm','gamma_panic'],
         mean=arr.mean(0).tolist(),se=(arr.std(0,ddof=1)/np.sqrt(len(arr))).tolist(),runs=res)
path=root/f'out/owncap_{family}_{name}.json'
json.dump(out,open(path,'w'),default=float)
print(path,'pinball mean %.4f se %.4f'%(out['mean'][2],out['se'][2]),flush=True)
