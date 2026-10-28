"""Independent reproduction of the 'does the theory predict where a constant step is unsafe' test (other-AI Test 3), using this package's own code.
Hypothesis under test: when gt_c << 0.02, gamma_th = 0.5*gt_c(true rho,d) beats gamma = 0.02."""
import sys, json, numpy as np, warnings; warnings.filterwarnings("ignore")
from core import *
import env_bench as eb
REG={'R1':(-2.0,100.0,80),'R3':(-2.0,90.0,120),'R5':(-2.0,120.0,150)}
name=sys.argv[1]; NS=int(sys.argv[2]); a,b,d=REG[name]; panic=Env(a=a,b=b,d=d)
gtc=gt_crit(panic.rho,d,gmax=0.08,n=1400); gth=0.5*gtc
A=np.array([eb.run(panic,None,s,fixed=0.02) for s in range(6000,6000+NS)]); B=np.array([eb.run(panic,None,s,fixed=gth) for s in range(6000,6000+NS)])
dpin=100*(B[:,2]-A[:,2])/A[:,2]; ci=1.96*dpin.std(ddof=1)/np.sqrt(NS)
print(f"{name}: rho={panic.rho:.2f} d={d} gt_c={gtc:.4f} gamma_th={gth:.4f} | pinball gamma_th vs 0.02: {dpin.mean():+.1f}% (95% CI +/-{ci:.1f}) | panic cov dev 0.02: {A[:,1].mean():.2f}pp th: {B[:,1].mean():.2f}pp | panic MSE 0.02: {A[:,4].mean():.0f} th: {B[:,4].mean():.0f}",flush=True)
json.dump(dict(name=name,rho=panic.rho,d=d,gtc=gtc,gth=gth,dpin=dpin.tolist(),covA=A[:,1].tolist(),covB=B[:,1].tolist()),open(f'out/verify3_{name}.json','w'))
