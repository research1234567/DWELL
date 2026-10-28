import sys, json, numpy as np, warnings; warnings.filterwarnings("ignore")
from core import *
import env_bench as eb
part=sys.argv[1]; S0=int(sys.argv[2]) if len(sys.argv)>2 else 0; NS=int(sys.argv[3]) if len(sys.argv)>3 else 20; a_,b_,d_=eb.TRUTH['severe']; panic=Env(a=a_,b=b_,d=d_); gtc=gt_crit(panic.rho,d_,gmax=0.6,n=800); R={}
def go(label,fn): R[label]=np.array([fn(s) for s in range(S0,S0+NS)]).tolist(); print(label,"done",flush=True)
if part=='ref':
    go('oracle cap',lambda s:eb.run(panic,0.5*gtc,s,common=True)); go('uncapped published',lambda s:eb.run(panic,np.inf,s,common=False))
else:
    kr=float(part)
    for kd in [0.5,1.0,2.0]:
        cap=eb.robust_cap(kr*panic.rho,max(1,int(round(kd*d_)))); go(f'EG rho x{kr}, dbar x{kd}',lambda s,cap=cap:eb.run(panic,cap,s,common=True))
json.dump(R,open(f'out/sens_{part}_{S0}.json' if S0 else f'out/sens_{part}.json','w'))
