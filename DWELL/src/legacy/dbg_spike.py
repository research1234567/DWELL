import numpy as np, warnings; warnings.filterwarnings("ignore")
import logging; logging.disable(logging.CRITICAL)
import alpha_bench as ab
from alpha_bench import *
# trace the estimator inside the closed loop (identified variant), one seed
sp_holder={}
orig=ab.Spike.update
def upd(self,t,eps,y,q):
    orig(self,t,eps,y,q); sp_holder.setdefault('tr',[]).append((t,self.d,self.rho,self.cap))
ab.Spike.update=upd
for s in range(2):
    sp_holder.clear(); ab.run_one('identified',s); tr=sp_holder['tr']
    print("seed",s)
    for (t,d,rho,cap) in [x for x in tr if x[0]%200==0]:
        reg='panic' if ((t-1)//800)%2==1 else 'calm '
        print(f"  t={t:5d} {reg} d_hat={d:2d} rho_hat={rho:5.2f} cap={cap:5.2f}   (true d={6 if reg=='panic' else 1}, rho={panic.rho if reg=='panic' else calm.rho:.2f}, gt_c={GT[id(panic if reg=='panic' else calm)]:.3f})")
