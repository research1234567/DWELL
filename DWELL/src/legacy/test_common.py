import sys, numpy as np, warnings; warnings.filterwarnings("ignore")
from core import *
import env_bench as eb
for name in sys.argv[1:]:
    a_,b_,d_=eb.TRUTH[name]; panic=Env(a=a_,b=b_,d=d_); capU=eb.robust_cap(3.0,24)
    for lab,kw in [('uncapped own-err',dict(cap=np.inf)),('uncapped common-err',dict(cap=np.inf,common=True)),('EG-univ own-err',dict(cap=capU)),('EG-univ common-err',dict(cap=capU,common=True))]:
        rs=np.array([eb.run(panic,kw['cap'],s,common=kw.get('common',False)) for s in range(6)]); m=rs.mean(0)
        print(f"{name:8s} {lab:22s} pin {m[2]:.3f} | panic MSE {m[4]:7.1f} | cov|dev| {m[0]:.2f}/{m[1]:.2f}",flush=True)
