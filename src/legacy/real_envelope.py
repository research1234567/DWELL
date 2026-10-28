import json, numpy as np
from core import *
K=['all90','all99','low10_90','low10_99','svb90','svb99']
M={k:json.load(open(f'out/real_{k}_marg.json')) for k in K}; Cc={k:json.load(open(f'out/real_{k}.json')) for k in K}
rho_bar=max(max(v['sum_hi'] for v in M.values()),max(v['sum_hi'] for v in Cc.values()))
mins=min(gt_crit(rho_bar,d,gmax=2.0,n=900) for d in range(1,25))
lo,hi=rho_bar,5.0
for _ in range(25):
    m=(lo+hi)/2; v=0.5*min(gt_crit(m,d,gmax=2.0,n=500) for d in range(1,25))
    if v<0.2: hi=m
    else: lo=m
print("rho_bar",rho_bar,"gt_rob",mins,"cap",0.5*mins,"rho*",hi,"ratio",hi/rho_bar)
json.dump(dict(rho_bar=rho_bar,gt_rob=mins,cap=0.5*mins,rho_star=hi,ratio=hi/rho_bar,cells=M,cells_ctrl=Cc),open('out/real_envelope.json','w'),indent=1)
