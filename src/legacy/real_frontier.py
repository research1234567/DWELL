import sys, json, numpy as np, warnings; warnings.filterwarnings("ignore")
from real_replay import *
from core import gt_crit
Y=series(); G=[0.003,0.005,0.01,0.02,0.03,0.05,0.1,0.2,0.4]; R={}
for (rho,d) in [(0.2,6),(1.0,6),(2.5,6),(2.5,12)]:
    gtc=gt_crit(rho,d,gmax=2.0,n=1500) if rho>0 else np.inf; row=[]
    for g in G:
        r=np.array([replay(Y[a],rho,d,fixed=g) for a in sorted(Y)]); row.append((g,float(r[:,2].mean()),float(abs(r[:,0]-.9).mean()*100)))
    R[f"{rho}_{d}"]=dict(gtc=float(gtc),rows=row)
    print(f"rho={rho} d={d} gtc={gtc:.3f} | "+" ".join(f"g={g}:{p:.2f}" for g,p,_ in row),flush=True)
json.dump(R,open('out/real_frontier.json','w'),indent=1)
