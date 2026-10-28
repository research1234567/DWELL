import sys, json, numpy as np, warnings; warnings.filterwarnings("ignore")
import logging; logging.disable(logging.CRITICAL)
from field_power import *
H=int(sys.argv[1]); s=float(sys.argv[2]); rho_inj=float(sys.argv[3]); reps=int(sys.argv[4]); perms=int(sys.argv[5]); d_true=3
y_all,hod_all=load(); Q0=float(np.quantile(y_all,TAU)); rej=[]; ests=[]
for rep in range(reps):
    rng=np.random.default_rng(7000+rep); u=s*rng.standard_normal(H); y=y_all[:H].copy()
    if rho_inj>0: y=y*np.exp(np.clip(-(rho_inj/Q0)*np.r_[np.zeros(d_true),u[:-d_true]],-5,5))
    e,_=fit(y,u,hod_all[:H]); null=[fit(y,rng.permutation(u),hod_all[:H])[0] for _ in range(perms)]
    p=(1+np.sum(np.abs(null)>=abs(e)))/(1+perms); rej.append(p<=0.05); ests.append(e)
print(f"Q0={Q0:.5f} H={H} s={s} rho_inj={rho_inj}: mean rho_hat {np.mean(ests):+.4f}  permutation-test rejection rate {np.mean(rej):.2f} ({reps} reps x {perms} perms)")
json.dump(dict(H=H,s=s,rho_inj=rho_inj,rate=float(np.mean(rej)),est=float(np.mean(ests)),reps=reps,perms=perms,Q0=Q0),open(f'out/fieldperm_H{H}_s{s}_r{rho_inj}.json','w'))
