import sys,json,time,numpy as np;sys.path.insert(0,'..')
from core import Env,gt_crit
import env_bench as eb
from window_boundary import *
def stable_d(eta,rho,d,W):
    a=np.zeros(W+d+2);a[0]=1;a[1]=eta-1;a[d+1]+=rho*(eta+1/W);a[-1]-=rho/W
    while len(a)>1:
        k=a[-1]
        if abs(k)>=1:return False
        a=(a[:-1]-k*a[:0:-1])/(1-k*k)
    return True
res={}
for nm,(a,b,d0) in eb.TRUTH.items():
    e=Env(a=a,b=b,d=d0);rb=1.5*e.rho;db=int(np.ceil(1.5*d0));rows=[]
    for d in range(1,db+1):
        cs=crossings(rb,d,500,ngrid=100001);ss=stable_d(1e-9,rb,d,500)
        first=None
        for eta,w in cs:
            if stable_d(eta*(1-1e-6),rb,d,500) and not stable_d(eta*(1+1e-6),rb,d,500):first=eta;break
        rows.append(dict(d=d,stable_start=bool(ss),eta_first=first,scalar=float(gt_crit(rb,d,gmax=2.0,n=700))))
    res[nm]=dict(rho=e.rho,rho_bar=rb,d_bar=db,rows=rows)
    ok=[r for r in rows if r['stable_start'] and r['eta_first']]
    print(nm,'rho_bar %.3f dbar %d'%(rb,db),'stable_start d:',[r['d'] for r in rows if r['stable_start']][:12],'... n_no_start',sum(not r['stable_start'] for r in rows),flush=True)
    if ok:m=min(ok,key=lambda r:r['eta_first']);print('  min window eta_first %.5f at d=%d ; min scalar %.5f'%(m['eta_first'],m['d'],min(r['scalar'] for r in rows)),flush=True)
json.dump(res,open('../../out/rev3_envelope_scan_double.json','w'),indent=1)
