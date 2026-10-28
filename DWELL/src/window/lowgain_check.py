import sys,json,numpy as np;sys.path.insert(0,'..')
from core import Env
import env_bench as eb
from window_boundary import schur_mp
def Bmax(rho,d,W):
    c=np.zeros(W+d+1);c[0]=1
    for k in range(W):c[W+d-k]+=rho/W   # z^{W+d}+ (rho/W) sum_{k<W} z^k
    r=np.roots(c);return float(max(abs(r)))
def stable_d(eta,rho,d,W):
    a=np.zeros(W+d+2);a[0]=1;a[1]=eta-1;a[d+1]+=rho*(eta+1/W);a[-1]-=rho/W
    while len(a)>1:
        k=a[-1]
        if abs(k)>=1:return False
        a=(a[:-1]-k*a[:0:-1])/(1-k*k)
    return True
out={}
e=Env(a=-2,b=120,d=150);etas=np.concatenate([np.logspace(-6,0,300),np.linspace(1,2.5,150)])
st=[float(x) for x in etas if stable_d(x,e.rho,150,500)]
out['R5']=dict(B_W_max_root_modulus=Bmax(e.rho,150,500),n_eta_probed=len(etas),stable_eta_found=st[:5],n_stable=len(st))
ex=Env(a=-2,b=120,d=60);rb=1.5*ex.rho;rows=[]
S=json.load(open('../../out/rev3_envelope_scan_double.json'))['extreme']['rows']
for r in S:
    if not r['stable_start']:
        m=Bmax(rb,r['d'],500);ns=sum(stable_d(x,rb,r['d'],500) for x in etas);rows.append((r['d'],round(m,6),ns))
out['extreme_nostart']=rows
json.dump(out,open('../../out/rev3_lowgain_check.json','w'),indent=1);print(json.dumps(out)[:1500])
