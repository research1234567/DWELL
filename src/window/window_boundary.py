"""Window-corrected first-loss gain eta_c^W(rho,d,W) for P_{W,eta}(z)=z^{W+d+1}+(eta-1)z^{W+d}+rho(eta+1/W)z^W-rho/W.
Candidates: unit-circle crossings from the frequency equation on a fine grid + Brent polish + z=-1 branch.
Certification: exact-arithmetic-style Schur recursion in mpmath (default 60 digits) just below / above the candidate,
and on a probe grid in (0, eta_first). Returns status string; never silently returns a cap when certification fails."""
import numpy as np, mpmath as mp
from scipy.optimize import brentq
mp.mp.dps=60
def schur_mp(eta,rho,d,W,dps=60):
    mp.mp.dps=dps
    r=mp.mpf(repr(float(rho)));e=mp.mpf(repr(float(eta)));n=W+d+2
    a=[mp.mpf(0)]*n;a[0]=mp.mpf(1);a[1]+=e-1;a[d+1]+=r*(e+mp.mpf(1)/W);a[-1]-=r/W
    while len(a)>1:
        k=a[-1]/a[0]
        if abs(k)>=1:return False
        s=a[0]*(1-k*k);a=[(a[j]-k*a[-1-j])/s for j in range(len(a)-1)]
    return True
def crossings(rho,d,W,ngrid=400001):
    def ec(w):
        z=np.exp(1j*w);return -(z-1+rho/W*(z**(-d)-z**(-W-d)))/(1+rho*z**(-d))
    ws=np.linspace(1e-9,np.pi-1e-9,ngrid);v=ec(ws).imag;cand=[]
    for j in np.where(v[:-1]*v[1:]<0)[0]:
        try:w=brentq(lambda x:ec(x).imag,ws[j],ws[j+1],xtol=1e-14)
        except Exception:continue
        z=ec(w)
        if z.real>0 and abs(z.imag)<1e-8:cand.append((float(z.real),float(w)))
    ef=ec(np.pi)
    if ef.real>0 and abs(ef.imag)<1e-8:cand.append((float(ef.real),float(np.pi)))
    return sorted(set(cand))
def first_loss(rho,d,W,probes=24,dps=60):
    """returns dict(status, eta_c, omega, candidates, stable_start)"""
    cand=crossings(rho,d,W);out=dict(candidates=cand)
    ss=schur_mp(1e-9,rho,d,W,dps);out['stable_start']=bool(ss)
    if not ss:out.update(status='NO_STABLE_START',eta_c=None);return out
    for eta,w in cand:
        lo=schur_mp(eta*(1-1e-7),rho,d,W,dps);hi=schur_mp(eta*(1+1e-7),rho,d,W,dps)
        if lo and not hi:
            grid=[eta*x for x in np.linspace(0.02,0.98,probes)]
            if all(schur_mp(g,rho,d,W,dps) for g in grid):
                out.update(status='OK',eta_c=eta,omega=w);return out
            out.update(status='INTERIOR_INSTABILITY_BEFORE_CANDIDATE',eta_c=None);return out
        if not lo:out.update(status='UNSTABLE_BELOW_CANDIDATE_(missed_crossing_or_numerics)',eta_c=None);return out
    out.update(status='NO_CROSSING_FOUND',eta_c=None);return out
if __name__=="__main__":
    import sys,time;sys.path.insert(0,'..')
    from core import Env,gt_crit
    for nm,(a,b,d) in {'R1':(-2,100,80),'R3':(-2,90,120),'R5':(-2,120,150)}.items():
        e=Env(a=a,b=b,d=d);t=time.time();r=first_loss(e.rho,d,500)
        print(nm,'rho %.4f d %d scalar %.5f'%(e.rho,d,gt_crit(e.rho,d,gmax=0.08,n=1400)),r['status'],r['eta_c'],'ncand',len(r['candidates']),'%.0fs'%(time.time()-t),flush=True)
