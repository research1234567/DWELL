import sys,time,json,hashlib
from pathlib import Path
import numpy as np
from scipy.special import ndtr,ndtri,expit,logit
from scipy.optimize import brentq
from scipy.stats import t as tdist
sys.path.insert(0,str(Path(__file__).parents[1]/'conformal_feedback_revised/code'))
from core import Env,gt_crit,companion
ROOT=Path(__file__).parent
SC=[('calm',-.5,3,1),('mild',-1,24,6),('severe',-1,48,12)]
RAT=np.array([.25,.5,.75,1,1.25,1.5,2.])
def stable(eta,rho,d,W):
    a=np.zeros(W+d+2); a[0]=1;a[1]=eta-1;a[d+1]+=rho*(eta+1/W);a[-1]-=rho/W
    while len(a)>1:
        k=a[-1]
        if abs(k)>=1:return False
        a=(a[:-1]-k*a[:0:-1])/(1-k*k)
    return True

def roll_bounds(rho,d,W):
    def ec(w):
        z=np.exp(1j*w)
        return -(z-1+rho/W*(z**(-d)-z**(-W-d)))/(1+rho*z**(-d))
    ws=np.linspace(1e-9,np.pi-1e-9,20001); v=ec(ws).imag; cand=[]
    for j in np.where(v[:-1]*v[1:]<0)[0]:
        w=brentq(lambda w:ec(w).imag,ws[j],ws[j+1],xtol=1e-12);eta=ec(w).real
        if eta>0 and abs(ec(w).imag)<1e-7:cand.append((float(eta),float(w)))
    ef=ec(np.pi)
    if ef.real>0 and abs(ef.imag)<1e-7:cand.append((float(ef.real),float(np.pi)))
    cand=sorted(cand)
    for eta,w in cand:
        if stable(eta*(1-1e-6),rho,d,W) and not stable(eta*(1+1e-6),rho,d,W):
            return dict(first_loss=eta,omega=w,stable_near_zero=stable(1e-8,rho,d,W),all_crossings=cand)
    return dict(first_loss=None,stable_near_zero=stable(1e-8,rho,d,W),all_crossings=cand)

def direct(e,gc):
    rows=[]
    for ratio in RAT:
        eta=ratio*gc;A=companion(eta,e.rho,e.d);ev,V=np.linalg.eig(A);j=np.argmax(abs(ev));lam=ev[j];vec=V[:,j].real;vec/=np.linalg.norm(vec)
        horizon=min(200,max(8,int(4.6/max(abs(np.log(abs(lam))),.025))))
        for amp in [1e-6,1e-5]:
            u=amp*e.qstar*vec;ql=e.qstar+u;al=.1-e.f*u; lin=u.copy()
            qhist=list(ql[::-1]); ahist=list(al[::-1]);quants=list(ql[::-1]);
            h=e.a-e.b*e.qstar
            n=0
            for n in range(1,horizon+1):
                q=qhist[-1];qd=qhist[-1-e.d];qn=q+eta/e.f*(.9-ndtr((logit(q)-e.a+e.b*qd)/e.sig));qhist.append(qn)
                a=ahist[-1];qf=expit(h+e.sig*ndtri(1-a));qfd=quants[-1-e.d];an=a+eta*(ndtr((logit(qf)-e.a+e.b*qfd)/e.sig)-.9);ahist.append(an);quants.append(expit(h+e.sig*ndtri(1-an)))
                lin=A@lin
                if np.linalg.norm(np.array(qhist[-e.d-1:])-e.qstar)>1e-3*e.qstar or np.linalg.norm(lin)<1e-13:break
            actual=np.linalg.norm(np.array(qhist[-e.d-1:])-e.qstar)
            frozen=np.linalg.norm(np.array(quants[-e.d-1:])-e.qstar)
            rows.append(dict(ratio=float(ratio),amp=amp,n=n,spectral_radius=float(abs(lam)),omega=float(abs(np.angle(lam))),predicted_norm=float(np.linalg.norm(lin)),nonlinear_norm=float(actual),frozen_norm=float(frozen),initial_norm=amp*e.qstar))
    return rows

def jaccheck(e,W,eta):
    rng=np.random.default_rng(892); hist=rng.normal(size=W+e.d+1);hist/=max(abs(hist))
    def predicted(u):return (1-eta)*u[-1]-e.rho*(eta+1/W)*u[-1-e.d]+e.rho/W*u[0]
    out=[]
    for eps in [1e-5,1e-6,1e-7]:
        q=e.qstar+eps*hist; old=q[:W];new=q[1:W+1];current=q[-1];resp=q[-1-e.d]
        v=.9-np.mean(ndtr((logit(current)-e.a+e.b*old)/e.sig)) # a-alpha
        vn=v+eta*(ndtr((logit(current)-e.a+e.b*resp)/e.sig)-.9)
        nextq=brentq(lambda x:np.mean(ndtr((logit(x)-e.a+e.b*new)/e.sig))-(.9-vn),1e-8,1-1e-8,xtol=1e-15)
        out.append(dict(epsilon=eps,finite_difference=(nextq-e.qstar)/eps,predicted=predicted(hist)))
    return out

def empirical(e,gc,W,N=64,T=4000):
    # Vectorize all gains, perturbations, seeds. One new common innovation per seed/time.
    rng=np.random.default_rng(47001);k=len(RAT);p=5; perturb=np.array([0,-.002,.002,-.01,.01]);shape=(k,p,N)
    z0=rng.standard_normal((N,W));win=np.broadcast_to(expit(e.a-e.b*e.qstar+e.sig*z0),(k,p,N,W)).copy()
    a=np.broadcast_to(.1+perturb[None,:,None],shape).copy();hist=np.full((e.d+1,)+shape,e.qstar)
    eta=(RAT*gc)[:,None,None]; qrecord=np.empty((T,)+shape,dtype=np.float64)
    sum_mse=np.zeros(shape);sum_q=np.zeros(shape);sum_pin=np.zeros(shape);sum_out=np.zeros(shape);sum_cross=np.zeros(shape);last=None
    seeds_ix=np.arange(N)[None,None,:];ratio_ix=np.arange(k)[:,None,None];p_ix=np.arange(p)[None,:,None]
    for t in range(T):
        # numpy linear quantile via partition at all needed order indices, exact per-row interpolation.
        lvl=1-np.clip(a,0,1);ind=lvl*(W-1);lo=np.floor(ind).astype(int);hi=np.ceil(ind).astype(int)
        # full sort vectorized is faster than many separately requested partition indices.
        s=np.sort(win,axis=-1);ql=s[ratio_ix,p_ix,seeds_ix,lo];qh=s[ratio_ix,p_ix,seeds_ix,hi];q=ql+(ind-lo)*(qh-ql)
        hist[t%(e.d+1)]=q;qr=hist[(t-e.d)%(e.d+1)]
        z=rng.standard_normal(N);y=expit(e.a-e.b*qr+e.sig*z[None,None,:]);err=(y>q)
        qrecord[t]=q
        if t>=1000:
            sum_mse+=(q-e.qstar)**2;sum_q+=q;sum_pin+=(y-q)*(.9-(y<q));sum_out+=((a<0)|(a>1))
            if last is not None:sum_cross+=(q-e.qstar)*(last-e.qstar)
        last=q.copy();a+=eta*(.1-err);win[...,t%W]=y
    n=T-1000
    metrics=dict(mse=sum_mse/n,mean_q=sum_q/n,pinball=sum_pin/n,outside_fraction=sum_out/n,raw_lag1=sum_cross/np.maximum(sum_mse,1e-30))
    metrics['variance']=metrics['mse']-(metrics['mean_q']-e.qstar)**2
    np.savez_compressed(ROOT/f'empirical_{e.d}_{W}.npz',mean_q_time=qrecord.mean(axis=-1),paired_small=(qrecord[:,:,2]-qrecord[:,:,1]).mean(axis=0),paired_large=(qrecord[:,:,4]-qrecord[:,:,3]).mean(axis=0),**metrics)
    rows=[]
    crit=tdist.ppf(.975,N-1)
    def summary(x):return dict(mean=float(x.mean()),se=float(x.std(ddof=1)/np.sqrt(N)),ci95=[float(x.mean()-crit*x.std(ddof=1)/np.sqrt(N)),float(x.mean()+crit*x.std(ddof=1)/np.sqrt(N))])
    for j,ratio in enumerate(RAT):
        row=dict(ratio=float(ratio),eta=float(ratio*gc),metrics={m:summary(v[j,0]) for m,v in metrics.items()})
        pairs=[]
        for neg,pos,delta in [(1,2,.002),(3,4,.01)]:
            response=(qrecord[:,j,pos]-qrecord[:,j,neg])/(2*delta)
            pairs.append(dict(delta=delta,last500_response=summary(response[-500:].mean(axis=0)),early100_response=summary(response[:100].mean(axis=0)),rms_mean_last500=float(np.sqrt(np.mean(response[-500:].mean(axis=1)**2)))))
        row['paired_response']=pairs;rows.append(row)
    # Matched aggregate comparison fixed in protocol, not a boundary estimator.
    compare={m:summary(v[4:,0].mean(axis=0)-v[:3,0].mean(axis=0)) for m,v in metrics.items()}
    return dict(rows=rows,above_minus_below=compare)

if __name__=='__main__':
    smoke='--smoke' in sys.argv;start=time.time();results=[]
    for name,a,b,d in SC[:1] if smoke else SC:
        e=Env(a=a,b=b,d=d);gc=gt_crit(e.rho,d);r=dict(name=name,a=a,b=b,d=d,qstar=e.qstar,f=e.f,rho=e.rho,threshold_eta_c=gc)
        if not smoke:r['direct']=direct(e,gc)
        r['windows']=[]
        for W in ([500] if smoke else [100,500]):
            t=time.time();w=dict(W=W)
            if not smoke:w.update(population=roll_bounds(e.rho,d,W),jacobian_check=jaccheck(e,W,.5*gc))
            w['empirical']=empirical(e,gc,W,N=8 if smoke else 64,T=1100 if smoke else 4000);w['seconds']=time.time()-t;r['windows'].append(w);print(name,W,w['seconds'],flush=True)
        results.append(r)
    out=dict(smoke=smoke,elapsed=time.time()-start,results=results)
    json.dump(out,open(ROOT/('smoke.json' if smoke else 'results.json'),'w'),indent=2)
