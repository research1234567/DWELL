"""Core library for Reflexive Conformal Prediction (Paper 1). Everything in the paper is generated from here.
DGP: logit(y_t) ~ N(a - b*q_{t-d}, sigma^2)   (performative logit-normal family; q = deployed threshold)
Controllers use the quantile-tracking (QT) form  q_{t+1} = Pi[0,1]( q_t + gamma_t * (1{y_t>q_t} - alpha) )."""
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import mpmath as mp

ALPHA = 0.10
logit = lambda p: np.log(p/(1-p))
expit = lambda x: 1/(1+np.exp(-x))

class Env:
    def __init__(s, a=-0.5, b=3.0, sig=0.5, d=0, alpha=ALPHA):
        s.a,s.b,s.sig,s.d,s.alpha=a,b,sig,d,alpha
        s.qstar=brentq(lambda q:(1-alpha)-s.F(q,q),1e-9,1-1e-9)
        z=(logit(s.qstar)-(a-b*s.qstar))/sig
        s.f=norm.pdf(z)/(sig*s.qstar*(1-s.qstar)); s.theta=norm.pdf(z)*b/sig
        s.K=s.f+s.theta; s.rho=s.theta/s.f
    def F(s,y,q): return norm.cdf((logit(y)-(s.a-s.b*q))/s.sig)
    def draw(s,q_resp,rng): return expit(s.a-s.b*q_resp+s.sig*rng.standard_normal())
    def mp_derivs(s):
        mp.mp.dps=40; A,B,S=map(mp.mpf,(str(s.a),str(s.b),str(s.sig)))
        H=lambda q: (1-mp.mpf(str(s.alpha)))-mp.ncdf((mp.log(q/(1-q))-(A-B*q))/S)
        q0=mp.findroot(H,(mp.mpf('0.01'),mp.mpf('0.99')),solver='anderson')
        return q0,[mp.diff(H,q0,n) for n in (1,2,3)]

def flip_normal_form(env):
    q0,(h1,h2,h3)=env.mp_derivs(); gc=2/(-h1)
    c1=gc**2/2*h2**2+gc/3*h3
    return float(q0),float(-h1),float(gc),float(c1),float(h2),float(h3)

# ------------------------------------------------------------------ deterministic skeleton (d=0)
def skeleton_orbit(env,g,n_burn=20000,n_keep=64,q0=0.3):
    h=lambda q:(1-env.alpha)-env.F(q,q); q=q0
    for _ in range(n_burn): q=q+g*h(q)
    out=[]
    for _ in range(n_keep): q=q+g*h(q); out.append(q)
    return np.array(out)
def lyapunov(env,g,n_burn=2000,n=6000,q0=0.3,eps=1e-6):
    h=lambda q:(1-env.alpha)-env.F(q,q); q=q0
    for _ in range(n_burn): q=q+g*h(q)
    s=0.0
    for _ in range(n):
        d=1+g*(-(norm.pdf((logit(q)-(env.a-env.b*q))/env.sig)/env.sig)*(1/(q*(1-q))+env.b)) 
        s+=np.log(abs(d)); q=q+g*h(q)
    return s/n

# ------------------------------------------------------------------ linear delay system  u_{t+1}=(1-gt)u_t - gt*rho*u_{t-d}
def companion(gt,rho,d):
    n=d+1; A=np.zeros((n,n)); A[0,0]=1-gt; A[0,d]-=gt*rho
    for i in range(1,n): A[i,i-1]=1
    return A
def spec_radius(gt,rho,d): return float(max(abs(np.linalg.eigvals(companion(gt,rho,d)))))
def gt_crit(rho,d,gmax=4.0,n=4000):
    """first gain at which a root reaches the unit circle (scan + bisection)"""
    gs=np.linspace(1e-4,gmax,n); prev=gs[0]
    for g in gs:
        if spec_radius(g,rho,d)>=1-1e-12:
            lo,hi=prev,g
            for _ in range(60):
                m=(lo+hi)/2
                if spec_radius(m,rho,d)>=1-1e-12: hi=m
                else: lo=m
            return hi
        prev=g
    return np.inf
def gt_crit_general(rhos,gmax=4.0,n=4000):
    """distributed lag: u_{t+1}=(1-gt)u_t - gt*sum_k rhos[k-1] u_{t-k}, k=1..D"""
    D=len(rhos)
    def sr(g):
        A=np.zeros((D+1,D+1)); A[0,0]=1-g; A[0,1:]-=g*np.asarray(rhos)
        for i in range(1,D+1): A[i,i-1]=1
        return max(abs(np.linalg.eigvals(A)))
    gs=np.linspace(1e-4,gmax,n); prev=gs[0]
    for g in gs:
        if sr(g)>=1-1e-12:
            lo,hi=prev,g
            for _ in range(50):
                m=(lo+hi)/2
                if sr(m)>=1-1e-12: hi=m
                else: lo=m
            return hi
        prev=g
    return np.inf
def sr_general(g,rhos):
    D=len(rhos); A=np.zeros((D+1,D+1)); A[0,0]=1-g; A[0,1:]-=g*np.asarray(rhos)
    for i in range(1,D+1): A[i,i-1]=1
    return max(abs(np.linalg.eigvals(A)))
def boundary_curve(d,n=4000):
    """exact stability boundary for d>=1 (Hopf/NS branch) parametrised by omega:
       rho(w) = -cos(w/2)/cos((2d+1)w/2),   gt(w) = -2 cos((2d+1)w/2) sin(w/2)/sin(d w)"""
    w=np.linspace(1e-4,np.pi-1e-4,n); c=np.cos((2*d+1)*w/2)
    with np.errstate(all='ignore'):
        rho=-np.cos(w/2)/c; gt=-2*c*np.sin(w/2)/np.sin(d*w)
    m=(rho>0)&(gt>0)&np.isfinite(rho)&np.isfinite(gt); return w[m],rho[m],gt[m]

# ------------------------------------------------------------------ controllers (QT form). state dict st
def make_fixed(g): return lambda q,y,miss,t,st: q+g*(miss-ALPHA)
def make_decay(g0): return lambda q,y,miss,t,st: q+g0/np.sqrt(t)*(miss-ALPHA)

def run(env_fn, ctrl, T, seed, q0=0.3, d_fn=None, dither=0.0, record=False):
    """env_fn(t)->Env (possibly regime switching); d_fn(t)-> delay. ctrl(q,y,miss,t,st)->q_next (uses st for memory).
       dither: std of iid Gaussian perturbation added to the *deployed* threshold (probing); controllers see deployed q."""
    rng=np.random.default_rng(seed); st={}; q=q0
    dep=[q0]*(1+64)           # deployed history (index -1 = current)
    out=dict(cov=[],pin=[],q=[],qstar=[],g=[],reg=[])
    for t in range(1,T+1):
        env=env_fn(t); d=env.d
        dsd=st.get('dither',dither); eps=dsd*rng.standard_normal() if dsd>0 else 0.0
        qd=min(1-1e-6,max(1e-6,q+eps)); dep.append(qd); dep.pop(0) if len(dep)>200 else None
        qresp=dep[-1-d] if len(dep)>d else qd
        y=env.draw(qresp,rng); miss=float(y>qd)
        out['cov'].append(1-miss); out['pin'].append((y-qd)*((1-ALPHA)-(y<qd)))
        out['q'].append(q); out['qstar'].append(env.qstar)
        st['eps']=eps; st['qd']=qd
        q=ctrl(q,y,miss,t,st); q=min(1.0,max(0.0,q))
        out['g'].append(st.get('g',np.nan))
    return {k:np.array(v) for k,v in out.items()}

# ------------------------------------------------------------------ second performative family (heavy-tailed, multiplicative)
from scipy.stats import t as _tdist
class EnvT(Env):
    """log y = a - b*q_{t-d} + sigma*T_nu  (Student-t, heavy right tail), y clipped to [0,1].  rho = theta/f = b*q* exactly.
       Parametrised by (q*, rho, sigma, nu, d): b = rho/q*, a = log q* + b q* - sigma*t_inv(1-alpha)."""
    def __init__(s, qstar=0.1, rho=1.0, sig=0.8, nu=3.0, d=0, alpha=ALPHA):
        s.sig,s.nu,s.d,s.alpha=sig,nu,d,alpha; s.b=rho/qstar; s.a=np.log(qstar)+s.b*qstar-sig*_tdist.ppf(1-alpha,nu)
        s.qstar=qstar; z=(np.log(qstar)-s.a+s.b*qstar)/sig; s.z=z
        s.f=_tdist.pdf(z,nu)/(sig*qstar); s.theta=_tdist.pdf(z,nu)*s.b/sig; s.K=s.f+s.theta; s.rho=s.theta/s.f
    def F(s,y,q): return _tdist.cdf((np.log(y)-s.a+s.b*q)/s.sig,s.nu)
    def draw(s,q_resp,rng): return float(min(1.0,np.exp(s.a-s.b*q_resp+s.sig*rng.standard_t(s.nu))))
