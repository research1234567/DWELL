import json, numpy as np
from core import *
N={}
# boundary parametrisation: some root on the unit circle
err=0
for d in [1,2,3,6,12]:
    w,r,g=boundary_curve(d,3000)
    for k in range(0,len(w),max(1,len(w)//60)):
        ev=np.linalg.eigvals(companion(g[k],r[k],d)); err=max(err,float(np.min(abs(abs(ev)-1))))
N['param_check_err']=err
# where is the Lyapunov exponent positive?
e=Env(); gs=np.linspace(0.74,1.2,93); pos=[(round(g,3),round(lyapunov(e,g,n_burn=800,n=1500),3)) for g in gs]
N['lyap_scan']=[p for p in pos if p[1]>0][:6]; N['lyap_first_pos']=next((p[0] for p in pos if p[1]>0),None)
# coverage bound: unprojected vs projected recursion
def cov_run(yseq,g,proj,q0=0.5):
    q=q0;miss=0;qmin=q;qmax=q
    for y in yseq:
        m=float(y>q); miss+=m; q=q+g*(m-ALPHA)
        if proj: q=min(1,max(0,q))
        qmin=min(qmin,q);qmax=max(qmax,q)
    return miss/len(yseq),qmin,qmax
T=2000;g=0.05; rr=np.random.default_rng(5)
cases=[("$y\\equiv0$",np.zeros(T)),("$y\\equiv1$",np.ones(T)),("alternating 0/1",np.tile([0.,1.],T//2)),("uniform",rr.uniform(size=T)),("Beta(5,1)",rr.beta(5,1,T)),("regime shift",np.r_[rr.uniform(0,.3,T//2),rr.uniform(.7,1,T//2)])]
rows=[]
for name,ys in cases:
    mp_,_,_=cov_run(ys,g,True); mu,lo,hi=cov_run(ys,g,False)
    rows.append((name,mp_,mu,lo,hi))
N['cov_rows']=rows; N['cov_T']=T; N['cov_g']=g; N['cov_B_unproj']=(1+g)/(g*T); N['cov_B_naive']=1/(g*T)
# randomized sweep for the unprojected bound
rng=np.random.default_rng(1); worst=0
for _ in range(300):
    Tt=int(rng.integers(200,3000)); gg=float(rng.choice([0.005,0.02,0.05,0.2,0.8])); ys=rng.beta(rng.uniform(.2,5),rng.uniform(.2,5),Tt)
    if rng.random()<.3: ys=np.round(ys)
    mu,_,_=cov_run(ys,gg,False); worst=max(worst,abs(mu-ALPHA)*gg*Tt/(1+gg))
N['cov_sweep_worst_ratio']=worst
json.dump(N,open('out/sec_C2.json','w'),indent=1,default=float)
print(N)
