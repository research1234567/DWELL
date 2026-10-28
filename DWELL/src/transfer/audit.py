import json,sys,time
import numpy as np
from scipy.special import expit
from scipy.optimize import brentq
import run_transfer as rt
from core import Env
out={}
# Quantile algorithm exactly matches np.quantile, including clamped levels.
rng=np.random.default_rng(8);x=rng.random((2,3,7,100));a=rng.uniform(-.4,1.4,(2,3,7));idx=(1-np.clip(a,0,1))*99;lo=np.floor(idx).astype(int);hi=np.ceil(idx).astype(int);s=np.sort(x,axis=-1);q=np.take_along_axis(s,lo[...,None],-1)[...,0]+(idx-lo)*(np.take_along_axis(s,hi[...,None],-1)[...,0]-np.take_along_axis(s,lo[...,None],-1)[...,0]);ref=np.array([np.quantile(xx,1-np.clip(aa,0,1)) for xx,aa in zip(x.reshape(-1,100),a.ravel())]).reshape(a.shape)
out['quantile_max_abs_difference']=float(max(abs(q-ref).ravel()))
# Independent dense eigenvalue check, severe W100 (strongest transfer shift).
r=json.load(open('results.json'));s=r['results'][2];rho=s['rho'];d=s['d'];W=100;ec=s['windows'][0]['population']['first_loss'];evrows=[]
for mult in [.999,1.001]:
 eta=ec*mult;N=W+d+1;A=np.zeros((N,N));A[0,0]=1-eta;A[0,d]-=rho*(eta+1/W);A[0,W+d]+=rho/W;A[1:,:-1]=np.eye(N-1);ev=np.linalg.eigvals(A);evrows.append(dict(multiplier=mult,radius=float(max(abs(ev))),schur_stable=rt.stable(eta,rho,d,W)))
out['dense_eigen_check']=evrows
# Numerical full population quantile initial derivatives check across all grid cells already stored.
out['jacobian_max_abs_errors_by_epsilon']={str(eps):max(abs(w['jacobian_check'][i]['finite_difference']-w['jacobian_check'][i]['predicted']) for s in r['results'] for w in s['windows']) for i,eps in enumerate([1e-5,1e-6,1e-7])}
# Direct nonlinear agreement must improve on 10x smaller initial perturbation, away from roundoff floor.
rows=[]
for s in r['results']:
 for ratio in rt.RAT:
  a=[x for x in s['direct'] if x['ratio']==ratio]
  rows.append(dict(scenario=s['name'],ratio=float(ratio),small_relative_error=abs(a[0]['nonlinear_norm']/a[0]['predicted_norm']-1),large_relative_error=abs(a[1]['nonlinear_norm']/a[1]['predicted_norm']-1),predicted=a[0]['predicted_norm']))
out['direct_convergence']=rows
json.dump(out,open('audit.json','w'),indent=2);print(json.dumps(out,indent=2))
# Scalar reference versus vectorized implementation with the identical innovations.
e=Env(a=-1,b=48,d=12);gc=rt.gt_crit(e.rho,e.d);W=100;N=8;T=1100
import tempfile
original_root=rt.ROOT
rt.ROOT=rt.Path(tempfile.mkdtemp(prefix='transfer-audit-'))
rt.empirical(e,gc,W,N,T)
stored=np.load(rt.ROOT/'empirical_12_100.npz')
rt.ROOT=original_root
rng=np.random.default_rng(47001);z0=rng.standard_normal((N,W));innov=rng.standard_normal((T,N));buf=list(expit(e.a-e.b*e.qstar+e.sig*z0[3]));hist=[e.qstar]*(e.d+1);level=.1;qs=[]
for t in range(T):
 q=float(np.quantile(buf,1-np.clip(level,0,1)));hist.append(q);y=expit(e.a-e.b*hist[-1-e.d]+e.sig*innov[t,3]);level+=.5*gc*(.1-float(y>q));buf[t%W]=y;qs.append(q)
out['scalar_reference_mse']=float(np.mean((np.array(qs[1000:])-e.qstar)**2));out['vectorized_mse']=float(stored['mse'][1,0,3]);out['scalar_reference_abs_difference']=abs(out['scalar_reference_mse']-out['vectorized_mse'])
json.dump(out,open('audit.json','w'),indent=2);print('scalar audit',out['scalar_reference_abs_difference'])
