"""Bounded independent population audit. Does not run stochastic experiments."""
import json,sys,time,platform
from pathlib import Path
import numpy as np
import scipy
from scipy.special import ndtr,logit
from scipy.optimize import brentq
from numpy.polynomial import chebyshev as C
import mpmath as mp
P=Path(__file__).parent
src=P.parent
sys.path.insert(0,str(src/'conformal_feedback_revised/code'))
from core import Env
mp.mp.dps=70

def coeff(eta,rho,d,W):
 a=np.zeros(W+d+2);a[0]=1;a[1]+=eta-1;a[d+1]+=rho*(eta+1/W);a[-1]-=rho/W;return a

def schur_mp(eta,rho,d,W):
 r=mp.mpf(str(rho));e=mp.mpf(str(eta));a=[mp.mpf(0)]*(W+d+2);a[0]=1;a[1]+=e-1;a[d+1]+=r*(e+mp.mpf(1)/W);a[-1]-=r/W
 margin=mp.mpf(1)
 while len(a)>1:
  k=a[-1]/a[0];margin=min(margin,1-abs(k))
  if abs(k)>=1:return False,float(margin)
  a=[(a[j]-k*a[-1-j])/(a[0]*(1-k*k)) for j in range(len(a)-1)]
 return True,float(margin)

def all_crossings(r,d,W):
 # sin(k*w)/sin(w)=U_(k-1)(cos(w)); expand in T basis.
 a=np.zeros(W+d)
 for k,v in [(1,1),(d+1,r),(d,-r*(1+1/W)),(W+d,r/W),(W,r*r/W)]:
  if not k:continue
  n=k-1
  for j in range(n%2,n+1,2):a[j]+=v*(1 if j==0 else 2)
 roots=C.chebroots(a);cand=[];maxres=0.
 for z in roots:
  if abs(z.imag)>1e-8 or not -1<z.real<1:continue
  w=float(np.arccos(z.real));D=1+r*np.exp(-1j*d*w)
  if abs(D)<1e-10:continue
  eta=-(np.exp(1j*w)-1+r/W*(np.exp(-1j*d*w)-np.exp(-1j*(W+d)*w)))/D
  if eta.real>0:
   maxres=max(maxres,abs(eta.imag));cand.append((float(eta.real),w))
 s=(-1)**d;den=1+r*s
 if abs(den)>1e-10:
  e=(2-r/W*s*(1-(-1)**W))/den
  if e>0:cand.append((e,float(np.pi)))
 return sorted(cand),maxres

def jac_direction(e,W,eta,h,eps):
 def step(q):
  # history newest first, current q[0], next mixture drops q[W+d], adds q[d].
  H=lambda x,hh:np.mean(ndtr((logit(x)-e.a+e.b*hh)/e.sig))
  a=1-e.alpha-H(q[0],q[e.d+1:e.d+W+1]);an=e.alpha+a+eta*(e.F(q[0],q[e.d])-(1-e.alpha))
  return brentq(lambda y:H(y,q[e.d:e.d+W])-(1-an),1e-10,1-1e-10,xtol=1e-15)
 return (step(e.qstar+eps*h)-step(e.qstar-eps*h))/(2*eps)

start=time.time();raw=json.load(open(src/'transfer_results/results.json'));out={'scope':'deterministic closure only; numerical diagnostics not proof','versions':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'mpmath':mp.__version__},'cells':[]}
rng=np.random.default_rng(28301)
for s in raw['results']:
 e=Env(a=s['a'],b=s['b'],d=s['d'])
 for w in s['windows']:
  W=w['W'];r=e.rho;d=e.d;ca,res=all_crossings(r,d,W);ec=ca[0][0]
  checks=[]
  for mult in [.99999,1.00001]:
   eta=ec*mult;ev=np.roots(coeff(eta,r,d,W));sm,margin=schur_mp(eta,r,d,W)
   checks.append({'multiplier':mult,'radius':float(max(abs(ev))),'schur70_stable':sm,'schur_min_margin':margin})
  errors=[]
  for k in range(8):
   h=rng.normal(size=W+d+1);h/=max(abs(h));eta=.5*s['threshold_eta_c'];pred=(1-eta)*h[0]-r*(eta+1/W)*h[d]+r/W*h[-1]
   errors.append(abs(jac_direction(e,W,eta,h,1e-6)-pred))
  row={'name':s['name'],'W':W,'rho':r,'d':d,'all_positive_candidates':ca,'imaginary_residual':res,'prior_first_loss':w['population']['first_loss'],'independent_first_candidate':ec,'abs_difference':abs(ec-w['population']['first_loss']),'nearby_checks':checks,'max_central_jacobian_error_8directions':max(errors)}
  out['cells'].append(row);print(s['name'],W,ec,checks,flush=True)
# Strong exact counterexamples, floating values provided only as illustration.
out['counterexample_W1_d0_rho09_eta02']={'scalar_root':1-.2*1.9,'rolling_roots':np.roots(coeff(.2,.9,0,1)).tolist()}
out['counterexample_W2_d1_rho15_eta0_H_roots']=[{'real':float(z.real),'imag':float(z.imag),'abs':float(abs(z))} for z in np.roots([1,0,.75,.75])]
out['elapsed_seconds']=time.time()-start
json.dump(out,open(P/'verification.json','w'),indent=2)
print('elapsed',out['elapsed_seconds'])
