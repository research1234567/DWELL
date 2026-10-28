"""Finite-buffer EG-DtACI: threshold = window quantile at clipped level (clip to window range).  Claim:
   -(alpha/g + (1-alpha) + (1-alpha)Vm) <= S_{T+1} <= (1-alpha)/g + alpha + alpha*Vp,
   S=sum(alpha-err), g=gamma_min, Vp=#{alpha_t>1, err=0}, Vm=#{alpha_t<0, err=1}.  Test on adversarial/random sequences."""
import numpy as np
from collections import deque
ALPHA=0.1; GAMS=np.array([0.005,0.01,0.02,0.05,0.1,0.2]); W=100; TAU=0.9
def run(y,cap,rng):
    buf=deque(list(y[:W]),maxlen=W); a=np.full(len(GAMS),ALPHA); w=np.ones(len(GAMS))/len(GAMS); ge=np.minimum(GAMS,cap); gmin=ge.min()
    S=0.0; Vp=Vm=0
    for t in range(W,len(y)):
        arr=np.fromiter(buf,float); al=float(w@a); q=float(np.quantile(arr,1-min(1,max(0,al)))); qi=np.quantile(arr,1-np.clip(a,0,1))
        err=float(y[t]>q); S+=ALPHA-err
        if al>1 and err==0: Vp+=1
        if al<0 and err==1: Vm+=1
        loss=(y[t]-qi)*(TAU-(y[t]<qi)); loss=loss-loss.min(); w=w*np.exp(-60*loss); w/=w.sum(); w=0.999*w+0.001/len(GAMS)   # arbitrary adapted weights
        a=a+ge*(ALPHA-err); buf.append(y[t])
    up=(1-ALPHA)/gmin+ALPHA+ALPHA*Vp; lo=-(ALPHA/gmin+(1-ALPHA)+(1-ALPHA)*Vm)
    return S,lo,up,Vp,Vm,gmin
rng=np.random.default_rng(0); n=0; viol_cases=0; worst=0; maxV=0; bad=0
gens={'uniform':lambda T:rng.uniform(size=T),'beta(5,1)':lambda T:rng.beta(5,1,T),'heavy tail':lambda T:np.abs(rng.standard_cauchy(T)),'zeros+spikes':lambda T:np.where(rng.random(T)<0.6,0,rng.exponential(1,T)*np.where(rng.random(T)<0.02,30,1)),
 'alternating':lambda T:np.tile([0.0,50.0],T//2),'regime shift':lambda T:np.r_[rng.uniform(0,.3,T//2),rng.uniform(10,12,T-T//2)],'ramp':lambda T:np.linspace(0,100,T)+rng.normal(0,1,T),'all equal':lambda T:np.ones(T)}
for name,g in gens.items():
    for cap in [np.inf,0.03,0.01]:
        for rep in range(8):
            T=int(rng.integers(600,2500)); y=g(T); S,lo,up,Vp,Vm,gmin=run(y,cap,rng); n+=1; base_up=(1-ALPHA)/gmin+ALPHA; base_lo=-(ALPHA/gmin+(1-ALPHA)); nov_bad=globals().get('nov_bad',0)+(0 if base_lo<=S<=base_up else 1); globals()['nov_bad']=nov_bad
            ok=(lo<=S<=up); bad+=0 if ok else 1; viol_cases+= (Vp+Vm>0); maxV=max(maxV,Vp+Vm); worst=max(worst,max(S/up if S>0 else 0,S/lo if S<0 else 0))
import json; json.dump(dict(n=n,nov_bad=globals()['nov_bad'],bad=bad,viol_cases=viol_cases,maxV=maxV,worst=worst),open('out/test_finite.json','w')); print(f"V-free bound (Thm 7) violated in {globals()['nov_bad']} of {n} runs"); print(f"{n} runs | inequality violated in {bad} | runs with violation events {viol_cases} | max violation count {maxV} | worst |S|/bound = {worst:.3f}")
