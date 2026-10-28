"""Tight finite-buffer EG-DtACI.  Conventions: alpha_t>1 -> deploy ZERO buffer, err forced to 1 (outcomes >=0)  => upward violations impossible.
   alpha_t<0 -> deploy window max (finite), err = 1{y>q}  => downward violations V- counted.
   Claim: T|mean(err)-alpha| = |S_{T+1}| <= max{ (1-alpha)/G + alpha ,  alpha/G + (1-alpha) + (1-alpha)V- },  G = min_t Gamma_t (realised).
   Physical miscoverage (zero buffer covers y=0) is <= algorithmic miscoverage."""
import numpy as np, json
from collections import deque
ALPHA=0.1; GAMS=np.array([0.005,0.01,0.02,0.05,0.1,0.2]); W=100; TAU=0.9
def run(y,cap,share=1e-3):
    buf=deque(list(y[:W]),maxlen=W); a=np.full(len(GAMS),ALPHA); w=np.ones(len(GAMS))/len(GAMS); ge=np.minimum(GAMS,cap); gmin=ge.min()
    S=0.0; Vm=0; G=np.inf; phys=0.0; n=0
    for t in range(W,len(y)):
        arr=np.fromiter(buf,float); al=float(w@a); Gt=float(w@ge); G=min(G,Gt)
        if al>1: q=0.0; err=1.0
        else: q=float(np.quantile(arr,1-max(al,0.0))); err=float(y[t]>q)
        qi=np.quantile(arr,1-np.clip(a,0,1))
        S+=ALPHA-err; Vm+=int(al<0 and err==1); phys+=float(y[t]>q); n+=1
        loss=(y[t]-qi)*(TAU-(y[t]<qi)); loss=loss-loss.min(); w=w*np.exp(-60*loss); w/=w.sum(); w=(1-share)*w+share/len(GAMS)
        a=a+ge*(ALPHA-err); buf.append(y[t])
    up=(1-ALPHA)/G+ALPHA; lo=-(ALPHA/G+(1-ALPHA)+(1-ALPHA)*Vm)
    up0=(1-ALPHA)/gmin+ALPHA; lo0=-(ALPHA/gmin+(1-ALPHA)+(1-ALPHA)*Vm)
    return dict(S=S,lo=lo,up=up,Vm=Vm,G=G,gmin=gmin,n=n,phys_excess=(phys-(n*ALPHA-S))) # phys misses minus algorithmic misses (<=0)
if __name__=="__main__":
    rng=np.random.default_rng(1); R=[]
    gens={'uniform':lambda T:rng.uniform(size=T),'beta(5,1)':lambda T:rng.beta(5,1,T),'heavy tail':lambda T:np.abs(rng.standard_cauchy(T)),'zeros+spikes':lambda T:np.where(rng.random(T)<0.6,0,rng.exponential(1,T)*np.where(rng.random(T)<0.02,30,1)),
     'alternating':lambda T:np.tile([0.0,50.0],T//2),'regime shift':lambda T:np.r_[rng.uniform(0,.3,T//2),rng.uniform(10,12,T-T//2)],'ramp':lambda T:np.linspace(0,100,T)+np.abs(rng.normal(0,1,T)),'all equal':lambda T:np.ones(T),'all zero':lambda T:np.zeros(T)}
    for name,g in gens.items():
        for cap in [np.inf,0.03,0.01]:
            for rep in range(8):
                T=int(rng.integers(600,2500)); r=run(g(T),cap); r['gen']=name; r['ok']=bool(r['lo']<=r['S']<=r['up']); r['Tn']=T; R.append(r)
    bad=sum(not r['ok'] for r in R); withV=sum(r['Vm']>0 for r in R)
    ratio=np.median([ (r['up']-r['lo'])/ ((1-ALPHA)/r['gmin']+ALPHA+ALPHA/r['gmin']+(1-ALPHA)+(1-ALPHA)*r['Vm']) for r in R])
    print(f"{len(R)} runs | bound violated in {bad} | runs with downward violations {withV} (max {max(r['Vm'] for r in R)}) | worst |S|/bound {max(max(r['S']/r['up'] if r['S']>0 else 0, r['S']/r['lo'] if r['S']<0 else 0) for r in R):.3f} | median width ratio tight/old {ratio:.2f} | mean G/gmin {np.mean([r['G']/r['gmin'] for r in R]):.2f}")
    worst=max(max(r['S']/r['up'] if r['S']>0 else 0, r['S']/r['lo'] if r['S']<0 else 0) for r in R)
    nov=sum(not (-(ALPHA/r['G']+(1-ALPHA))<=r['S']<=(1-ALPHA)/r['G']+ALPHA) for r in R)
    json.dump(dict(n=len(R),bad=bad,withV=withV,maxV=max(r['Vm'] for r in R),ratio=float(ratio),Gratio=float(np.mean([r['G']/r['gmin'] for r in R])),worst=float(worst),nov=int(nov)),open('out/test_finite2.json','w'))
    print('V-free tight bound violated in',nov,'of',len(R))
