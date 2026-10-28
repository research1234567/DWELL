import json, numpy as np, warnings; warnings.filterwarnings("ignore")
from collections import deque
import real_replay as rr
from real_replay import *
from core import gt_crit
def eg_check(y,rho,d,cap):
    s=10*y[:W].mean()+1e-12; z=y/s; q0=float(np.quantile(z[:W],TAU))+1e-9; kappa=rho/q0
    buf=deque(list(z[:W]),maxlen=W); a=np.full(len(GAMS),ALPHA); w=np.ones(len(GAMS))/len(GAMS); dep=[q0]*(d+2)
    ge=np.minimum(GAMS,cap if cap is not None else np.inf); gmin=ge.min(); S=0.0; Vm=0; G=np.inf; n=len(z)-W; phys=0
    for i in range(n):
        arr=np.fromiter(buf,float); qi=np.quantile(arr,1-np.clip(a,0,1)); al=float(w@a); Gt=float(w@ge); G=min(G,Gt)
        if al>1: q=0.0; err=1.0; forced=True
        else: q=float(np.quantile(arr,1-max(al,0.0))); err=None; forced=False
        dep.append(q); yt=z[W+i]*np.exp(np.clip(-kappa*(dep[-1-d]-q0),-5,5)) if rho>0 else z[W+i]
        if err is None: err=float(yt>q)
        S+=ALPHA-err; Vm+=int(al<0 and err==1); phys+=float(yt>q)
        loss=(yt-qi)*(TAU-(yt<qi)); loss=loss-loss.min(); w=w*np.exp(-ETA*loss); w/=w.sum(); w=(1-SHARE)*w+SHARE/len(GAMS)
        a=a+ge*(ALPHA-err); buf.append(yt)
    up=(1-ALPHA)/G+ALPHA; lo=-(ALPHA/G+(1-ALPHA)+(1-ALPHA)*Vm); up0=(1-ALPHA)/gmin+ALPHA; lo0=-(ALPHA/gmin+(1-ALPHA)+(1-ALPHA)*Vm)
    return dict(n=n,S=S,Vm=Vm,G=G,gmin=gmin,up=up,lo=lo,up_old=up0,lo_old=lo0,ok=bool(lo<=S<=up),phys_dev=abs(1-phys/n-0.9))
Y=series(); rows=[]
for rho,d in [(0.0,1),(1.0,6),(2.5,6),(2.5,12)]:
    rb=2*rho if rho>0 else 0.2
    cap=0.5*min(gt_crit(max(rb,1e-3),k,gmax=2.0,n=500) for k in range(1,max(d*2,2)+1))
    for a in sorted(Y): r=eg_check(Y[a],rho,d,cap); r.update(rho=rho,d=d,asset=a); rows.append(r)
    print(f"rho={rho} d={d}: max(Vp+Vm)={max(r['Vm'] for r in rows if r['rho']==rho and r['d']==d)}  all ok={all(r['ok'] for r in rows if r['rho']==rho and r['d']==d)}",flush=True)
sel=[r for r in rows]; 
print("n cells",len(sel),"| inequality holds in all:",all(r['ok'] for r in sel),"| max V-",max(r['Vm'] for r in sel),"| max |S|",max(abs(r['S']) for r in sel),"| min G",min(r['G'] for r in sel),"| max bound used",max(max(r['up'],-r['lo']) for r in sel),"| n eval",sel[0]['n'])
json.dump(rows,open('out/real_guarantee.json','w'),default=float)
